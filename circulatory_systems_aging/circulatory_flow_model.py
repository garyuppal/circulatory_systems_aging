import torch
import numpy as np
from utils import pickle_save, get_device
from pathlib import Path
import configparser
from datetime import datetime


class FlowAgingModelTorch:
    def __init__(self, 
                 diff_goods,
                 diff_toxins,
                 vel, 
                 decay_goods,
                 decay_toxins, 
                 sec_rate_goods,
                 sec_rate_toxins,
                 hill_k_goods,
                 hill_k_toxins,
                 death_scale_alpha,
                 mixing_gamma, 
                 n_pop = 2,
                 xmin=0, 
                 xmax=100, 
                 nx=1000,
                 dt=0.0001,
                 device=get_device()):
        # model constants
        self.diff_goods = diff_goods
        self.diff_toxins = diff_toxins
        self.vel = vel
        self.decay_goods = decay_goods
        self.decay_toxins = decay_toxins
        self.sec_rate_goods = sec_rate_goods
        self.sec_rate_toxins = sec_rate_toxins
        self.hill_k_goods = hill_k_goods
        self.hill_k_toxins = hill_k_toxins
        self.death_scale_alpha = death_scale_alpha
        self.mixing_gamma = mixing_gamma
        self.dt = dt #* time step
        self.n_pop = n_pop #* number of organisms
        self.nx = nx
        self.device = device
        print(f"Using device: {self.device}")

        # cells
        self.state = torch.ones((n_pop, nx), device=self.device) # one cell per spatial disc cell
        self.cell_xpositions = torch.linspace(xmin,xmax,nx).to(self.device) #* cell positions
        self.dx = self.cell_xpositions[1] - self.cell_xpositions[0] #* spatial disc cell size
        # map field locations to cell positions...
        self.field_to_cell_idx = np.floor((self.cell_xpositions/self.dx).cpu().detach().clone().numpy()).astype(int) + 1 # +1 shift for ghost cells

        # fields
        self.goods_field = torch.zeros((n_pop,nx+2), device=self.device) #* including 'ghost' nodes
        self.toxins_field = torch.zeros((n_pop,nx+2), device=self.device) #* including 'ghost' nodes

    def secrete_source(self, secrate):
        secfield = torch.zeros_like(self.goods_field)
        secfield[:,1:-1] = secrate*self.state/self.dx
        return secfield

    def update_ghosts(self, field):
        #* assuming periodic boundary conditions
        field[:,0] = field[:,-2]  # First ghost cell
        field[:,-1] = field[:,1]  # Last ghost cell
        return field

    def laplacian(self, field):
        temp = torch.zeros_like(field)
        mmx = (1.0/self.dx)**2
        L = field[:,:-2] 
        R = field[:,2:]
        C = field[:,1:-1] 
        temp[:,1:-1] = mmx*(L+R- 2*C)
        return temp
    
    def advect(self, field, vel):
        mmx = (1.0/self.dx)
        # Upwind scheme
        temp = torch.zeros_like(field)
        temp[:,1:-1] = -vel * mmx * (field[:,1:-1] - field[:,:-2])
        return temp

    def update_field(self,field,diff,vel,decay,secrate):
        # add sources
        field_secrete = self.secrete_source(secrate)
        # apply boundary conditions
        field = self.update_ghosts(field)
        # diffuse, and advect
        diff_field = diff*self.laplacian(field)
        adv_field = self.advect(field, vel)
        # update field
        field = field + (field_secrete + diff_field + adv_field - decay*field)*self.dt
        return field

    def update_goods(self):
        self.goods_field = self.update_field(self.goods_field,
                                             self.diff_goods,
                                             self.vel,
                                             self.decay_goods,
                                             self.sec_rate_goods)

    def update_toxins(self):
        self.toxins_field = self.update_field(self.toxins_field,
                                              self.diff_toxins,
                                              self.vel,
                                              self.decay_toxins,
                                              self.sec_rate_toxins)
    
    def update_cells(self):
        # stochastically kill cells based on death probabilities computed from fields
        goods_term = (1.0/(1.0 + torch.pow(self.goods_field[:,self.field_to_cell_idx],self.hill_k_goods)))
        toxins_term = ((torch.pow(self.toxins_field[:,self.field_to_cell_idx],self.hill_k_toxins))/(1.0 + torch.pow(self.toxins_field[:,self.field_to_cell_idx],self.hill_k_toxins)))
        hill_terms = self.mixing_gamma*goods_term + (1.0 - self.mixing_gamma)*toxins_term
        death_probs = self.death_scale_alpha*hill_terms*self.dt
        rk = torch.rand(*self.state.shape, device=self.device)
        self.state[rk < death_probs] = 0

    def calc_surv_mort(self, age_delay, times, pvt):
        """
        Calculate survival and mortality metrics over time.
        
        Parameters:
        - age_delay: float, time before aging starts
        - times: np.ndarray, array of time points
        - pvt: np.ndarray, shape (ntimes, n_pop), population count over time
        
        Returns:
        - t95: float, time when only 5% of population is alive
        - surv: np.ndarray, survival count at each time step
        - mort: np.ndarray, estimated mortality rate at each time step
        - times_mort: np.ndarray, time points corresponding to mortality rates
        """
        num_organisms = self.n_pop
        alive_threshold = 0.01 * self.nx  # Threshold for considering a population "alive"
        
        # Compute survival count over time
        surv = (pvt > alive_threshold).sum(axis=1)

        # Find the first time each organism dies
        d_idxs = np.argmax(pvt == 0, axis=0)  # Index of first zero for each organism
        dtimes = times[d_idxs] - age_delay  # Time of death adjusted for age delay

        # Compute t95 (time when only 5% of organisms are alive)
        surv95_threshold = 0.05 * num_organisms
        surviving_indices = np.where(surv > surv95_threshold)[0]
        
        if surviving_indices.size > 0:
            t95 = times[surviving_indices[-1]] - age_delay  # Last timepoint with >5% survival
        else:
            t95 = np.inf  # If survival never drops below 5%, set to infinity
        
        # # Compute mortality rate
        # delta_t = np.mean(np.diff(times))  # Average time step
        # dtidx = max(int(np.std(dtimes / delta_t) * 0.25), 1)  # Smoothing window

        # Compute mortality rate
        delta_t = np.mean(np.diff(times))  # Average time step
        t95idx = int(t95/delta_t)
        # check if bigger than 0.25*t95idx
        dtidx = min(int(np.std(dtimes) / delta_t * 0.25), int(0.25*t95idx))
        # check if smaller than 1
        dtidx = max(dtidx, 1)  # Smoothing window

        
        astart = np.argmax(times > age_delay)  # Start index after aging begins
        at95 = np.argmax(times > (t95 + age_delay)) if np.isfinite(t95) else len(times)
        
        s95 = surv[astart:at95]
        nt_mort = len(s95) - dtidx
        
        if nt_mort <= 0:
            return t95, surv, np.array([]), np.array([])  # No meaningful mortality data

        mort = np.zeros(nt_mort)
        times_mort = np.zeros(nt_mort)
        
        for i in range(nt_mort):
            prev_surv = s95[i]
            next_surv = s95[i + dtidx]
            
            if prev_surv > 0:
                mort[i] = -(next_surv - prev_surv) / (prev_surv * (dtidx * delta_t))
            else:
                mort[i] = 0  # Avoid division by zero
            
            times_mort[i] = times[i + astart] - age_delay
        
        return t95, surv, mort, times_mort

    
    def age(self, runtime, age_delay=1.0, save_steps=1, save_full=False, 
            calc_aging=True, update_fields=True):
        ntimesave = int((runtime/self.dt)/save_steps) + 2
        if save_full:
            goods = np.zeros((ntimesave, self.n_pop, self.nx+2))
            toxins = np.zeros((ntimesave, self.n_pop, self.nx+2))
            cells_v_time = np.zeros((ntimesave, self.n_pop, self.nx))
        k = 0
        # pop_v_time = np.zeros((ntimesave, self.n_pop))
        pop_v_time = []
        times = []
        t = 0
        i = 0
        while t < runtime:
            if update_fields:
                self.update_goods()
                self.update_toxins()
            if t > age_delay:
                self.update_cells()
            t += self.dt
            if i % save_steps == 0:
                # pop_v_time[k,:] = self.state.sum(dim=1).cpu().detach().clone().numpy()
                pop_v_time.append(self.state.sum(dim=1).cpu().detach().clone().numpy())
                if save_full:
                    goods[k,:,:] = self.goods_field.cpu().detach().clone().numpy()
                    toxins[k,:,:] = self.toxins_field.cpu().detach().clone().numpy()
                    cells_v_time[k,:] = self.state.cpu().detach().clone().numpy()
                k += 1
                times.append(t)
                print(f"Time: {t}")
            i += 1                    
        pop_v_time = np.array(pop_v_time)
        times = np.array(times)
        if calc_aging:
            t95, surv, mort, times_mort = self.calc_surv_mort(age_delay, times, pop_v_time)
            apvt = np.mean(pop_v_time, axis=1)
            res = {"t95": t95, "apvt": apvt, "surv": surv, "mort": mort, "pop_v_time": pop_v_time, 
                   "times": times, "times_mort": times_mort, "age_delay": age_delay}
        else:
            res = None
        if save_full:
            full_state_v_time = {"goods": goods, "toxins": toxins, "cells": cells_v_time}
        else:
            full_state_v_time = None
        return res, full_state_v_time
    


def run_model(config):
    start_time = datetime.now()
    seed = config.getint("General", "random_seed", fallback=42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    diff_goods = float(config["Goods"]["diffusion"])
    diff_toxins = float(config["Toxins"]["diffusion"])
    vel = float(config["General"]["velocity"])
    decay_goods = float(config["Goods"]["decay"])
    decay_toxins = float(config["Toxins"]["decay"])
    sec_rate_goods = float(config["Goods"]["secretion_rate"])
    sec_rate_toxins = float(config["Toxins"]["secretion_rate"])
    hill_k_goods = float(config["Goods"]["hill_coefficient"])
    hill_k_toxins = float(config["Toxins"]["hill_coefficient"])
    death_scale_alpha = float(config["General"]["death_rate_alpha"])
    mixing_gamma = float(config["General"]["mixing_gamma"])
    n_pop = int(config["General"]["n_population"])
    
    runtime = float(config["General"]["runtime"])
    outpath = Path(config["General"]["outdir"])
    outpath.mkdir(parents=True, exist_ok=True)

    age_delay = config.getfloat("General", "age_delay", fallback=1.0)
    save_steps = config.getint("General", "save_interval", fallback=100)
    dt = config.getfloat("General", "dt", fallback=0.0001)

    save_full = config.getboolean("General", "save_full", fallback=False)

    disable_GPU = config.getboolean("General", "disable_GPU", fallback=False)
    if disable_GPU:
        device = torch.device("cpu")
    else:
        # else use GPU if available
        device = get_device()

    model = FlowAgingModelTorch(diff_goods,
                                diff_toxins,
                                vel,
                                decay_goods,
                                decay_toxins,
                                sec_rate_goods,
                                sec_rate_toxins,
                                hill_k_goods,
                                hill_k_toxins,
                                death_scale_alpha,
                                mixing_gamma,
                                n_pop=n_pop,
                                dt=dt,
                                device=device)
    
    calc_aging = True
    update_fields = True

    if "Debug" in config.sections():
        update_fields = config.get("Debug", "update_fields", fallback=True)
        active_cells = config.get("Debug", "active_cells", fallback=None)
        if active_cells is not None:
            if active_cells == "all":
                model.state[:,:] = 1.0
            else:
                active_cells = [int(x) for x in active_cells.split(",")]
                model.state[:,:] = 0.0
                model.state[:,active_cells] = 1.0
        goods_source = config.getfloat("Debug", "initial_source_concentration_goods", fallback=None)
        toxins_source = config.getfloat("Debug", "initial_source_concentration_toxins", fallback=None)
        for i in range(n_pop):
            idx = torch.cat((torch.tensor([False],device=model.device),(model.state[i,:]>0.5),torch.tensor([False],device=model.device))) # add ghost cell idxs
            if goods_source is not None:
                model.goods_field[i,idx] = goods_source
            if toxins_source is not None:
                model.toxins_field[i,idx] = toxins_source
        calc_aging = config.getboolean("Debug", "calc_aging", fallback=True)

    res, full_state = model.age(runtime, age_delay, save_steps, save_full=save_full, calc_aging=calc_aging, update_fields=update_fields)
    pickle_save(outpath / "results.pkl", res)
    print("saved results to", outpath / "results.pkl")
    if full_state is not None:
        pickle_save(outpath / "full_state_v_time.pkl", full_state)

    # get runtime
    end_time = datetime.now()
    elapsed = end_time - start_time
    # Breakdown into days, hours, minutes, and seconds
    days = elapsed.days
    seconds = elapsed.seconds
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Runtime: {days}d {hours}h {minutes}m {seconds}s")


def main(args):
    # read configuration file
    config = configparser.ConfigParser()
    config.read(args.configfile) 

    # override parameters
    for override in args.overrides:
        key, value = override.split("=")
        section, param = key.split(":")
        if section in config:
            if param in config[section]:
                print(f"Overriding parameter {param} in section {section} with value {value}")
                config[section][param] = value
            else:
                print(f"Warning: {param} is not in the {section} section. Adding it.")
                config[section][param] = value
        else:
            print(f"Warning: {section}:{param} is not in the default configuration. Adding it.")
            config[section] = {}
            config[section][param] = value

    # save the updated configuration
    outpath = Path(config["General"]["outdir"])
    outpath.mkdir(parents=True, exist_ok=True)
    with open(outpath / "config.ini", 'w') as configfile:
        config.write(configfile)

    # run the model
    run_model(config)
    print("\n\nDone!\n\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run circulatory flow model")
    parser.add_argument("--config", type=str, dest='configfile', help="Path to configuration file")
    parser.add_argument("--overrides", type=str, 
                        nargs='+', help="Override parameters in section:parameter=value format.", default=[])
    args = parser.parse_args()
    main(args)
