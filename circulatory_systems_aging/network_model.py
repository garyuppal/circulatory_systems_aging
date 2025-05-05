import numpy as np
import torch 
from utils import pickle_save, get_device
import configparser
from pathlib import Path
import networks as nw
from datetime import datetime


class NetworkAgingModel:
    def __init__(self, 
                 adjacency_matrix,
                 num_organisms,
                 init_nonfunctional,
                 prob_death,
                 prob_repair,
                 device):
        self.device = device
        self.num_nodes = adjacency_matrix.shape[0]
        self.num_organisms = num_organisms
        self.eta = init_nonfunctional
        self.pi_death = prob_death
        self.pi_repair = prob_repair
        self.adjmat = torch.from_numpy(adjacency_matrix).to(dtype=torch.float, device=device)
        self.number_neighbors = self.adjmat.sum(dim=0)
        
        self.init_state()
        self.fraction_alive = None
        self.survival = None
        self.mortality = None
    
    def init_state(self):
        self.state = torch.ones((self.num_organisms, self.num_nodes), dtype=torch.float, device=self.device)
        r = torch.rand(self.num_organisms, self.num_nodes, device=self.device)
        self.state[r < self.eta] = 0

    def calc_mort(self, dtimes, surv):
        surv95_threshold = 0.05*self.num_organisms # timepoint for 5% population
        times = np.arange(len(surv))
        tsurv = times[surv>surv95_threshold] # times upto t95 (where >95% alive)
        if len(tsurv) == 0:
            raise ValueError("no t95!")
        t95 = len(tsurv)
        dt = max(int(np.std(dtimes)*0.25),1)
        s95 = surv[:t95]
        
        nt_mort = len(s95)-dt
        mort = np.zeros(nt_mort)
        for i in range(nt_mort):
            mort[i] = -(s95[i+dt]-s95[i])/(s95[i]*dt)
        return t95, mort

    def age_network(self, n_time, tau=0.01, verbose=-1, save_all=False):
        self.fraction_alive = torch.zeros((n_time, self.num_organisms), dtype=torch.float, device=self.device)
        organism_state = torch.ones((n_time, self.num_organisms), dtype=torch.float, device=self.device)
        
        if save_all is True:
            state_dynamics = np.zeros((n_time, self.num_organisms, self.num_nodes)) #, dtype=torch.float, device=self.device)

        for i in range(n_time):
            # kill nodes
            rk = torch.rand(self.num_organisms, self.num_nodes, device=self.device)
            self.state[rk < self.pi_death] = 0

            # repair nodes
            rr = torch.rand(self.num_organisms, self.num_nodes, device=self.device)
            self.state[rr < self.pi_repair] = 1
            # cascade failures 
            repeat = True # while can still propagate failures to neighboring nodes
            while repeat:
                num_active_neighbors = self.state @ self.adjmat
                frac_active_neighbors = (num_active_neighbors + 1e-20)/(self.number_neighbors + 1e-20)
                n_living_prev = self.state.sum()
                self.state[frac_active_neighbors < 0.5] = 0
                n_living = self.state.sum()
                # if reach steady state (no update), stop cascade loop
                if torch.abs(n_living-n_living_prev) < 0.1:
                    repeat = False

            self.fraction_alive[i,:] = (torch.sum(self.state, dim=1)/self.num_nodes)

            if verbose > 0:
                if i % verbose == 0:
                    print(f"epoch {i} of {n_time}")

            organism_state[i:,self.fraction_alive[i,:]<tau] = 0
            if save_all is True:
                state_dynamics[i,:] = self.state.cpu().clone().detach().numpy()

        # compute survival and mortality curves...
        survival = organism_state.sum(dim=1).cpu().detach().clone().numpy()
        dtimes = np.argmax(self.fraction_alive.cpu().detach().clone().numpy() < tau, axis=0)
        t95, mort = self.calc_mort(dtimes, survival)
        self.survival = survival
        self.mortality = mort
        # print("done")
        # return t95, dtimes, self.survival, self.mortality, self.state
        res = { 't95': t95,
                'dead_times': dtimes,
                'survival': self.survival,
                'mortality': self.mortality,
                'final_state': self.state}
        if save_all is True:
            res['state_dynamics'] = state_dynamics
        return res
    

def run_model(config):
    start_time = datetime.now()
    seed = config.getint("General", "random_seed", fallback=42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # load network
    amat = nw.get_adjacency_matrix(config)
    
    # load parameters
    eta = config.getfloat("General", "init_nonfunctional")
    pi_death = config.getfloat("General", "prob_death")
    pi_repair = config.getfloat("General", "prob_repair")
    n_organisms = config.getint("General", "num_organisms")
    n_time = config.getint("General", "num_timepoints")

    outpath = Path(config["General"]["outdir"])
    outpath.mkdir(parents=True, exist_ok=True)

    disable_GPU = config.getboolean("General", "disable_GPU", fallback=False)
    if disable_GPU:
        device = torch.device("cpu")
    else:
        # else use GPU if available
        device = get_device()

    model = NetworkAgingModel(adjacency_matrix=amat,
                         num_organisms=n_organisms,
                         init_nonfunctional=eta,
                         prob_death=pi_death, 
                         prob_repair=pi_repair,
                         device=device)
    res = model.age_network(n_time, tau=0.01) #, verbose=100, save_all=True) # TODO: add config options for tau, verbose, save_all
    pickle_save(outpath / "results.pkl", res)
    print("saved results to", outpath / "results.pkl")
    
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
    parser = argparse.ArgumentParser(description="Run network model")
    parser.add_argument("--config", type=str, dest='configfile', help="Path to configuration file")
    parser.add_argument("--overrides", type=str, 
                        nargs='+', help="Override parameters in section:parameter=value format.", default=[])
    args = parser.parse_args()

    main(args)

# TODO
#! check default parameters
#! check aging calc and separate params/methods; plotting...