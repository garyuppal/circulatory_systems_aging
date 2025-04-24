import numpy as np
import torch 
# from utils import pickle_save


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
        survival = organism_state.sum(dim=1).cpu().detach().clone().numpy() #! make so that an organism can't come back to life
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
        