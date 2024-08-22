import numpy as np
import torch 
# from utils import pickle_save


class NetworkAging:
    def __init__(self, 
                 adjacency_matrix,
                 num_organisms,
                 init_nonfunctional,
                 prob_death,
                 prob_repair):
        self.num_nodes = adjacency_matrix.shape[0]
        self.num_organisms = num_organisms
        self.eta = init_nonfunctional
        self.pi_death = prob_death
        self.pi_repair = prob_repair
        self.adjmat = adjacency_matrix
        self.number_neighbors = self.adjmat.sum(axis=0)
        
        self.init_state()
        self.fraction_alive = None
        self.survival = None
        self.mortality = None
    
    def init_state(self):
        self.state = np.ones((self.num_organisms, self.num_nodes))
        r = np.random.rand(self.num_organisms, self.num_nodes)
        self.state[r < self.eta] = 0

    def age_network(self, n_time, tau=0.01, verbose=-1):
        self.fraction_alive = np.zeros((n_time, self.num_organisms))

        for i in range(n_time):
            # kill nodes
            rk = np.random.rand(self.num_organisms, self.num_nodes)
            self.state[rk < self.pi_death] = 0
            # repair nodes
            rr = np.random.rand(self.num_organisms, self.num_nodes)
            self.state[rr < self.pi_repair] = 1
            # cascade failures
            num_active_neighbors = self.state @ self.adjmat
            frac_active_neighbors = (num_active_neighbors + 1e-20)/(self.number_neighbors + 1e-20)
            self.state[frac_active_neighbors < 0.5] = 0

            self.fraction_alive[i,:] = (np.sum(self.state, axis=1)/self.num_nodes)

            if verbose > 0:
                if i % verbose == 0:
                    print(f"epoch {i} of {n_time}")
        
        # compute survival and mortality curves...
        survival = (self.fraction_alive > tau).sum(axis=1)
        dtimes = np.argmax(self.fraction_alive < tau, axis=0)
        delta_t = max(int(0.25*np.std(dtimes)), 1)
        mortality = np.zeros(n_time-delta_t)
        for i in range(n_time-delta_t):
            mortality[i] = -(survival[i+delta_t] - survival[i])/(survival[i]*delta_t)
        
        self.survival = survival
        self.mortality = mortality
        print("done")
        return self.fraction_alive, self.survival, self.mortality


# pytorch module
class NetworkAgingTorch:
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

    def age_network(self, n_time, tau=0.01, verbose=-1):
        self.fraction_alive = torch.zeros((n_time, self.num_organisms), dtype=torch.float, device=self.device)

        for i in range(n_time):
            # kill nodes
            rk = torch.rand(self.num_organisms, self.num_nodes, device=self.device)
            self.state[rk < self.pi_death] = 0
            # repair nodes
            rr = torch.rand(self.num_organisms, self.num_nodes, device=self.device)
            self.state[rr < self.pi_repair] = 1
            # cascade failures
            num_active_neighbors = self.state @ self.adjmat
            frac_active_neighbors = (num_active_neighbors + 1e-20)/(self.number_neighbors + 1e-20)
            self.state[frac_active_neighbors < 0.5] = 0

            self.fraction_alive[i,:] = (torch.sum(self.state, dim=1)/self.num_nodes)

            if verbose > 0:
                if i % verbose == 0:
                    print(f"epoch {i} of {n_time}")
        
        # compute survival and mortality curves...
        survival = (self.fraction_alive > tau).sum(dim=1).cpu().detach().clone().numpy()
        dtimes = np.argmax(self.fraction_alive.cpu().detach().clone().numpy() < tau, axis=0)
        delta_t = max(int(0.25*np.std(dtimes)), 1)
        mortality = np.zeros(n_time-delta_t)
        for i in range(n_time-delta_t):
            mortality[i] = -(survival[i+delta_t] - survival[i])/(survival[i]*delta_t)
        
        self.survival = survival
        self.mortality = mortality
        print("done")
        return self.survival, self.mortality
    