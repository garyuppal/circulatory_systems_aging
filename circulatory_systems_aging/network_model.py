import numpy as np
import torch 

#! NOTES:
#* want directed edges



# assign nodes as functional or not
#* simulate for 10000 networks...

#! would gpu and/or jit speed things up??

class NetworkDynamics:
    def __init__(self, adjacency_matrix, initial_nonfunctional_fraction, prob_death, prob_repair, tau):
        self.p_death = prob_death
        self.p_repair = prob_repair
        self.adj_mat = adjacency_matrix
        self.n_nodes = adjacency_matrix.shape[0]
        self.eta = initial_nonfunctional_fraction

        state = np.ones(self.n_nodes)
        r = np.random.rand(self.n_nodes)
        state[r < self.eta] = 0
        self.state = state
        
        self.num_connected = self.adj_mat.sum(axis=0)
        self.tau_threshold = tau # TODO: not sure we need this here...
    
    def age_network(self):
        # kill nodes with probability p_death
        rd = np.random.rand(self.n_nodes)
        self.state[rd < self.p_death] = 0

        # reactivate nodes with probability p_recover
        rr = np.random.rand(self.n_nodes)
        self.state[rr < self.p_repair] = 1

        # kill interdependent nodes with majority dead neighbors
        functional_connections = (self.state[:,None]*self.adj_mat).sum(axis=0)
        fraction_functional_connections = functional_connections/(self.num_connected + 1e-20) # TODO: what to do with 0/0??

        self.state[fraction_functional_connections < 0.5] = 0        

        return self.state


class NetworkDynamicsVectorized:
    def __init__(self, n_population, adjacency_matrix, initial_nonfunctional_fraction, prob_death, prob_repair, tau):
        self.n_population = n_population
        self.p_death = prob_death
        self.p_repair = prob_repair
        self.adj_mat = adjacency_matrix
        self.n_nodes = adjacency_matrix.shape[0]
        self.eta = initial_nonfunctional_fraction

        state = np.ones((self.n_population, self.n_nodes))
        r = np.random.rand(self.n_population, self.n_nodes)
        state[r < self.eta] = 0
        self.state = state
        
        self.num_connected = self.adj_mat.sum(axis=0)
        self.tau_threshold = tau # TODO: not sure we need this here...
    
    def age_network(self):
        # kill nodes with probability p_death
        rd = np.random.rand(self.n_population, self.n_nodes)
        self.state[rd < self.p_death] = 0

        # reactivate nodes with probability p_recover
        rr = np.random.rand(self.n_population, self.n_nodes)
        self.state[rr < self.p_repair] = 1

        # kill interdependent nodes with majority dead neighbors
        functional_connections = self.state @ self.adj_mat 
        # (self.state[:,None]*self.adj_mat).sum(axis=0)
        fraction_functional_connections = functional_connections/(self.num_connected[None,:] + 1e-20) # TODO: what to do with 0/0?? -> =1? 

        self.state[fraction_functional_connections < 0.5] = 0        

        return self.state

