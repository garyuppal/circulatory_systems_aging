import torch
import numpy as np

#* may not be differentiable?; first at least code the algo, then try to make 'optimizable' and then check with original model
#* can use bernoulli random var to kill cells, with gumbel softmax
#* do we also want to use neural odes for backprop??? - otherwise has to backprop through full simulation...

#* compute number cells alive at given time <n>(t) - averaged over ensemble
#* population (number organisms alive) S(t) over time
#* mortality mu(t) = -(1/s) ds/dt


#* simulation consists of cells, chemical fields, 

#! start with one system; then try to parallelize and run ensemble simultaneously

class ChemicalField:
    def __init__(self, diff, vel, decay, hill_k, xmin=0, xmax=100, nx=0.01):
        self.diff = diff
        self.vel = vel
        self.decay = decay
        self.hill_k = hill_k

        self.dx = (xmax-xmin)/nx
        self.x = np.arange(xmin, xmax, self.dx)
        self.field = np.zeros(nx+3) #* including 'ghost' nodes

    def add(self, cells):
        x_secrete = cells.state*cells.x_position #* inactive ones don't contribute to field

    def get_diff_field(self):
        pass

    def get_adv_field(self):
        pass

    def step(self, dt):
        diff_field = self.get_diff_field()
        adv_field = self.get_adv_field()

        self.field = self.field + dt*(diff_field + adv_field - self.decay*self.field)


class Cells:
    def __init__(self, scale_alpha, mix_gamma, xmin=0, xmax=100, n=10000):
        self.alpha = scale_alpha
        self.gamma = mix_gamma
        self.state = np.ones(n)
        self.x_position = np.linspace(xmin, xmax, n)

    def step(self, good_field, toxin_field):
        #* cells only die, don't recover?...
        pass
    

class Organism:
    def __init__(self):
        pass


class Age:
    def __init__(self, n_epochs):
        # TODO: parallelize over an ensemble of organisms...
        pass