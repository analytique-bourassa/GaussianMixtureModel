import torch
import os
import pyro
from torch.distributions import constraints
from matplotlib import pyplot
from collections import defaultdict

import pyro.distributions as dist
from pyro import poutine
from pyro.contrib.autoguide import AutoDelta
from pyro.optim import Adam
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate

smoke_test = ('CI' in os.environ)
assert pyro.__version__.startswith('0.3.0')

class GaussianMixtureModel(object):


    def __init__(self,data, number_of_hidden_states=3):

        self.number_of_hidden_states = number_of_hidden_states
        self.data = data
        self.global_guide = AutoDelta(poutine.block(self.model, expose=['weights', 'locs', 'scale']))
        self.svi = None
        self.losses = None
        self.gradient_norms = None


    @config_enumerate
    def model(self, data):

        # Global variables.
        weights = pyro.sample('weights', dist.Dirichlet(0.5 * torch.ones(self.number_of_hidden_states)))

        with pyro.plate('components', self.number_of_hidden_states):
            locs = pyro.sample('locs', dist.Normal(0., 10.))
            scale = pyro.sample('scale', dist.LogNormal(0., 2.))

        with pyro.plate('data', len(data)):
            # Local variables.
            assignment = pyro.sample('assignment', dist.Categorical(weights))
            pyro.sample('obs', dist.Normal(locs[assignment], scale[assignment]), obs=data)

    def initialize(self, seed):

        pyro.set_rng_seed(seed)
        pyro.clear_param_store()

        pyro.param('auto_weights', 0.5 * torch.ones(self.number_of_hidden_states), constraint=constraints.simplex)
        pyro.param('auto_scale', (self.data.var() / 2).sqrt(), constraint=constraints.positive)
        pyro.param('auto_locs',
                   self.data[torch.multinomial(torch.ones(len(self.data)) / len(self.data),
                                          self.number_of_hidden_states)])

        optim = pyro.optim.Adam({'lr': 0.1, 'betas': [0.8, 0.99]})
        elbo = TraceEnum_ELBO(max_plate_nesting=1)

        self.svi = SVI(self.model, self.global_guide, optim, loss=elbo)
        loss = self.svi.loss(self.model, self.global_guide, self.data)

        return loss

    def train(self):

        loss, seed = min((self.initialize(seed), seed) for seed in range(100))
        self.initialize(seed)

        gradient_norms = defaultdict(list)
        for name, value in pyro.get_param_store().named_parameters():
            value.register_hook(lambda g, name=name: gradient_norms[name].append(g.norm().item()))

        losses = []
        for i in range(200 if not smoke_test else 2):
            loss = self.svi.step(self.data)
            losses.append(loss)

        self.losses = losses
        self.gradient_norms = gradient_norms

    def show_losses(self):

        assert self.losses is not None, "must train the model before showing losses" \
                                        ""
        pyplot.figure(figsize=(10, 3), dpi=100).set_facecolor('white')
        pyplot.plot(self.losses)
        pyplot.xlabel('iters', size=18)
        pyplot.ylabel('loss', size=18)
        #pyplot.yscale('log')
        pyplot.grid()
        pyplot.title('Convergence of stochastic variational inference', size=20)

        pyplot.show()


    def show_gradients_norm(self):

        pyplot.figure(figsize=(10, 4), dpi=100).set_facecolor('white')
        for name, grad_norms in self.gradient_norms.items():
            pyplot.plot(grad_norms, label=name)
        pyplot.xlabel('iters')
        pyplot.ylabel('gradient norm')
        pyplot.yscale('log')
        pyplot.legend(loc='best')
        pyplot.title('Gradient norms during SVI')
        pyplot.show()

    def return_map_estimate(self):

        map_estimates = self.global_guide(self.data)
        weights = map_estimates['weights'].data.numpy()
        locs = map_estimates['locs'].data.numpy()
        scale = map_estimates['scale'].data.numpy()

        return weights, locs, scale




