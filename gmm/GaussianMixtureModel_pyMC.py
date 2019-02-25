import numpy as np
import pymc3 as pm
import theano.tensor as tt
import matplotlib.pyplot as plt
from GaussianMixtureFunctor_1d import GaussianMixtureFunctor_1d

class GaussianMixtureModel_pyMC(object):

    def __init__(self, data, number_of_hidden_states=3):

        self.number_of_hidden_states = number_of_hidden_states
        self.data = data
        self.number_of_data = data.shape[0]


        self.model = None
        self.trace = None

        self._define_model()

    def _define_model(self):

        self.model = pm.Model()
        with self.model:


            p = pm.Dirichlet('p', a=np.array([1., 1., 1.]), shape=self.number_of_hidden_states)
            p_min_potential = pm.Potential('p_min_potential', tt.switch(tt.min(p) < .1, -np.inf, 0))

            means = pm.Normal('means', mu=[0, 0, 0], sd=2.0, shape=self.number_of_hidden_states)

            # break symmetry
            order_means_potential = pm.Potential('order_means_potential',
                                                 tt.switch(means[1] - means[0] < 0, -np.inf, 0)
                                                 + tt.switch(means[2] - means[1] < 0, -np.inf, 0))

            sd = pm.HalfCauchy('sd', beta=2, shape=self.number_of_hidden_states)
            category = pm.Categorical('category',
                                      p=p,
                                      shape=self.number_of_data)

            points = pm.Normal('obs',
                               mu=means[category],
                               sd=sd[category],
                               observed=self.data)

    def sample(self, number_of_samples=50000):

        with self.model:
            # step1 = pm.NUTS(vars=[p, sd, means])
            # step2 = pm.CategoricalGibbsMetropolis(vars=[category], proposal='proportional')
            self.trace = pm.sample(number_of_samples)  # , step=[step1, step2])

    def show_chains(self, end_burning_index=10000, fraction_to_show=5):

        assert self.trace is not None, "must use the method sample"

        pm.plots.traceplot(self.trace[end_burning_index::fraction_to_show], ['p', 'sd', 'means'])
        plt.show()

    def show_autocorrelation(self, end_burning_index=10000, fraction_to_show=5):

        assert self.trace is not None, "must use the method sample"

        pm.autocorrplot(self.trace[end_burning_index::fraction_to_show], varnames=['sd'])
        plt.show()

    def show_mixture(self, end_burning_index=10000):

        assert self.trace is not None, "must use the method sample"

        mu = np.mean(self.trace.get_values('means', burn=end_burning_index, combine=True), axis=0)
        sigma = np.mean(self.trace.get_values('sd', burn=end_burning_index, combine=True), axis=0)
        categorical_p = np.mean(self.trace.get_values('p', burn=end_burning_index, combine=True), axis=0)

        mixture = GaussianMixtureFunctor_1d(categorical_p, mu, sigma)

        x = np.linspace(-1, 11, 100)
        density = np.array([mixture(value) for value in x])

        plt.grid(True)
        plt.plot(x, density, linewidth=3, label="density calculated")
        plt.hist(self.data, normed=True, color="blue", bins=20, alpha=0.5, label="data simulated")
        plt.ylabel("density")
        plt.xlabel("feature value (x)")
        plt.title("Gaussian mixture obtained using Expectation-Maximization")
        plt.legend()
        plt.show()

    def get_means_of_parameters(self, end_burning_index=10000):

        assert self.trace is not None, "must use the method sample"

        mu = np.mean(self.trace.get_values('means', burn=end_burning_index, combine=True), axis=0)
        sigma = np.mean(self.trace.get_values('sd', burn=end_burning_index, combine=True), axis=0)
        categorical_p = np.mean(self.trace.get_values('p', burn=end_burning_index, combine=True), axis=0)

        return mu, sigma, categorical_p


