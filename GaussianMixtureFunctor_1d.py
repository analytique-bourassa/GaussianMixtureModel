from scipy.stats import norm

class GaussianMixtureFunctor_1d():

    def __init__(self, pi, mu, sigma):

        self.number_of_hidden_states = len(pi)
        self.pi = pi
        self.distributions = list()

        for index in range(self.number_of_hidden_states):
            self.distributions.append(norm(loc=mu[index], scale=sigma[index]))

    def __call__(self, value):

        density = 0.0
        for index in range(self.number_of_hidden_states):
            density += self.pi[index] * self.distributions[index].pdf(value)

        return density