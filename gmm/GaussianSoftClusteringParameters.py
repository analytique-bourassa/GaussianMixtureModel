import numpy as np


class GaussianSoftClusteringParameters(object):

    def __init__(self):

        self.hidden_states_distribution = None
        self.hidden_states_prior = None
        self.mu = None
        self.sigma = None

    def initialize_parameters(self, number_of_clusters, number_of_features, number_of_observations):

        pi = 1 / float(number_of_clusters) * np.ones(number_of_clusters)
        mu = np.random.randn(number_of_clusters, number_of_features)
        sigma = np.zeros((number_of_clusters, number_of_features, number_of_features))
        sigma[...] = np.identity(number_of_features)

        self.hidden_states_prior = pi
        self.mu = mu
        self.sigma = sigma
        self.hidden_states_distribution = np.zeros((number_of_observations, number_of_clusters))

    # TODO implement setter and getter
    # assert isinstance np.ndarray
