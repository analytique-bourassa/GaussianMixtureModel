import numpy as np
from scipy.stats import multivariate_normal
from tqdm import tqdm

from gmm.GaussianSoftClusteringParameters import GaussianSoftClusteringParameters

np.random.seed(42)

class GaussianSoftClustering(object):
    """
    Based on assignment from week 2 Bayesian method for machine learning of Coursera.
    """

    def __init__(self):

        self.parameters = GaussianSoftClusteringParameters()

    def _E_step(self, observations, parameters):

        """
        Performs E-step on GMM model
        # P(i|x)=p(x|i)p(i)/z
        # p(x_n|i)=N(x_n| mu_i,sigma_i)

        changed:
        --------

        parameters.hidden_states_distribution: [|data| x |states|], probabilities of states for objects

        keeped constant:
        ----------------

        parameters.mu
        parameters.sigma
        parameters.hidden_states_prior

        """
        assert isinstance(observations, np.ndarray)

        number_of_observations = observations.shape[0]
        number_of_clusters = parameters.hidden_states_prior.shape[0]
        hidden_states_distribution = np.zeros((number_of_observations, number_of_clusters))

        for cluster_index in range(number_of_clusters):

            multivariate_normal_pdf = multivariate_normal.pdf(observations,
                                                              mean=parameters.mu[cluster_index, :],
                                                              cov=parameters.sigma[cluster_index, ...])

            hidden_states_distribution[:, cluster_index] = multivariate_normal_pdf *\
                                                           (parameters.hidden_states_prior[cluster_index])

        hidden_states_distribution /= np.sum(hidden_states_distribution, 1).reshape(-1, 1)

        parameters.hidden_states_distribution = hidden_states_distribution

    def _M_step(self, observations, parameters):
        """
        Performs M-step on GMM model

        changed:
        --------

        parameters.mu
        parameters.sigma
        parameters.hidden_states_prior

        keeped constant:
        ----------------

        parameters.hidden_states_distribution: [|data| x |states|], probabilities of states for objects

        """

        assert isinstance(observations, np.ndarray)
        assert isinstance(parameters.hidden_states_distribution, np.ndarray)

        number_of_objects = observations.shape[0]
        number_of_clusters = parameters.hidden_states_distribution.shape[1]
        number_of_features = observations.shape[1]

        normalization_constants = np.sum(parameters.hidden_states_distribution, 0)

        mu = np.dot(parameters.hidden_states_distribution.T, observations) / normalization_constants.reshape(-1, 1)
        hidden_states_prior = normalization_constants / number_of_objects
        sigma = np.zeros((number_of_clusters, number_of_features, number_of_features))

        for state_index in range(number_of_clusters):

            x_mu = observations - mu[state_index]
            hidden_state_weights_diag = np.diag(parameters.hidden_states_distribution[:, state_index])
           
            sigma_state = np.dot(np.dot(x_mu.T, hidden_state_weights_diag), x_mu)
            sigma[state_index, ...] = sigma_state / normalization_constants[state_index]

        parameters.hidden_states_prior = hidden_states_prior
        parameters.mu = mu
        parameters.sigma = sigma

    def compute_vlb(self, observations, parameters):
        """
        observations: [ |data| x |features| ]
        
        hidden_states_distribution: [|data| x |states|]
        states_prior: [|states|]
        
        mu: [|states| x |features|]
        sigma: [|states| x |features| x |features|] 

        Returns value of variational lower bound
        """
        
        assert isinstance(observations, np.ndarray)
        assert isinstance(parameters.hidden_states_prior, np.ndarray)
        assert isinstance(parameters.mu, np.ndarray)
        assert isinstance(parameters.sigma, np.ndarray)
        assert isinstance(parameters.hidden_states_distribution, np.ndarray)
        
        number_of_observations = observations.shape[0]
        number_of_clusters = parameters.hidden_states_distribution.shape[1]

        loss_per_observation = np.zeros(number_of_observations)
        for k in range(number_of_clusters):

            energy = parameters.hidden_states_distribution[:, k] * (np.log(parameters.hidden_states_prior[k]) +
                        multivariate_normal.logpdf(observations, mean=parameters.mu[k, :], cov=parameters.sigma[k, ...]))

            entropy = parameters.hidden_states_distribution[:, k] * np.log(parameters.hidden_states_distribution[:, k])

            loss_per_observation += energy
            loss_per_observation -= entropy

        total_loss = np.sum(loss_per_observation)

        return total_loss

    def train_EM(self, observations, number_of_clusters, reducing_factor=1e-3, max_iter=100, restarts=10):

   
        number_of_features = observations.shape[1] 
        number_of_observations = observations.shape[0]

        best_loss = -1e7
        best_parameters = GaussianSoftClusteringParameters()

        for _ in tqdm(range(restarts)):

            try:
                parameters = GaussianSoftClusteringParameters()
                parameters.initialize_parameters(number_of_clusters,
                                                 number_of_features,
                                                 number_of_observations)

                self._E_step(observations, parameters)

                prev_loss = self.compute_vlb(observations,
                                             parameters)

                for _ in range(max_iter):

                    self._E_step(observations, parameters)
                    self._M_step(observations, parameters)

                    loss = self.compute_vlb(observations, parameters)

                    if loss / prev_loss < reducing_factor:
                        break

                    if loss > best_loss:

                        best_loss = loss
                        best_parameters = parameters

                    prev_loss = loss

            except np.linalg.LinAlgError:
                print("Singular matrix: components collapsed")
                pass

        return best_loss, best_parameters
