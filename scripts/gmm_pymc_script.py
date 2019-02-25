from data_generator import DataGenerator
from gmm.GaussianMixtureModel_pyMC import GaussianMixtureModel_pyMC
import matplotlib.pyplot as plt

from GaussianMixtureFunctor_1d import GaussianMixtureFunctor_1d
import numpy as np

data_generated = DataGenerator()
data_generated.show()

observations = data_generated.x.copy()

ndata = len(observations)
data = observations
k = 3

model_gmm = GaussianMixtureModel_pyMC(data, 3)
model_gmm.sample()
model_gmm.show_chains()

mu, sigma, p = model_gmm.get_means_of_parameters()

mixture = GaussianMixtureFunctor_1d(p, mu, sigma)

x = np.linspace(-1, 11, 100)
density = np.array([mixture(value) for value in x])

plt.grid(True)
plt.plot(x, density, linewidth=3, label="density calculated")
plt.hist(data_generated.x, normed=True, color="blue", bins=20, alpha=0.5, label="data simulated")
plt.ylabel("density")
plt.xlabel("feature value (x)")
plt.title("Gaussian mixture obtained using Markov Chains Monte-Carlo")
plt.legend()
plt.show()


























































