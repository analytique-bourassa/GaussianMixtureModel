import os
import pyro
import torch
import numpy as np
from data_generator import DataGenerator
import matplotlib.pyplot as plt

from gmm.GaussianMixtureModel_Pyro import GaussianMixtureModel
from GaussianMixtureFunctor_1d import GaussianMixtureFunctor_1d

smoke_test = ('CI' in os.environ)
assert pyro.__version__.startswith('0.3.0')


pyro.enable_validation(True)
torch.set_default_tensor_type('torch.DoubleTensor')

data_generated = DataGenerator()
data_generated.show()

observations = data_generated.x.copy()

data = torch.tensor(observations)
K = 3  # Fixed number of components.

model_gmm = GaussianMixtureModel(data, K)
model_gmm.train()
model_gmm.show_losses()
model_gmm.show_gradients_norm()
weights, locs, scale = model_gmm.return_map_estimate()

mixture = GaussianMixtureFunctor_1d(weights, locs, scale)

x = np.linspace(-1, 11, 100)
density = np.array([mixture(value) for value in x])

plt.grid(True)
plt.plot(x, density,  linewidth=3, label="density calculated")
plt.hist(data_generated.x, normed=True, color="blue", bins=20, alpha = 0.5, label="data simulated")
plt.ylabel("density")
plt.xlabel("feature value (x)")
plt.title("Gaussian mixture obtained using stochastic variational inference")
plt.legend()
plt.show()
