import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(20)

class DataGenerator(object):

    def __init__(self):

        self._number_of_clusters = 3
        self._number_of_datapoints = 100

        self.prob_per_cluster = np.array([0.3, 0.3, 0.4])
        mean_per_cluster = np.array([1, 4, 8])

        p_assignment = np.random.uniform(0,1, size=self._number_of_datapoints)

        cluster_indexes = []
        upper_limit_probability = np.cumsum(self.prob_per_cluster)
        upper_limit_probability = np.concatenate((np.array([0.0]), upper_limit_probability))

        for index in range(self._number_of_clusters):
            lower_limit = upper_limit_probability[index]
            upper_limit = upper_limit_probability[index+1]
            indexes_true = np.argwhere((lower_limit <= p_assignment) & (p_assignment < upper_limit))

            cluster_indexes.append(np.ravel(indexes_true))

        self._x = np.zeros(self._number_of_datapoints)

        for index_cluster, list_indexes_for_cluster in enumerate(cluster_indexes):

            values = np.random.normal(loc=mean_per_cluster[index_cluster],
                             scale=1.0,
                             size=len(list_indexes_for_cluster))

            self._x[list_indexes_for_cluster] = values

    @property
    def number_of_clusters(self):
        return self._number_of_clusters

    @property
    def number_of_datapoints(self):
        return self._number_of_datapoints

    @property
    def x(self):
        return self._x

    def show(self):


        plt.grid(True)
        plt.ylabel("Density")
        plt.xlabel("value of feature (x)")
        plt.title("Gaussian mixtures")
        plt.hist(self.x, bins=20, normed=True)
        plt.show()




