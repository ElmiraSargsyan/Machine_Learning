import numpy as np

def get_distance(data, means):
    dist = lambda point, data: [np.sum((datapoint-point)**2) for datapoint in data]
    return [dist(mean, data) for mean in means]

def random_initialize(data_array: np.ndarray,
                      num_clusters: int) -> list:
    """
    Initialize cluster centers by sampling `num_clusters` points
    uniformly from data_array. Returned values must not repeat.
    """
    return list(data_array[np.random.choice(np.arange(data_array.shape[0]), num_clusters, replace=False)])
    

def plus_plus_initialize(data_array: np.ndarray,
                         num_clusters: int) -> list:
    """
    Initialize cluster centers using k-means++ algorithm.
    """
    means = []

    # initialize randomly the first clusters center
    means.append(data_array[np.random.choice(np.arange(data_array.shape[0]))])

    for _ in range(num_clusters - 1):

        # define probability of each point being chosen as next cluster-center
        # proportional to minimal square distance from existing cluster-centers
        prob = np.min(get_distance(data_array, means), axis = 0)

        # chose next center randomly
        # take into account probabilities assigned to points
        idx = np.argsort(prob)[::-1]       
        sorted = data_array[idx]
        next_center = sorted[0]
        i = 0

        while any([all(next_center == m) for m in means]):
            # chose another point as next center same as above
            i += 1
            next_center = sorted[i]

        means.append(next_center)

    return means


class KMeans:
    def __init__(self, num_mixtures: int):
        self.K = num_mixtures
        self.means = []

    def initialize(self, data: np.ndarray):
        """
        Initialize cluster centers

        :param data: data, numpy 2-D array
        """
        # Hint: Use one of the function at the top of the file.
        self.means = plus_plus_initialize(data, self.K)
        

    def fit(self, data: np.ndarray):
        """
        Initialize Mixtures, then run EM algorithm until it converges.

        :param data: data to fit, numpy 2-D array
        """

        self.initialize(data)
        old_labels = np.zeros(len(data))

        # repeat EM algorithm until there is no change in labels
        while True:
            # optimize labels, with means fixed
            new_labels = np.argmin(get_distance(data, self.means), axis = 0)  

            # break if algorithm converged
            if all(new_labels == old_labels):
                break

            # optimize means, with labels fixed as new_labels
            clusters = {}
            for i in range(self.K):    
                clusters[i] = data[i == new_labels]
  
            self.means = [np.mean(cluster, axis =  0) for cluster in clusters.values()] 

            old_labels = new_labels

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Determine which cluster each of the points belongs to.

        :param data: data, numpy 2-D array
        :return: labels, numpy 1-D array
        """
        # assign each point to the closest cluster
        return np.argmin(get_distance(data, self.means), axis = 0)

    def get_centers(self) -> list:
        """
        Return list of centers of the clusters, i.e. means
        """
        return self.means
