import numpy as np

from kmeans import KMeans

def multi_normal(x: np.ndarray,
                 mean: np.ndarray,
                 covariance: np.ndarray) -> float:
    """
    Evaluates Multivariate Gaussian Distribution density function
    :param x: location where to evaluate the density function
    :param mean: Center of the Gaussian Distribution
    :param covariance: Covariance of the Gaussian Distribution
    :return: density function evaluated at point x
    """
    cov = covariance
    return np.exp(-0.5 * ((x-mean).T @ np.linalg.inv(cov) @ (x-mean))) / (((np.linalg.det(cov))**(0.5)) * ((2 * np.pi)**(len(x) / 2)))

def evaluate_loss(data: np.ndarray,
                  num_mixtures: int, weights: list,
                  centers: list, covariances: list) -> float:

    """
    Evaluates loss function for given data and parameters
    """
    logL = 0
    for n in range(data.shape[0]):
        L = 0
        for k in range(num_mixtures):    
            L += weights[k] * multi_normal(data[n], centers[k], covariances[k])
        logL += np.log(L.clip(min = 1e-10))

    return logL 


class GaussianMixtureModel:
    def __init__(self, num_mixtures: int):
        self.K: int = num_mixtures
        self.centers: list = []
        self.weights: list = []
        self.covariances: list = []
        self.r = None  # Matrix of responsibilities, i.e. gamma

    def initialize(self, data: np.ndarray):
        """
        Initializes cluster centers, weights, and covariances

        :param data: data, numpy 2-D array
        """
        km = KMeans(self.K)
        km.fit(data)
        _ = km.predict(data)
        self.centers = km.get_centers()
        self.weights = np.random.uniform(0,1,(self.K,))
        self.weights = self.weights/np.sum(self.weights)
        self.covariances = np.array([np.eye(data.shape[-1])] * self.K) * 10e8


    def fit(self, data: np.ndarray,
            max_iter: int = 100,
            precision: float = 1e-6):
        """
        Initializes Mixtures, then runs EM algorithm until it converges.

        :param data: data to fit, numpy 2-D array
        """
        print("K ", self.K)
        self.initialize(data)
        gammas = np.empty([data.shape[0], self.K])
        #num_data = len(data)
        old_loss = evaluate_loss(
            data, self.K, self.weights, self.centers, self.covariances)

        for iteration in range(1, max_iter + 1):

            # Perform E step i. e. calculate matrix of responsibilities           
            for i in range(data.shape[0]):                
                for k in range(self.K):
                    gammas[i][k] = self.weights[k] * multi_normal(data[i], self.centers[k], self.covariances[k])
 
            self.r = gammas / np.sum(gammas, axis = 1)[None].T.clip(min = 1e-10)

            # Perform M step
            Nk = np.sum(self.r, axis = 0).clip(min = 1e-10)
            
            # Optimize cluster-centers, evaluate self.centers
            self.centers = (self.r.T @ data) / Nk[:, None]
            
            # Optimize clusters' covariances, evaluate self.covariances
            for k in range(self.K):
                self.covariances[k] = np.zeros((data.shape[-1], data.shape[-1]))
                for i in range(data.shape[0]):
                    self.covariances[k] += self.r[i][k] * np.dot((data[i]-self.centers[k])[None].T, (data[i]-self.centers[k][None]))
            self.covariances /= Nk[:, None, None]

            # Optimize weights, evaluate self.weights
            self.weights = Nk / data.shape[0]

            new_loss = evaluate_loss(
                data, self.K, self.weights, self.centers, self.covariances)
            
            print(f"iter: {iteration}, log likelihood: {new_loss}")
            
            # Check for termination
            if abs(new_loss - old_loss) < precision:
                print("Finished")
                break
           
            old_loss = new_loss

    def get_centers(self):
        return self.centers

    def get_covariances(self):
        return self.covariances

    def get_weights(self):
        return self.weights

    def predict_cluster(self, data: np.ndarray) -> np.ndarray:
        """
        Returns index of the clusters that each point is most likely to belong.
        :param data: data, numpy 2-D array
        :return: labels, numpy 1-D array
        """
        # Calculate responsibilities for each data point i. e. gammas.
        # Assign each data point to cluster with biggest responsibility.

        gammas = np.empty([data.shape[0], self.K])
        for i in range(data.shape[0]):                
                for k in range(self.K):
                    gammas[i][k] = self.weights[k] * multi_normal(data[i], self.centers[k], self.covariances[k])
        
        gammas = gammas/np.sum(gammas, axis = 1)[None].T

        return np.argmax(gammas, axis = 1)
        
