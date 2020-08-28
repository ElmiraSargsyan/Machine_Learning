import numpy as np

from practical3.gradient_descent import newton_raphson_method


class LogisticRegression:
    def __init__(self, 
                 l: float = 0.1,
                 update_method=newton_raphson_method,
                 update_params=None,
                 weights = None,
                 epochs=10):
        """Initialize Logistic Regression model

        :param dims: number of dimensions of data
        :param epochs: number of iterations over whole data
        :param update_method: update formula to use
        :param update_params: additional key word parameters to pass
                              to update function
        """
        if update_params is None:
            update_params = {}
        if update_method is newton_raphson_method:
            update_params['hessianloss'] = self.hessianloss
            
                  
        self.epochs = epochs
        self.update_method = update_method
        self.update_params = update_params
        self.weights = weights
        self.l = l
        

    @staticmethod
    def _generate_initial_weights(dims):
        # Fill with random initial values
        np.random.seed(42)
        return np.random.rand(dims,)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-1 / 100*x))
        
    @staticmethod
    def standardize(data, m, std):
        data = data.astype("float32")
        for datapoint in data:
            datapoint -= m
            datapoint /= std
        return data
    
    @staticmethod
    def normalize(data):
        return data / 255.0
        
    def fit(self, data, labels):
        """Fit the model and fix weight vector

        :param data: [N, rows, cols] dimensional numpy array of floats
        :param labels: [N] dimensional numpy array of 1s and -1s denoting class
        :yield: the function will yield weight vector after each update
        """

        data = self.transform_data(data, fit=True)
        
        if self.weights is None:
            self.w = self._generate_initial_weights(data.shape[1])
        else:
            self.w = np.loadtxt(self.weights, delimiter = ',')
            
        for num_epoch in range(self.epochs):
            #print(f"epoch N{num_epoch+1}:")
            # FIXME: Won't work correctly for windows, sorry :/
            for dw in self.update_method(data, labels, self.gradloss
                                        ,**self.update_params
                                         ):
                
                self.w -= dw
            yield self.w

    def loss(self, data, labels):
        """Calculate the loss of the model for current weights on the given data
        The loss function is equal to -log likelihood.
        :param data: [N, dims] dimensional numpy array of datapoints
        :param labels: [N] dimensional array of 1s and 0s
        """
        data = self.transform_data(data)
        
        y_n = LogisticRegression.sigmoid(np.dot(data, self.w))
        t_n = labels
        
        return ((-t_n * np.log(y_n) - (1 - t_n) * np.log(1 - y_n)).sum() + self.l / 2 * np.dot(self.w.T, self.w)) / len(data)

    def gradloss(self, data, labels):
        """Calculate the gradient of loss
        
        :param data: [N, dims] dimensional numpy array of datapoints
        :param labels: [N] dimensional array of 1s and 0s
        """
        
        y_n = LogisticRegression.sigmoid(np.dot(data, self.w))
        t_n = labels
        return (np.dot(data.T, (y_n-t_n)) + self.l * self.w) / len(data)
    
    def hessianloss(self, data, labels):
        """Calculate the Hessian matrix of loss
        :param data: [N, dims] dimensional numpy array of datapoints
        :param labels: [N] dimensional array of 1s and 0s
        """
        y_n = LogisticRegression.sigmoid(np.dot(data, self.w))
        R = np.dot(np.diag(y_n), np.diag(1.0 - y_n))
                           
        return np.dot(np.dot(data.T, R), data)   
        

    def calculate_probabilities(self, data):
        """Calculate probabilities for each datapoint of the given data
           of being from the first class

        :param data: [N, rows, cols] dimensional numpy array to predict classes
        :return: numpy array of probabilities,
                 where return_i denotes data_i's class
        >>> model = LogisticRegression(2)
        >>> model.w = np.array([1, 2])
        >>> np.all(model.calculate_probabilities(np.array([[-2, 1]]))
        ...        == np.array([0.5]))
        True
        >>> np.all(model.calculate_probabilities(np.array([[2, 0]]))
        ...        == model.calculate_probabilities(np.array([[0, 1]])))
        True
        """
        data = self.transform_data(data)
        return LogisticRegression.sigmoid(np.dot(data, self.w))

    def predict(self, data):
        """Calculate labels for each datapoint of the given data

        :param data: [N, rows, cols] dimensional numpy array to predict classes
        :return: numpy array of 1s and 0s,
                 where return_i denotes data_i's class
        >>> model = LogisticRegression(2)
        >>> model.w = np.array([1, 2])
        >>> np.all(model.predict(np.array([[2, 1], [1, 0], [0, -1]]))
        ...        == np.array([1, 1, 0]))
        True
        """
        return np.round(self.calculate_probabilities(data))
        
    def accuracy(self, data, labels):
        
        res = LogisticRegression.predict(self, data) == labels
        return res.sum() * 100 / len(res)
    
    def transform_data(self, data, fit=False):
        """Transform the data adding features, scaling, etc.
        Don't forget to save scaling parameters on `self` if fit is True

        :param data: [N, rows, cols] shaped numpy array of images
        :param fit: if True, scaling parameters should be calculated,
                    if False, scaling should be done with already calculated parameters
                    Note: any processing is not required
        :return: [N, dims] shaped numpy array
        """
        data = [datapoint.flatten() for datapoint in data]
        data = np.array(data)
        
        if fit:
            self.mask = []
            
            for i in range(data.shape[1]):
                if set(data[:, i]) == {0}:
                    self.mask.append(False)
                else:
                    self.mask.append(True)
                    
            self.m = np.mean(data, axis = 0)
            self.std = np.std(data, axis = 0)
       
        data = self.standardize(data, self.m, self.std)
        #data = self.normalize(data)    
        
            
        data = np.c_[np.ones(len(data)), data[:, self.mask]]
        
        return np.array(data)
        
