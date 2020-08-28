import numpy as np

from gradient_descent import newton_raphson_method


class LogisticRegression:
    def __init__(self, dims,
                 update_method=newton_raphson_method,
                 update_params=None,
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
        self.dims = dims
        self.epochs = epochs
        self.update_method = update_method
        self.update_params = update_params
        self.w = self._generate_initial_weights(dims)

    @staticmethod
    def _generate_initial_weights(dims):
        # FIXME: Fill with random initial values
        return np.ones(dims)
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def fit(self, data, labels):
        """Fit the model and fix weight vector

        :param data: [N X dims] dimensional numpy array of floats
        :param labels: [N] dimensional numpy array of 1s and -1s denoting class
        :yield: the function will yield weight vector after each update
        """
        yield self.w
        for num_epoch in range(self.epochs):
            print(f"epoch N{num_epoch}:", end='\r', flush=True)
            # FIXME: Won't work correctly for windows, sorry :/
            for dw in self.update_method(data, labels, self.gradloss,
                                         **self.update_params):
                self.w -= dw
                yield self.w

    def loss(self, data, labels):
        """Calculate the loss of the model for current weights on the given data
        The loss function is equal to -log likelihood.
        :param data: [N, dims] dimensional numpy array of datapoints
        :param labels: [N] dimensional array of 1s and 0s
        """
        y_n = sigmoid(np.dot(data, self.w))
        t_n = labels
        return (-t_n * np.log(y_n) - (1 - t_n) * np.log(1 - y_n)).sum()
        #maybe here divided by len(data)
        

    def gradloss(self, data, labels):
        """Calculate the gradient of loss
        
        :param data: [N, dims] dimensional numpy array of datapoints
        :param labels: [N] dimensional array of 1s and 0s
        """
        y_n = sigmoid(np.dot(data, self.w))
        t_n = labels
        return np.dot(data.T, (y_n-t_n)).sum()
        
    
    def hessianloss(self, data, labels):
        """Calculate the Hessian matrix of loss
        :param data: [N, dims] dimensional numpy array of datapoints
        :param labels: [N] dimensional array of 1s and 0s
        """
        raise NotImplementedError()

    def calculate_probabilities(self, data):
        """Calculate labels for each datapoing of the given data

        :param data: [N X dims] dimensional numpy array to predict classes
        :return: numpy array of 1s and -1s,
                 where return_i denotes data_i's class
        >>> model = LogisticRegression(2)
        >>> model.w = np.array([1, 2])
        >>> model.predict(np.array([[-2, 1]]))
        array([0.5])
        >>> (model.predict(np.array([[2, 0]]))
        ...  == model.predict(np.array([[0, 1]])))
        True
        """
        raise NotImplementedError()

    def predict(self, data):
        """Calculate labels for each datapoing of the given data

        :param data: [N X dims] dimensional numpy array to predict classes
        :return: numpy array of 1s and -1s,
                 where return_i denotes data_i's class
        >>> model = LogisticRegression(2)
        >>> model.w = np.array([1, 2])
        >>> model.predict(np.array([[2, 1], [1, 0], [0, -1]]))
        array([1, 1, 0])
        """
        raise NotImplementedError()

