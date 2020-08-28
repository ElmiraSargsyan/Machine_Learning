import numpy as np


def stochastic_gradient_descent(data, labels, gradloss,
                                learning_rate=1):
    """Calculate updates using stochastic gradient descent algorithm

    :param data: numpy array of shape [N, dims] representing N datapoints
    :param labels: numpy array of shape [N]
                   representing datapoints' labels/classes
    :param gradloss: function, that calculates the gradient of the loss
                     for the given array of datapoints and labels
    :param learning_rate: gradient scaling parameter
    :yield: yields scaled gradient
    """
    # FIXME: yield gradient for each datapoint
    yield np.zeros(data.shape[1])
    yield np.arange(data.shape[1])

def minibatch_gradient_descent(data, labels, gradloss,
                               batch_size=10, learning_rate=1):
    """Calculate updates using minibatch gradient descent algorithm

    :param data: numpy array of shape [N, dims] representing N datapoints
    :param labels: numpy array of shape [N]
                   representing datapoints' labels/classes
    :param gradloss: function, that calculates the gradient of the loss
                     for the given array of datapoints and labels
    :param batch_size: number of datapoints in each batch
    :param learning_rate: gradient scaling parameter
    :yield: yields scaled gradient
    """
    # TODO: You need to split the data into batches of batch_size
    # If there is a remaining part with less length than batch_size
    # Then use that as a batch
    # FIXME: yield gradient for each batch of datapoints
    yield np.zeros(data.shape[1])
    yield np.ones(data.shape[1])

def batch_gradient_descent(data, labels, gradloss,
                           learning_rate=1):
    """Calculate updates using batch gradient descent algorithm

    :param data: numpy array of shape [N, dims] representing N datapoints
    :param labels: numpy array of shape [N]
                   representing datapoints' labels/classes
    :param gradloss: function, that calculates the gradient of the loss
                     for the given array of datapoints and labels
    :param learning_rate: gradient scaling parameter
    :yield: yields scaled gradient
    """
    # FIXME: yield the gradient of right scale
    yield np.ones(data.shape[1])

def newton_raphson_method(data, labels, gradloss, hessianloss):
    """Calculate updates using Newton-Raphson update formula

    :param data: numpy array of shape [N, dims] representing N datapoints
    :param labels: numpy array of shape [N]
                   representing datapoints' labels/classes
    :param gradloss: function, that calculates the gradient of the loss
                     for the given array of datapoints and labels
    :param hessianloss: function, that calculates the Hessian matrix
                        of the loss for the given array of datapoints and labels
    :yield: yield once the update of Newton-Raphson formula
    """
    gradient = gradloss(data, labels)
    hessian = hessianloss(data, labels)
    yield np.linalg.inv(hessian) @ gradient
