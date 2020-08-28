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
    # Yield gradient for each datapoint

    indexes = np.arange(len(data))
    np.random.shuffle(indexes)

    data = [data[i] for i in indexes]
    labels = [labels[i] for i in indexes]

    
    for datapoint, label in zip(data, labels):      
        yield learning_rate * gradloss(np.array([datapoint]), np.array([label]))


def minibatch_gradient_descent(data, labels, gradloss,
                               batch_size=10, learning_rate=0.1):
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
    # Split the data into batches of batch_size
    # If there is a remaining part with less length than batch_size
    # Then use that as a batch
    # Yield gradient for each batch of datapoints
    
    if batch_size == 1:
        for datapoint, label in zip(data, labels):      
            yield learning_rate * gradloss(np.array([datapoint]), np.array([label]))
            
    else:
        indexes = np.arange(len(data))
        np.random.shuffle(indexes)
        
        data = [data[i] for i in indexes]
        labels = [labels[i] for i in indexes]
        
        N = len(data) // batch_size + 1*(len(data) // batch_size != len(data) / batch_size)
        
        for i in range(N):
            data_ = np.array(data[i * batch_size:(i + 1) * batch_size])
            labels_ = np.array(labels[i * batch_size:(i + 1) * batch_size])

            yield learning_rate * gradloss(data_, labels_)

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
    # Yield the gradient of right scale
    yield np.array(gradloss(data, labels))

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
