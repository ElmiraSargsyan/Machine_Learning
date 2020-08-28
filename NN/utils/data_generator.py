import numpy as np


def onehot(y, K=None):
    assert len(y.shape) == 1
    if K is None:
        K = np.max(y)
    Y = np.zeros((len(y), K + 1))
    Y[range(len(y)), y] = 1
    return Y


def generate_two_classes(N: int = 500):
    X1 = np.random.randn(N, 2) + np.array([2, 2])
    X2 = np.random.randn(N, 2) + np.array([-2, -2])

    Y = np.concatenate([np.ones(N), np.zeros(N)])[:, None]
    Y = np.hstack([Y, 1 - Y])
    X = np.vstack([X1, X2])

    return X, Y


def generate_spirale(N: int = 100, K: int = 3):
    """
    N -  number of datapoints
    K - number of classes
    """
    D = 2  # dimensionality
    p = 4  # polynomial degree
    X = np.zeros((N * K, D))  # data matrix (each row = single example)
    y = np.zeros(N * K, dtype='uint8')  # class labels
    for j in range(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N)  # radius
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(
            N) * 0.4  # theta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
    return X, onehot(y)


def generate_cat_eye(N: int = 500):
    a = np.random.rand(N) * 8 - 4
    X = np.sqrt(16 - a ** 2) * ((np.random.rand(N) > 0.5) * 2 - 1)
    X1 = np.vstack((a * 2, X)).T + np.random.randn(N, 2) * 0.2
    X2 = np.vstack((X * 2, a)).T + np.random.randn(N, 2) * 0.2
    X3 = np.random.randn(N, 2)
    X3[:, 0] *= 0.7
    Y = np.concatenate([np.zeros(N), np.zeros(N), np.ones(N)])[:, None]
    Y = np.hstack([Y, 1 - Y])

    X = np.vstack([X1, X2, X3])
    return X, Y
