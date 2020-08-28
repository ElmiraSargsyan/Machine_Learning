import numpy as np


def linear_kernel(u, v):
    """
    Calculates linear kernel u.T * v.

    :param u: Numpy array with shape (N,).
    :param v: Numpy array with shape (N,).
    :return: Float number.
    >>> linear_kernel(np.array([2, 3]), np.array([3, 4]))
    18
    """
    return np.dot(u.T, v)

def polynomial_kernel(u, v, p=3):
    """
    Calculates polynomial kernel wih degree equal to p.

    :param u: Numpy array with shape (N,).
    :param v: Numpy array with shape (N,).
    :param p: Degree of polynomial to calculate with.
    :return: Float number.
    >>> polynomial_kernel(np.array([2, 3]), np.array([3, 4]), p=2)
    361
    """
    c = 1
    return (np.dot(u.T, v) + c) ** p
    


def gaussian_kernel(u, v, sigma=5.0):
    """
    Calculates Gaussian kernel (radial basis function).
    http://web.mit.edu/6.034/wwwbob/svm-notes-long-08.pdf

    :param u: Numpy array with shape (N,).
    :param v: Numpy array with shape (N,).
    :param sigma: Parameter sigma inside kernel function.
    :return: Float number.
    >>> p = gaussian_kernel(np.array([2, 3]), np.array([3, 4]))
    >>> np.isclose(p, 0.96078943, atol=1e-8)
    True
    """
    return np.exp(-1/(2*sigma**2) * (np.dot(u.T, u) + np.dot(v.T, v) - 2*np.dot(u.T, v)))
 
