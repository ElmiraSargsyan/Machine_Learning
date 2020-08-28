import numpy as np

def generate_data(n, seed=0):
    np.random.seed(seed)
    cov = np.array([[3, 1],
                    [1, 2]])
    blue_mean = np.array([-2, 0])
    red_mean = np.array([4, -1])
    blues = np.random.multivariate_normal(blue_mean, cov, n)
    reds = np.random.multivariate_normal(red_mean, cov, n)
    return blues, reds
