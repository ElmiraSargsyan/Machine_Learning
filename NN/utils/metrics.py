import numpy as np

"""
Define all metrics (e.g. accuracy, AUC, f1-score, etc.) here.
"""


def accuracy_score(Y_hat, Y_true):
    return np.sum(Y_hat == Y_true) / Y_true.shape[0]
