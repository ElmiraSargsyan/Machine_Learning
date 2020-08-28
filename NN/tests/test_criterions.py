import unittest
import numpy as np

from criterions import *

EPS = 1e-7
input_shape = (5,3)


def check_grad(crt: Criterion,inpt, target,h = 1e-7):
    """
    This function is for testing. Do not change this. This evaluates gradients
    numerically.
    """
    output = crt.forward(inpt,target)
    
    gradInput = crt.backward(inpt, target)

    gradInputEstimate = np.zeros_like(gradInput)

    for i in range(input_shape[0]):
        for j in range(input_shape[1]):
            inpt_d = inpt.copy()
            inpt_d[i,j] += h
            output_d = crt.forward(inpt_d, target)
            grad_estimate = ((output_d - output)/h)
            gradInputEstimate[i,j] = grad_estimate
    return gradInput, gradInputEstimate


class TestCriterions(unittest.TestCase):
    def test_mse_grad(self):
        crt = MSECriterion()

        grad, grad_estimate = check_grad(crt,np.random.randn(*input_shape), 
                                             np.random.randn(*input_shape))
        relative_diff = abs((grad-grad_estimate)/(grad_estimate+EPS)).max()
        self.assertTrue(relative_diff<1e-4)

    def test_crossenthropy_grad(self):
        crt = CrossEntropyCriterion()

        grad, grad_estimate = check_grad(crt, np.random.rand(*input_shape),
            np.stack([np.ones(input_shape[0]),np.zeros(input_shape[0]),np.zeros(input_shape[0])]).T)

        relative_diff = abs((grad-grad_estimate)/(grad_estimate+EPS)).max()
        self.assertTrue(relative_diff<1e-4)
    
    def test_multiclass_grad(self):
        crt = MultiLabelCriterion()

        grad, grad_estimate = check_grad(crt, np.random.rand(*input_shape),
            np.stack([np.ones(input_shape[0]),np.zeros(input_shape[0]),np.ones(input_shape[0])]).T)

        relative_diff = abs((grad-grad_estimate)/(grad_estimate+EPS)).max()
        self.assertTrue(relative_diff<1e-4)
