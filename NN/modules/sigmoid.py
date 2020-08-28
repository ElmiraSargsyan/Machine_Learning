import numpy as np

from modules.module import Module

"""
Implement well-known **Sigmoid** non-linearity
"""


class Sigmoid(Module):
    def __init__(self):
         super(Sigmoid, self).__init__()
    
    def updateOutput(self, inpt):
        self.output = 1/(1 + np.exp(-inpt))

        return self.output
    
    def updateGradInput(self, inpt, gradOutput):
        # <Your Code Goes Here>
        sigmoid = 1/(1 + np.exp(-inpt))
        self.gradInput = np.multiply(gradOutput, sigmoid * (1 - sigmoid))
        return self.gradInput
    
    def __repr__(self):
        return "Sigmoid"
