import numpy as np

from modules.module import Module

"""
Implement **SoftPlus**
(https://en.wikipedia.org/wiki%2FRectifier_%28neural_networks%29) activations.
Look, how they look a lot like ReLU.
"""


class SoftPlus(Module):
    def __init__(self):
        super(SoftPlus, self).__init__()
    
    def updateOutput(self, inpt):
        #  if x>30  log(1+exp(x)) ~= log(exp(x)) = x
        #self.output = (inpt <= 30) * np.log(1 + np.exp(inpt)) + (inpt > 30) * inpt
        self.output = np.log(1 + np.exp(inpt))
        return self.output
    
    def updateGradInput(self, inpt, gradOutput):
        sigmoid = 1/(1 + np.exp(-inpt))
        self.gradInput = np.multiply(gradOutput, sigmoid)
        return self.gradInput
    
    def __repr__(self):
        return "SoftPlus"
