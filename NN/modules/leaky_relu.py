import numpy as np
from modules.module import Module

"""
Implement **Leaky Rectified Linear Unit**
(http://en.wikipedia.org/wiki%2FRectifier_%28neural_networks%29%23Leaky_ReLUs). 
Expriment with slope. 
"""


class LeakyReLU(Module):
    def __init__(self, slope = 0.03):
        super(LeakyReLU, self).__init__()
            
        self.slope = slope
        
    def updateOutput(self, inpt):
        #self.output = np.maximum(inpt, self.slope * inpt)
        self.output = inpt * (1 - (1 -self.slope) * (inpt < 0))
        return self.output
    
    def updateGradInput(self, inpt, gradOutput):
        #d = np.abs(np.maximum(np.sign(input), self.slope * np.sign(inpt)))
        #self.gradInput = np.multiply(gradOutput, d)

        self.gradInput = gradOutput * (1 - (inpt < 0) * (1 - self.slope))
        return self.gradInput
    
    def __repr__(self):
        return "LeakyReLU"
