import numpy as np
from modules.module import Module

"""
Implement **Rectified Linear Unit** non-linearity (aka **ReLU**): 
"""


class ReLU(Module):
    def __init__(self):
         super(ReLU, self).__init__()
    
    def updateOutput(self, inpt):
        self.output = np.maximum(inpt, np.zeros_like(inpt))
        
        return self.output
    
    def updateGradInput(self, inpt, gradOutput):
        # The gradient of ReLU is 1 for x>0 and 0 for x<0 
        self.gradInput = np.multiply(gradOutput, np.maximum(np.sign(inpt), np.zeros_like(inpt)))

        return self.gradInput
    
    def __repr__(self):
        return "ReLU"
