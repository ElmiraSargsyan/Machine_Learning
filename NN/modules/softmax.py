import numpy as np

from modules.module import Module

"""
This one is probably the hardest but as others only takes 5 lines of code in total. 
- input:   **batch_size x n_feats**
- output: **batch_size x n_feats**
"""


class SoftMax(Module):
    def __init__(self):
         super(SoftMax, self).__init__()
    
    def updateOutput(self, inpt):
        exp = np.exp(inpt)
        self.output = exp / np.sum(exp, axis = 1, keepdims=True)
        
        return self.output
    
    def updateGradInput(self, inpt, gradOutput):

        inpt = self.output
        diff = np.sum(inpt * gradOutput, axis = 1, keepdims=True)
        self.gradInput = inpt * (gradOutput - diff)

        return self.gradInput
    
    def __repr__(self):
        return "SoftMax"
