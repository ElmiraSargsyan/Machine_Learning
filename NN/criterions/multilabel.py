from criterions.criterion import Criterion
import numpy as np
"""
**MultiLabelCriterion** for atribute classification, i.e. target is multiple-hot
encoded, could be multiple ones i.e. sample can be classified to more than one
classes.
"""

class MultiLabelCriterion(Criterion):
    def __init__(self):
        super(MultiLabelCriterion, self).__init__()
    
    def updateOutput(self, inpt, target): 
        
        # Use this trick to avoid numerical errors
        input_clamp = np.maximum(1e-15, np.minimum(inpt, 1 - 1e-15) )
        
        self.output = - np.sum(target * np.log(input_clamp)) / inpt.size
        return self.output

    def updateGradInput(self, inpt, target):
        
        # Use this trick to avoid numerical errors
        input_clamp = np.maximum(1e-15, np.minimum(inpt, 1 - 1e-15) )
        self.gradInput = - target / (input_clamp * inpt.size)
        return self.gradInput
    
    def __repr__(self):
        return "MultiLabelCriterion"
