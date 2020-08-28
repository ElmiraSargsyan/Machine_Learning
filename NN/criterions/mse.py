from criterions.criterion import Criterion
import numpy as np

"""
The **MSECriterion**, which is basic L2 norm usually used for regression.
"""
class MSECriterion(Criterion):
    def __init__(self):
        super(MSECriterion, self).__init__()

    def updateOutput(self, inpt, target):
        diff = inpt - target
        self.output = np.sum(np.square(diff)) / inpt.size

        return self.output 
 
    def updateGradInput(self, inpt, target):
        self.gradInput = 2 * (inpt - target) / inpt.size

        return self.gradInput

    def __repr__(self):
        return "MSECriterion"
