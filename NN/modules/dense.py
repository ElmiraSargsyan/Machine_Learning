import numpy as np

from modules.module import Module


class Dense(Module):
    """
    Also called "Linear" or "Fully Connected" layer. 
    A module which applies a linear transformation 
    A common name is fully-connected layer, dense layer, InnerProductLayer in caffe. 
    
    The module should work with 2D input of shape (n_samples, n_feature).
    """
    def __init__(self, n_in, n_out):
        super(Dense, self).__init__()
       
        # This is a nice initialization
        stdv = 1./np.sqrt(n_in)
        self.W = np.random.uniform(-stdv, stdv, size = (n_out, n_in))
        self.b = np.random.uniform(-stdv, stdv, size = n_out)
        
        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)
        
    def updateOutput(self, inpt):
        # <Your Code Goes Here>
        self.output = inpt @ self.W.T + self.b
               
        return self.output

    def updateGradInput(self, inpt, gradOutput):
        # <Your Code Goes Here>
        self.gradInput = gradOutput @ self.W
        
        return self.gradInput
    
    def accGradParameters(self, inpt, gradOutput):
        # <Your Code Goes Here>
        self.gradW = gradOutput.T @ inpt
        self.gradB = gradOutput.sum(axis = 0)
    
    def zeroGradParameters(self):
        self.gradW.fill(0)
        self.gradb.fill(0)
        
    def getParameters(self):
        return [self.W, self.b]
    
    def getGradParameters(self):
        return [self.gradW, self.gradb]
    
    def __repr__(self):
        s = self.W.shape
        q = 'Dense %d -> %d' %(s[1],s[0])
        return q