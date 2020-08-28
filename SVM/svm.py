import cvxopt
import numpy as np

from kernel import linear_kernel


class SVM:
    def __init__(self,
                 C,
                 kernel=linear_kernel,
                 kernel_params={}):
        self.C = C and float(C)
        self.kernel = kernel
        self.kernel_params = kernel_params

        self.b = None  # intercept
        self.w = None  # you can use this if kernel is linear for faster computation
        self.sv_alphas = None  # solution of qp problem which satisfy condition
        self.sv_s = None  # array of data of support vector points
        self.sv_labels = None  # array of labels of support vector points

    def gram_matrix(self, data):
        """Calculates P matrix.
        WARNING! - You need to implement at least linear_kernel function inside
                   kernel.py before using this function.
        :param data: Data to compute P matrix with [N, M] shape.
        :return: P matrix inside quadratic problem calculated with kernel
                 function.
        """
        
        P = []
        for d1 in data:
            p = []
            for d2 in data:
                p.append(self.kernel(d1,d2))
            P.append(p)
        
        return P

    def fit(self, data, labels):
        """
        Fit SVM model to data by updating w and b attributes.
        NOTE - There is already a function in cvxopt to solve svm (softmargin)
               problem. But we are recommending to implement using cvxopt
               quadratic programming problem solver.

        :param data: Data with [N, M] shape.
        :param labels: Labels with [N] shape.
        :return: Void.
        """
        m,n = data.shape
        
        P = cvxopt.matrix(self.gram_matrix(data))
        q = cvxopt.matrix(-np.ones((m, 1)))
        G = cvxopt.matrix(-np.eye(m))
        h = cvxopt.matrix(np.zeros(m))
        A = cvxopt.matrix(labels.reshape(1, -1), tc='d')
        
        b = cvxopt.matrix(np.zeros(1))
        
        self.sv_alphas = np.array(cvxopt.solvers.qp(P, q, G, h, A, b)['x'])
        
        sv = (self.sv_alphas > 1e-5).flatten()
        
        #self.sv_alphas = alphas[sv]
        self.sv_s = data[sv]
        self.sv_labels = labels[sv]
        
        #self.b = self.sv_labels - np.dot(self.sv_s, self.w)
        
        sv_alphas = self.sv_alphas[sv]    
        self.b = 0.
        for n, a in enumerate(sv_alphas):
            self.b += self.sv_labels[n]
            sum = 0
            for m, x in enumerate(self.sv_s):
                sum += sv_alphas[m] * self.sv_labels[m] * self.kernel(self.sv_s[n],self.sv_s[n])
            self.b -= sum
        self.b *= 1/len(sv)
        
        self.w = np.sum(((labels * self.sv_alphas).T @ data), axis = 0)
        
        
        
    def predict(self, data):
        """
        Predict labels for data.

        :param data: Data with [N, M] shape.
        :return: Predicted labels for given data with [N] shape.
        """
        return np.sign(np.dot(data, self.w) + self.b)