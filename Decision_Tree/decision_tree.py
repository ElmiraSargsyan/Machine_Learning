#from __future__ import annotations
import numpy as np
import math

class DecisionNode:
    def __init__(self,
                 data: np.ndarray,
                 labels: np.ndarray,
                 column: int = None,
                 value: float = None,
                 false_branch = None,
                 true_branch = None,
                 is_leaf: bool = False):
        """
        Building block of the decision tree.

        :param data: numpy 2d array data can for example be
         np.array([[1, 2], [2, 6], [1, 7]])
         where [1, 2], [2, 6], and [1, 7] represent each data point
        :param labels: numpy 1d array
         labels indicate which class each point belongs to
        :param column: the index of feature by which data is splitted
        :param value: column's splitting value
        :param true_branch(false_branch): child decision node
        true_branch(false_branch) is DecisionNode instance that contains data
        that satisfies(doesn't satisfy) the filter condition.
        :param is_leaf: is true when node has no child

        """
        self.data = data
        self.labels = labels
        self.column = column
        self.value = value
        self.false_branch = false_branch
        self.true_branch = true_branch
        self.is_leaf = is_leaf


class DecisionTree:

    def __init__(self,
                 max_tree_depth=4,
                 criterion="gini",
                 task="classification"):
        self.tree = None
        self.max_depth = max_tree_depth
        self.task = task

        if criterion == "entropy":
            self.criterion = self._entropy
        elif criterion == "square_loss":
            self.criterion = self._square_loss
        elif criterion == "gini":
            self.criterion = self._gini
        else:
            raise RuntimeError(f"Unknown criterion: '{criterion}'")

    @staticmethod
    def _gini(labels: np.ndarray) -> float:
        """
        Gini criterion for classification tasks.

        """
        p_sum = 0
        unique, counts = np.unique(labels, return_counts=True)
        c_sum = counts.sum() ** 2
        for count in counts:
            p_sum = p_sum + count ** 2 / c_sum
        
        return 1.0 - p_sum
        

    @staticmethod
    def _entropy(labels: np.ndarray) -> float:
        """
        Entropy criterion for classification tasks.

        """
        p_sum = 0
        unique, counts = np.unique(labels, return_counts=True)
        c_sum = counts.sum()
        for count in counts:
            p = count / c_sum
            p_sum = p_sum + p * math.log2(p)
        if p_sum:
            return -1.0 *  p_sum
        else:
            return p_sum

    @staticmethod
    def _square_loss(labels: np.ndarray) -> float:
        """
        Square loss criterion for regression tasks.

        """
        variance_sum = sum((labels - labels.mean())**2)
        return variance_sum / len(labels)

    def _iterate(self,
                 data: np.ndarray,
                 labels: np.ndarray,
                 current_depth=0) -> DecisionNode:
        """
        This method creates the whole decision tree, by recursively iterating
         through nodes.
        It returns the first node (DecisionNode object) of the decision tree,
         with it's child nodes, and child nodes' children, ect.
        """

        if len(labels) == 1:
            # return a node is_leaf=True
            return DecisionNode(data, labels, is_leaf=True)
            

        impurity = self.criterion(labels)
        best_column, best_value = None, None
      
        for column, column_values in enumerate(data.T):
            for split_value in np.arange(
                    min(column_values), max(column_values),
                    (max(column_values) - min(column_values)) / 50):
                # find optimal way of splitting the data
                
                left_labels = labels[column_values <  split_value]
                right_labels = labels[column_values >=  split_value]
                
                left_criterion = self.criterion(left_labels)
                right_criterion = self.criterion(right_labels)

                criterion =  (left_criterion*len(left_labels) + right_criterion*len(right_labels)) / len(labels)
                
                if criterion < impurity: 
                
                    impurity = criterion
                    best_column = column
                    best_value = split_value
                        
        if best_column is None or current_depth == self.max_depth:
            
            return DecisionNode(data, labels, is_leaf=True)
        
        else:
        
            # return DecisionNode with true(false)_branch=self._iterate(...)
            left = data[(data[:, [best_column]]).flatten() <  best_value]
            left_labels = labels[(data[:, [best_column]]).flatten() <  best_value]
            
            right = data[(data[:, [best_column]]).flatten() >=  best_value]
            right_labels = labels[(data[:, [best_column]]).flatten() >=  best_value]
            
            return DecisionNode(data, labels, column = best_column, value = best_value, 
                                true_branch =  self._iterate(np.array(left), np.array(left_labels), current_depth = current_depth + 1), 
                                false_branch = self._iterate(np.array(right), np.array(right_labels), current_depth = current_depth + 1))
            

    def fit(self, data: np.ndarray, labels: np.ndarray):
        self.tree = self._iterate(data, labels)

    def predict(self, point: np.ndarray) -> float or int:
    
        """
        This method iterates nodes starting with the first node i. e.
        self.tree. Returns predicted label of a given point (example [2.5, 6],
        where 2.5 and 6 are points features).

        """
        node = self.tree

        while True:
            if node.is_leaf:
                if self.task == "classification":
                    # predict and return the label for classification task
                    counts = np.bincount(node.labels)
                    return np.argmax(counts)
                else:
                    # predict and return the label for regression task
                    return np.mean(node.labels)

            if point[node.column] >= node.value:
                node = node.false_branch
            else:
                node = node.true_branch
""