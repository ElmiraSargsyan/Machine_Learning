#from __future__ import annotations
import numpy as np
import math

class DecisionNode:
    def __init__(self,
                 data: np.ndarray,
                 labels: np.ndarray,
                 column: int = None,
                 value: float = None,
                 column_type: str = None,
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
        self.column_type = column_type
        self.false_branch = false_branch
        self.true_branch = true_branch
        self.is_leaf = is_leaf
        


class DecisionTree:

    def __init__(self,
                 num_features: int,
                 max_tree_depth=4,
                 max_features: int or float = 0.8,
                 min_samples: int = 2,
                 criterion="gini",
                 task="classification"):
        self.tree = None
        self.max_depth = max_tree_depth
        self.task = task
        self.num_features = num_features
        self.min_samples = min_samples
        
        if isinstance(max_features, int):
            self.max_features = max_features
        elif 0 < max_features <= 1.0:
            self.max_features = int(np.round(max_features * num_features))

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

        if len(labels) == self.min_samples:
            # return a node is_leaf=True
            return DecisionNode(data = data, labels = labels, is_leaf=True)
        
        impurity = self.criterion(labels)
        column_type, best_column, best_value = None, None, None
        
        # picking random features
        
        random_data = data[:, np.random.choice(self.num_features, self.max_features, replace = False)]
        
        for column, column_values in enumerate(random_data.T):
            
            # if column contains categorical features
            if len(np.unique(column_values)) <= 10:
                
                for split_value in np.unique(column_values):
                                
                    left_labels = labels[column_values ==  split_value]
                    right_labels = labels[column_values !=  split_value]
                    
                    #checking for valid splits
                    if len(left_labels) >= self.min_samples and len(right_labels) >= self.min_samples:
                        left_criterion = self.criterion(left_labels)
                        right_criterion = self.criterion(right_labels)

                        criterion =  (left_criterion*len(left_labels) + right_criterion*len(right_labels)) / len(labels)
                        
                        if criterion < impurity: 
                        
                            impurity = criterion
                            best_column = column
                            best_value = split_value
                            column_type = "categorical"
                        
                    
             
            # if column contains numeric features
            else:
                for split_value in np.arange(
                        min(column_values), max(column_values),
                        (max(column_values) - min(column_values)) / 50):
                                      
                    left_labels = labels[column_values <  split_value]
                    right_labels = labels[column_values >=  split_value]
                    
                    #checking for valid splits
                    if len(left_labels) >= self.min_samples and len(right_labels) >= self.min_samples:
                        left_criterion = self.criterion(left_labels)
                        right_criterion = self.criterion(right_labels)

                        criterion =  (left_criterion*len(left_labels) + right_criterion*len(right_labels)) / len(labels)
                        
                        if criterion < impurity: 
                        
                            impurity = criterion
                            best_column = column
                            best_value = split_value
                            column_type = "numeric"
               
            if best_column is None or current_depth == self.max_depth:
                
                return DecisionNode(data = data, labels = labels, is_leaf = True)
            
            elif column_type == "categorical":
                
                left = data[(data[:, [best_column]]).flatten() ==  best_value]
                left_labels = labels[(data[:, [best_column]]).flatten() == best_value]
                
                right = data[(data[:, [best_column]]).flatten() !=  best_value]
                right_labels = labels[(data[:, [best_column]]).flatten() !=  best_value]
                
            elif column_type == "numeric":
            
                left = data[(data[:, [best_column]]).flatten() <  best_value]
                left_labels = labels[(data[:, [best_column]]).flatten() <  best_value]
                
                right = data[(data[:, [best_column]]).flatten() >=  best_value]
                right_labels = labels[(data[:, [best_column]]).flatten() >=  best_value]
                
            #Returning DecisionNode
            return DecisionNode(data = data, labels = labels, column = best_column, value = best_value, column_type = column_type,
                                true_branch =  self._iterate(np.array(left), np.array(left_labels), current_depth = current_depth + 1), 
                                false_branch = self._iterate(np.array(right), np.array(right_labels), current_depth = current_depth + 1))

    def fit(self, data: np.ndarray, labels: np.ndarray):
        self.tree = self._iterate(data, labels)

    def predict(self, point: np.ndarray, tree = None) -> float or int:
    
        """
        This method iterates nodes starting with the first node i. e.
        self.tree. Returns predicted label of a given point (example [2.5, 6],
        where 2.5 and 6 are points features).

        """
        
        if tree:
            node = tree
        else:
            node = self.tree

        while True:
            if node.is_leaf:
                    counts = np.bincount(node.labels)
                    return np.argmax(counts)
                
            if node.column_type == "numeric":
                
                if point[node.column] < node.value:
                    node = node.true_branch
                else:
                    node = node.false_branch
            elif node.column_type == "categorical":
            
                if point[node.column] == node.value:
                    node = node.true_branch
                else:
                    node = node.false_branch
""

#***********************************************************************************

class RandomForestClassifier(object):
    def __init__(self, 
                 n_estimators: int = 10,
                 split_size: float = 0.6,
                 max_features: int or float = 0.6,
                 max_depth = 4,
                 criterion = "gini"                                
                 ):
        """
        :param n_estimators: number of trees in the forest
        :param split_size: number of samples to train each base estimators
        :param max_features: number of features to be considered in each time looking for the best split
        :param max_depth: the maximum depth of the tree
        :param criterion: the function for measuring the quality of a split.
        
        :attr estimators_: the list of fitted sub-estimators / list of DecisionTree objects /
        """
        self.n_estimators = n_estimators
        
        if 0 < split_size <= 1.0:
            self.split_size = split_size
        else:
            raise RuntimeError(f"Invalid split_size. It must be in range (0, 1]: '{split_size}'")
            
        if isinstance(max_features, int):
            self.max_features = max_features
        elif 0 < max_features <= 1.0:
            self.max_features = max_features
        else:
           raise RuntimeError(f"Invalid max_features. It must be in range (0, 1]: '{max_features}'")
            
            
        self.max_depth = max_depth
        self.criterion = criterion
        
        self.estimators_ = []
            

    def fit(self, data: np.ndarray, labels: np.ndarray):
        """
        :param data: array of features for each point
        :param labels: array of labels for each point
        """
        
        N, features = data.shape
        
        for i in range(self.n_estimators):
            indexes = np.random.choice(range(N), int(np.round(self.split_size * N)), replace = False)
            splitted_data = data[indexes]
            splitted_labels = labels[indexes]
            
            estimator = DecisionTree(max_tree_depth = self.max_depth,
                                     criterion = self.criterion,
                                     task = "classification",
                                     max_features = self.max_features,
                                     num_features = features)
            
            estimator.fit(splitted_data, splitted_labels)
            self.estimators_.append(estimator.tree)

    def predict(self, data: np.ndarray) -> np.ndarray:
        result = []
        for sample in data:
            predictions = [DecisionTree.predict(estimator, sample, tree = estimator) for estimator in self.estimators_]
            result.append(np.bincount(predictions).argmax())
        return result
        
    
    
def f1_score(y_true: np.ndarray, y_predicted: np.ndarray):
    """
    only 0 and 1 should be accepted labels and 1 is the positive class
    """   
    assert set(y_true).union({1, 0}) == {1, 0}
    TN, TP, FN, FP = 0, 0, 0, 0
    for y, y_ in zip(y_true, y_predicted):

        if y == 0 and y_ == 0:
            TN = TN + 1
            continue
        if y == 1 and y_ == 1:
            TP = TP + 1
            continue
        if y == 1 and y_ == 0:
            FN = FN + 1
            continue
        if y == 0 and y_ == 1:
            FP = FP + 1
    try:
        recall =  TP / (FN + TP)
        precision = TP / (FP + TP)
            
        return 2 * ((precision * recall) / (precision + recall))
    except ZeroDivisionError:
        return 0
    


def data_preprocess(data: np.array) -> np.array:
    return data

#TUNING
import pandas as pd
from sklearn.model_selection import KFold
       

def tune(X, Y):
    
    n__estimators = [10]
    split__size = np.arange(0.5, 1.0, 0.1)
    max__features = np.arange(0.5, 1.0, 0.1)
    
    N =len(n__estimators)*len(split__size)*len(max__features)
    count = 1
    accuracy = 0
    
    for n_estimators in n__estimators:
        for split_size in split__size:
            for max_features in max__features:
                acc = []
                print(f"[TUNING] {count} out of {N}")
                count = count + 1
                
                kf = KFold(n_splits=5)
                for train_index, test_index in kf.split(X):
    
                    X_train, x_test = X[train_index], X[test_index]
                    Y_train, y_test = Y[train_index], Y[test_index]
                
                    RF = RandomForestClassifier(n_estimators = n_estimators, split_size = split_size, max_features = max_features)
                    RF.fit(X_train, Y_train)
                    
                    acc.append(f1_score(y_test, RF.predict(x_test)))
                    
                if np.mean(acc) > accuracy:
                    accuracy = np.mean(acc)
                    best_n_estimators = n_estimators
                    best_split_size = split_size
                    best_max_features = max_features
                
                    print("[NUMBER OF ESTIMATORS]", best_n_estimators)   
                    print("[SPLIT SIZE]          ", best_split_size) 
                    print("[MAX FEATURES]        ", best_max_features)   
                    print("[ACCURACY]            ", accuracy)

model = RandomForestClassifier()

train_data = pd.read_csv("../data/train.csv", encoding = "latin1")
df = pd.DataFrame(train_data)

labels = train_data['label'].values
x_train = np.array(train_data.drop('label', axis=1))
y_train = labels

x_train = data_preprocess(x_train)

tune(x_train, y_train)

       
        
        
        
  
