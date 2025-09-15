"""
Decision Tree implementation for classification.
"""

import numpy as np
from collections import Counter


class Node:
    """
    Node class for decision tree.
    """
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # Index of feature to split on
        self.threshold = threshold  # Threshold value for split
        self.left = left           # Left child node
        self.right = right         # Right child node
        self.value = value         # Value if leaf node (class prediction)


class DecisionTree:
    """
    Decision Tree classifier using recursive binary splitting.
    
    Parameters:
    -----------
    max_depth : int, default=10
        Maximum depth of the tree.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node.
    """
    
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None
        self.classes = None
    
    def _gini_impurity(self, y):
        """
        Calculate Gini impurity for a set of labels.
        """
        proportions = np.bincount(y) / len(y)
        return 1 - np.sum(proportions ** 2)
    
    def _information_gain(self, parent, left_child, right_child):
        """
        Calculate information gain from a split.
        """
        n_parent = len(parent)
        n_left = len(left_child)
        n_right = len(right_child)
        
        if n_left == 0 or n_right == 0:
            return 0
        
        parent_gini = self._gini_impurity(parent)
        left_gini = self._gini_impurity(left_child)
        right_gini = self._gini_impurity(right_child)
        
        weighted_avg_gini = (n_left / n_parent) * left_gini + (n_right / n_parent) * right_gini
        return parent_gini - weighted_avg_gini
    
    def _best_split(self, X, y):
        """
        Find the best split for the given data.
        """
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        for feature in range(n_features):
            # Get unique values for this feature
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                # Split data
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                # Calculate information gain
                gain = self._information_gain(y, y[left_mask], y[right_mask])
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree.
        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Stopping criteria
        if (depth >= self.max_depth or 
            n_classes == 1 or 
            n_samples < self.min_samples_split):
            # Create leaf node
            most_common_class = Counter(y).most_common(1)[0][0]
            return Node(value=most_common_class)
        
        # Find best split
        best_feature, best_threshold, best_gain = self._best_split(X, y)
        
        if best_feature is None or best_gain <= 0:
            # No good split found, create leaf node
            most_common_class = Counter(y).most_common(1)[0][0]
            return Node(value=most_common_class)
        
        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Recursively build left and right subtrees
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return Node(feature=best_feature, threshold=best_threshold, 
                   left=left_child, right=right_child)
    
    def fit(self, X, y):
        """
        Fit the decision tree to the training data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        """
        X = np.array(X)
        y = np.array(y)
        
        self.classes = np.unique(y)
        self.root = self._build_tree(X, y)
    
    def _predict_sample(self, x, node):
        """
        Predict a single sample by traversing the tree.
        """
        if node.value is not None:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data.
            
        Returns:
        --------
        predictions : array of shape (n_samples,)
            Predicted class labels.
        """
        if self.root is None:
            raise ValueError("Model must be fitted before making predictions.")
        
        X = np.array(X)
        predictions = [self._predict_sample(x, self.root) for x in X]
        return np.array(predictions)
    
    def score(self, X, y):
        """
        Calculate accuracy score.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        y : array-like of shape (n_samples,)
            True labels.
            
        Returns:
        --------
        accuracy : float
            Accuracy score.
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)