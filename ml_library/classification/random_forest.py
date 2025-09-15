"""
Random Forest implementation for classification.
"""

import numpy as np
from .decision_tree import DecisionTree
from collections import Counter


class RandomForest:
    """
    Random Forest classifier using ensemble of decision trees.
    
    Parameters:
    -----------
    n_estimators : int, default=100
        Number of trees in the forest.
    max_depth : int, default=10
        Maximum depth of the trees.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node.
    max_features : str or int, default='sqrt'
        Number of features to consider for each split:
        - 'sqrt': sqrt(n_features)
        - 'log2': log2(n_features) 
        - int: exact number
        - None: all features
    bootstrap : bool, default=True
        Whether to use bootstrap sampling.
    random_state : int, default=None
        Random seed for reproducibility.
    """
    
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=2,
                 min_samples_leaf=1, max_features='sqrt', bootstrap=True, 
                 random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees = []
        self.feature_indices = []
        self.classes = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _get_bootstrap_sample(self, X, y):
        """
        Create a bootstrap sample from the training data.
        """
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]
    
    def _get_feature_subset_size(self, n_features):
        """
        Determine the number of features to use for each tree.
        """
        if self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        else:
            return n_features
    
    def fit(self, X, y):
        """
        Fit the random forest to the training data.
        
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
        n_samples, n_features = X.shape
        max_features_per_tree = self._get_feature_subset_size(n_features)
        
        self.trees = []
        self.feature_indices = []
        
        for i in range(self.n_estimators):
            # Create decision tree
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
            
            # Get bootstrap sample
            if self.bootstrap:
                X_sample, y_sample = self._get_bootstrap_sample(X, y)
            else:
                X_sample, y_sample = X, y
            
            # Select random subset of features
            feature_indices = np.random.choice(
                n_features, max_features_per_tree, replace=False
            )
            self.feature_indices.append(feature_indices)
            
            # Train tree on selected features
            X_subset = X_sample[:, feature_indices]
            tree.fit(X_subset, y_sample)
            self.trees.append(tree)
    
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
        if not self.trees:
            raise ValueError("Model must be fitted before making predictions.")
        
        X = np.array(X)
        n_samples = X.shape[0]
        
        # Collect predictions from all trees
        tree_predictions = []
        for i, tree in enumerate(self.trees):
            feature_indices = self.feature_indices[i]
            X_subset = X[:, feature_indices]
            predictions = tree.predict(X_subset)
            tree_predictions.append(predictions)
        
        # Convert to array for easier processing
        tree_predictions = np.array(tree_predictions)
        
        # Majority voting
        final_predictions = []
        for sample_idx in range(n_samples):
            sample_predictions = tree_predictions[:, sample_idx]
            most_common = Counter(sample_predictions).most_common(1)[0][0]
            final_predictions.append(most_common)
        
        return np.array(final_predictions)
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data.
            
        Returns:
        --------
        probabilities : array of shape (n_samples, n_classes)
            Predicted probabilities for each class.
        """
        if not self.trees:
            raise ValueError("Model must be fitted before making predictions.")
        
        X = np.array(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        
        # Collect predictions from all trees
        tree_predictions = []
        for i, tree in enumerate(self.trees):
            feature_indices = self.feature_indices[i]
            X_subset = X[:, feature_indices]
            predictions = tree.predict(X_subset)
            tree_predictions.append(predictions)
        
        # Convert to array for easier processing
        tree_predictions = np.array(tree_predictions)
        
        # Calculate probabilities based on voting
        probabilities = np.zeros((n_samples, n_classes))
        
        for sample_idx in range(n_samples):
            sample_predictions = tree_predictions[:, sample_idx]
            for class_idx, class_label in enumerate(self.classes):
                # Count votes for this class
                votes = np.sum(sample_predictions == class_label)
                probabilities[sample_idx, class_idx] = votes / self.n_estimators
        
        return probabilities
    
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