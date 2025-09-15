"""
Data preprocessing utilities.
"""

import numpy as np


class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.
    """
    
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.fitted = False
    
    def fit(self, X):
        """
        Compute the mean and std to be used for later scaling.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        """
        X = np.array(X)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Avoid division by zero
        self.std_[self.std_ == 0] = 1
        self.fitted = True
        return self
    
    def transform(self, X):
        """
        Perform standardization by centering and scaling.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Data to be scaled.
            
        Returns:
        --------
        X_scaled : array of shape (n_samples, n_features)
            Scaled data.
        """
        if not self.fitted:
            raise ValueError("StandardScaler must be fitted before transforming.")
        
        X = np.array(X)
        return (X - self.mean_) / self.std_
    
    def fit_transform(self, X):
        """
        Fit to data, then transform it.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data.
            
        Returns:
        --------
        X_scaled : array of shape (n_samples, n_features)
            Scaled data.
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X):
        """
        Scale back the data to the original representation.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Scaled data.
            
        Returns:
        --------
        X_original : array of shape (n_samples, n_features)
            Original scale data.
        """
        if not self.fitted:
            raise ValueError("StandardScaler must be fitted before inverse transforming.")
        
        X = np.array(X)
        return X * self.std_ + self.mean_


class MinMaxScaler:
    """
    Transform features by scaling each feature to a given range.
    """
    
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.scale_ = None
        self.fitted = False
    
    def fit(self, X):
        """
        Compute the minimum and range to be used for later scaling.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        """
        X = np.array(X)
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        self.range_ = self.max_ - self.min_
        # Avoid division by zero
        self.range_[self.range_ == 0] = 1
        
        feature_min, feature_max = self.feature_range
        self.scale_ = (feature_max - feature_min) / self.range_
        self.min_scaled_ = feature_min - self.min_ * self.scale_
        
        self.fitted = True
        return self
    
    def transform(self, X):
        """
        Scale features to the given range.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Data to be scaled.
            
        Returns:
        --------
        X_scaled : array of shape (n_samples, n_features)
            Scaled data.
        """
        if not self.fitted:
            raise ValueError("MinMaxScaler must be fitted before transforming.")
        
        X = np.array(X)
        return X * self.scale_ + self.min_scaled_
    
    def fit_transform(self, X):
        """
        Fit to data, then transform it.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data.
            
        Returns:
        --------
        X_scaled : array of shape (n_samples, n_features)
            Scaled data.
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X):
        """
        Undo the scaling of X according to feature_range.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Scaled data.
            
        Returns:
        --------
        X_original : array of shape (n_samples, n_features)
            Original scale data.
        """
        if not self.fitted:
            raise ValueError("MinMaxScaler must be fitted before inverse transforming.")
        
        X = np.array(X)
        return (X - self.min_scaled_) / self.scale_


def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split arrays into random train and test subsets.
    
    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        Training data.
    y : array-like of shape (n_samples,)
        Target values.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
    random_state : int, default=None
        Random seed for reproducibility.
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : arrays
        Split data.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    X = np.array(X)
    y = np.array(y)
    
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    
    # Create random indices
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]