"""
Utility functions for data preprocessing and dataset management.
"""

import numpy as np


def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split dataset into training and testing sets.
    
    Args:
        X (array-like): Features of shape (n_samples, n_features)
        y (array-like): Target values of shape (n_samples,)
        test_size (float): Proportion of dataset to include in test split (0.0 to 1.0)
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
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
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test


def normalize_features(X, method='standardize'):
    """
    Normalize features using standardization or min-max scaling.
    
    Args:
        X (array-like): Features to normalize of shape (n_samples, n_features)
        method (str): 'standardize' for z-score normalization or 'minmax' for min-max scaling
    
    Returns:
        tuple: (X_normalized, scaler_params) where scaler_params contains normalization parameters
    """
    X = np.array(X)
    
    if method == 'standardize':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        # Avoid division by zero
        std = np.where(std == 0, 1, std)
        X_normalized = (X - mean) / std
        scaler_params = {'mean': mean, 'std': std, 'method': 'standardize'}
        
    elif method == 'minmax':
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)
        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals = np.where(range_vals == 0, 1, range_vals)
        X_normalized = (X - min_vals) / range_vals
        scaler_params = {'min': min_vals, 'max': max_vals, 'method': 'minmax'}
        
    else:
        raise ValueError("Method must be 'standardize' or 'minmax'")
    
    return X_normalized, scaler_params


def apply_normalization(X, scaler_params):
    """
    Apply previously computed normalization parameters to new data.
    
    Args:
        X (array-like): Features to normalize
        scaler_params (dict): Normalization parameters from normalize_features
    
    Returns:
        np.ndarray: Normalized features
    """
    X = np.array(X)
    
    if scaler_params['method'] == 'standardize':
        return (X - scaler_params['mean']) / scaler_params['std']
    elif scaler_params['method'] == 'minmax':
        range_vals = scaler_params['max'] - scaler_params['min']
        range_vals = np.where(range_vals == 0, 1, range_vals)
        return (X - scaler_params['min']) / range_vals
    else:
        raise ValueError("Unknown normalization method in scaler_params")


def generate_regression_data(n_samples=100, n_features=1, noise=0.1, random_state=None):
    """
    Generate synthetic regression dataset.
    
    Args:
        n_samples (int): Number of samples
        n_features (int): Number of features
        noise (float): Standard deviation of noise
        random_state (int): Random seed
    
    Returns:
        tuple: (X, y) where X is features and y is target values
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    X = np.random.randn(n_samples, n_features)
    true_coefficients = np.random.randn(n_features)
    y = X @ true_coefficients + noise * np.random.randn(n_samples)
    
    return X, y


def generate_classification_data(n_samples=100, n_features=2, random_state=None):
    """
    Generate synthetic binary classification dataset.
    
    Args:
        n_samples (int): Number of samples
        n_features (int): Number of features
        random_state (int): Random seed
    
    Returns:
        tuple: (X, y) where X is features and y is binary labels (0 or 1)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate two clusters
    cluster_1 = np.random.multivariate_normal([1, 1], [[0.5, 0], [0, 0.5]], n_samples // 2)
    cluster_2 = np.random.multivariate_normal([-1, -1], [[0.5, 0], [0, 0.5]], n_samples - n_samples // 2)
    
    X = np.vstack([cluster_1, cluster_2])
    y = np.hstack([np.ones(n_samples // 2), np.zeros(n_samples - n_samples // 2)])
    
    # If we need more features, add noise features
    if n_features > 2:
        noise_features = np.random.randn(n_samples, n_features - 2)
        X = np.column_stack([X, noise_features])
    elif n_features == 1:
        # Use only first feature
        X = X[:, :1]
    
    # Shuffle the data
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    return X, y