"""
Sample dataset generators and loaders.
"""

import numpy as np


def make_regression_data(n_samples=100, n_features=1, noise=0.1, random_state=None):
    """
    Generate a random regression problem.
    
    Parameters:
    -----------
    n_samples : int, default=100
        Number of samples.
    n_features : int, default=1
        Number of features.
    noise : float, default=0.1
        Standard deviation of the Gaussian noise.
    random_state : int, default=None
        Random seed for reproducibility.
        
    Returns:
    --------
    X : array of shape (n_samples, n_features)
        Input samples.
    y : array of shape (n_samples,)
        Target values.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    X = np.random.randn(n_samples, n_features)
    
    # Generate true coefficients
    true_coef = np.random.randn(n_features)
    
    # Generate target with some nonlinearity and noise
    y = X @ true_coef + 0.5 * np.sum(X**2, axis=1) + noise * np.random.randn(n_samples)
    
    return X, y


def make_classification_data(n_samples=100, n_features=2, n_classes=2, 
                           n_clusters_per_class=1, random_state=None):
    """
    Generate a random classification problem.
    
    Parameters:
    -----------
    n_samples : int, default=100
        Number of samples.
    n_features : int, default=2
        Number of features.
    n_classes : int, default=2
        Number of classes.
    n_clusters_per_class : int, default=1
        Number of clusters per class.
    random_state : int, default=None
        Random seed for reproducibility.
        
    Returns:
    --------
    X : array of shape (n_samples, n_features)
        Input samples.
    y : array of shape (n_samples,)
        Target values.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    samples_per_class = n_samples // n_classes
    samples_per_cluster = samples_per_class // n_clusters_per_class
    
    X = []
    y = []
    
    for class_id in range(n_classes):
        for cluster_id in range(n_clusters_per_class):
            # Generate cluster center
            center = np.random.randn(n_features) * 3
            
            # Generate samples around center
            cluster_samples = np.random.randn(samples_per_cluster, n_features) + center
            cluster_labels = np.full(samples_per_cluster, class_id)
            
            X.append(cluster_samples)
            y.append(cluster_labels)
    
    X = np.vstack(X)
    y = np.hstack(y)
    
    # Shuffle the data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y


def load_iris_like():
    """
    Generate an Iris-like dataset for multiclass classification.
    
    Returns:
    --------
    X : array of shape (150, 4)
        Input samples (sepal length, sepal width, petal length, petal width).
    y : array of shape (150,)
        Target values (0: setosa, 1: versicolor, 2: virginica).
    feature_names : list
        Names of the features.
    target_names : list
        Names of the target classes.
    """
    np.random.seed(42)  # For reproducibility
    
    # Generate three classes with different characteristics
    n_samples_per_class = 50
    
    # Class 0: Setosa (smaller flowers)
    setosa_X = np.random.multivariate_normal(
        mean=[5.0, 3.4, 1.5, 0.2],
        cov=[[0.12, 0.10, 0.01, 0.01],
             [0.10, 0.14, 0.01, 0.01],
             [0.01, 0.01, 0.03, 0.01],
             [0.01, 0.01, 0.01, 0.01]],
        size=n_samples_per_class
    )
    setosa_y = np.zeros(n_samples_per_class, dtype=int)
    
    # Class 1: Versicolor (medium flowers)
    versicolor_X = np.random.multivariate_normal(
        mean=[5.9, 2.8, 4.3, 1.3],
        cov=[[0.27, 0.09, 0.17, 0.05],
             [0.09, 0.10, 0.08, 0.04],
             [0.17, 0.08, 0.22, 0.07],
             [0.05, 0.04, 0.07, 0.04]],
        size=n_samples_per_class
    )
    versicolor_y = np.ones(n_samples_per_class, dtype=int)
    
    # Class 2: Virginica (larger flowers)
    virginica_X = np.random.multivariate_normal(
        mean=[6.6, 3.0, 5.6, 2.0],
        cov=[[0.40, 0.09, 0.30, 0.05],
             [0.09, 0.10, 0.07, 0.05],
             [0.30, 0.07, 0.30, 0.05],
             [0.05, 0.05, 0.05, 0.07]],
        size=n_samples_per_class
    )
    virginica_y = np.full(n_samples_per_class, 2, dtype=int)
    
    # Combine all classes
    X = np.vstack([setosa_X, versicolor_X, virginica_X])
    y = np.hstack([setosa_y, versicolor_y, virginica_y])
    
    # Shuffle the data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    target_names = ['setosa', 'versicolor', 'virginica']
    
    return X, y, feature_names, target_names


def load_boston_like():
    """
    Generate a Boston housing-like dataset for regression.
    
    Returns:
    --------
    X : array of shape (506, 13)
        Input samples (house features).
    y : array of shape (506,)
        Target values (house prices in thousands).
    feature_names : list
        Names of the features.
    """
    np.random.seed(42)  # For reproducibility
    
    n_samples = 506
    
    # Generate correlated features similar to Boston housing
    # Features: crime rate, zoned lots, industrial, river, nox, rooms, age, 
    #          distance, highways, tax, pupil-teacher, low status, distance to employment
    
    # Start with some base features
    crime = np.random.exponential(scale=3, size=n_samples)
    zoned_lots = np.random.uniform(0, 100, size=n_samples)
    industrial = np.random.uniform(0, 30, size=n_samples)
    river = np.random.choice([0, 1], size=n_samples, p=[0.93, 0.07])
    nox = np.random.uniform(0.3, 0.9, size=n_samples)
    rooms = np.random.normal(6.3, 0.7, size=n_samples)
    age = np.random.uniform(0, 100, size=n_samples)
    distance = np.random.uniform(1, 12, size=n_samples)
    highways = np.random.uniform(1, 24, size=n_samples)
    tax = np.random.uniform(150, 750, size=n_samples)
    pupil_teacher = np.random.uniform(12, 22, size=n_samples)
    low_status = np.random.uniform(1, 38, size=n_samples)
    employment_distance = np.random.uniform(1, 12, size=n_samples)
    
    X = np.column_stack([
        crime, zoned_lots, industrial, river, nox, rooms, age,
        distance, highways, tax, pupil_teacher, low_status, employment_distance
    ])
    
    # Generate target based on features with some noise
    y = (50 - 0.5 * crime - 0.1 * industrial + 5 * river - 20 * nox + 
         9 * rooms - 0.1 * age - 1.5 * distance - 0.3 * highways - 
         0.01 * tax - 1.0 * pupil_teacher - 0.5 * low_status - 
         0.7 * employment_distance + np.random.normal(0, 3, size=n_samples))
    
    # Ensure positive prices
    y = np.maximum(y, 5)
    
    feature_names = [
        'crime_rate', 'zoned_lots', 'industrial', 'river', 'nox_concentration',
        'avg_rooms', 'age', 'distance_employment', 'highway_accessibility',
        'tax_rate', 'pupil_teacher_ratio', 'low_status_population', 'employment_distance'
    ]
    
    return X, y, feature_names