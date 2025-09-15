"""
Evaluation metrics for supervised learning models.
"""

import numpy as np


def mean_squared_error(y_true, y_pred):
    """
    Calculate Mean Squared Error.
    
    Args:
        y_true (array-like): True target values
        y_pred (array-like): Predicted values
    
    Returns:
        float: Mean squared error
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    return np.mean((y_true - y_pred) ** 2)


def r2_score(y_true, y_pred):
    """
    Calculate R² (coefficient of determination) score.
    
    Args:
        y_true (array-like): True target values
        y_pred (array-like): Predicted values
    
    Returns:
        float: R² score
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    return 1 - (ss_res / ss_tot)


def accuracy_score(y_true, y_pred):
    """
    Calculate classification accuracy.
    
    Args:
        y_true (array-like): True class labels
        y_pred (array-like): Predicted class labels
    
    Returns:
        float: Accuracy score (fraction of correct predictions)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    return np.mean(y_true == y_pred)


def mean_absolute_error(y_true, y_pred):
    """
    Calculate Mean Absolute Error.
    
    Args:
        y_true (array-like): True target values
        y_pred (array-like): Predicted values
    
    Returns:
        float: Mean absolute error
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    return np.mean(np.abs(y_true - y_pred))


def confusion_matrix(y_true, y_pred):
    """
    Calculate confusion matrix for binary classification.
    
    Args:
        y_true (array-like): True class labels (0 or 1)
        y_pred (array-like): Predicted class labels (0 or 1)
    
    Returns:
        np.ndarray: 2x2 confusion matrix [[TN, FP], [FN, TP]]
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    
    return np.array([[tn, fp], [fn, tp]])