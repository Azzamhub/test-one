"""
Evaluation metrics for machine learning models.
"""

import numpy as np


def accuracy_score(y_true, y_pred):
    """
    Calculate accuracy classification score.
    
    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
        
    Returns:
    --------
    accuracy : float
        Accuracy score.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(y_true == y_pred)


def precision_score(y_true, y_pred, average='binary', pos_label=1):
    """
    Calculate precision score.
    
    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    average : str, default='binary'
        Type of averaging: 'binary', 'macro', 'micro'
    pos_label : int, default=1
        The positive label for binary classification.
        
    Returns:
    --------
    precision : float
        Precision score.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if average == 'binary':
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
        
        if tp + fp == 0:
            return 0.0
        return tp / (tp + fp)
    
    elif average == 'macro':
        classes = np.unique(y_true)
        precisions = []
        for cls in classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fp = np.sum((y_true != cls) & (y_pred == cls))
            
            if tp + fp == 0:
                precisions.append(0.0)
            else:
                precisions.append(tp / (tp + fp))
        
        return np.mean(precisions)
    
    elif average == 'micro':
        tp_total = 0
        fp_total = 0
        classes = np.unique(y_true)
        
        for cls in classes:
            tp_total += np.sum((y_true == cls) & (y_pred == cls))
            fp_total += np.sum((y_true != cls) & (y_pred == cls))
        
        if tp_total + fp_total == 0:
            return 0.0
        return tp_total / (tp_total + fp_total)


def recall_score(y_true, y_pred, average='binary', pos_label=1):
    """
    Calculate recall score.
    
    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    average : str, default='binary'
        Type of averaging: 'binary', 'macro', 'micro'
    pos_label : int, default=1
        The positive label for binary classification.
        
    Returns:
    --------
    recall : float
        Recall score.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if average == 'binary':
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
        
        if tp + fn == 0:
            return 0.0
        return tp / (tp + fn)
    
    elif average == 'macro':
        classes = np.unique(y_true)
        recalls = []
        for cls in classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fn = np.sum((y_true == cls) & (y_pred != cls))
            
            if tp + fn == 0:
                recalls.append(0.0)
            else:
                recalls.append(tp / (tp + fn))
        
        return np.mean(recalls)
    
    elif average == 'micro':
        tp_total = 0
        fn_total = 0
        classes = np.unique(y_true)
        
        for cls in classes:
            tp_total += np.sum((y_true == cls) & (y_pred == cls))
            fn_total += np.sum((y_true == cls) & (y_pred != cls))
        
        if tp_total + fn_total == 0:
            return 0.0
        return tp_total / (tp_total + fn_total)


def f1_score(y_true, y_pred, average='binary', pos_label=1):
    """
    Calculate F1 score.
    
    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    average : str, default='binary'
        Type of averaging: 'binary', 'macro', 'micro'
    pos_label : int, default=1
        The positive label for binary classification.
        
    Returns:
    --------
    f1 : float
        F1 score.
    """
    precision = precision_score(y_true, y_pred, average=average, pos_label=pos_label)
    recall = recall_score(y_true, y_pred, average=average, pos_label=pos_label)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)


def mean_squared_error(y_true, y_pred):
    """
    Calculate mean squared error.
    
    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.
        
    Returns:
    --------
    mse : float
        Mean squared error.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)


def r2_score(y_true, y_pred):
    """
    Calculate R-squared (coefficient of determination) score.
    
    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.
        
    Returns:
    --------
    r2 : float
        R-squared score.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    y_mean = np.mean(y_true)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    
    if ss_tot == 0:
        return 1.0
    
    return 1 - (ss_res / ss_tot)