"""
Linear Regression implementation for supervised learning.
"""

import numpy as np


class LinearRegression:
    """
    Simple Linear Regression using ordinary least squares.
    
    Attributes:
        coefficients_ (np.ndarray): Model coefficients after fitting
        intercept_ (float): Model intercept after fitting
    """
    
    def __init__(self):
        self.coefficients_ = None
        self.intercept_ = None
        self.is_fitted_ = False
    
    def fit(self, X, y):
        """
        Fit the linear regression model.
        
        Args:
            X (array-like): Training features of shape (n_samples, n_features)
            y (array-like): Training targets of shape (n_samples,)
        
        Returns:
            self: Returns the fitted estimator
        """
        X = np.array(X)
        y = np.array(y)
        
        # Add bias term (intercept)
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        
        # Compute coefficients using normal equation: (X^T X)^-1 X^T y
        try:
            coefficients = np.linalg.solve(X_with_bias.T @ X_with_bias, X_with_bias.T @ y)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if matrix is singular
            coefficients = np.linalg.pinv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
        
        self.intercept_ = coefficients[0]
        self.coefficients_ = coefficients[1:]
        self.is_fitted_ = True
        
        return self
    
    def predict(self, X):
        """
        Make predictions using the fitted model.
        
        Args:
            X (array-like): Features to predict on of shape (n_samples, n_features)
        
        Returns:
            np.ndarray: Predicted values of shape (n_samples,)
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.array(X)
        return X @ self.coefficients_ + self.intercept_
    
    def score(self, X, y):
        """
        Calculate R² score (coefficient of determination).
        
        Args:
            X (array-like): Test features
            y (array-like): True target values
        
        Returns:
            float: R² score
        """
        from .metrics import r2_score
        y_pred = self.predict(X)
        return r2_score(y, y_pred)