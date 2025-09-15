"""
Polynomial Regression implementation.
"""

import numpy as np
from .linear_regression import LinearRegression


class PolynomialRegression:
    """
    Polynomial Regression model that extends linear regression with polynomial features.
    
    Parameters:
    -----------
    degree : int, default=2
        The degree of the polynomial features.
    learning_rate : float, default=0.01
        The learning rate for gradient descent.
    max_iterations : int, default=1000
        Maximum number of iterations for gradient descent.
    method : str, default='gradient_descent'
        Method to use: 'gradient_descent' or 'normal_equation'
    tolerance : float, default=1e-6
        Tolerance for convergence in gradient descent.
    """
    
    def __init__(self, degree=2, learning_rate=0.01, max_iterations=1000,
                 method='gradient_descent', tolerance=1e-6):
        self.degree = degree
        self.linear_model = LinearRegression(
            learning_rate=learning_rate,
            max_iterations=max_iterations,
            method=method,
            tolerance=tolerance
        )
    
    def _create_polynomial_features(self, X):
        """
        Create polynomial features up to the specified degree.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data.
            
        Returns:
        --------
        X_poly : array of shape (n_samples, poly_features)
            Polynomial features.
        """
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        X_poly = X.copy()
        
        # Add polynomial features
        for degree in range(2, self.degree + 1):
            X_poly = np.column_stack([X_poly, X ** degree])
        
        return X_poly
    
    def fit(self, X, y):
        """
        Fit the polynomial regression model to the training data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        """
        X_poly = self._create_polynomial_features(X)
        self.linear_model.fit(X_poly, y)
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data.
            
        Returns:
        --------
        y_pred : array of shape (n_samples,)
            Predicted values.
        """
        X_poly = self._create_polynomial_features(X)
        return self.linear_model.predict(X_poly)
    
    def score(self, X, y):
        """
        Calculate R-squared score.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        y : array-like of shape (n_samples,)
            True values.
            
        Returns:
        --------
        score : float
            R-squared score.
        """
        X_poly = self._create_polynomial_features(X)
        return self.linear_model.score(X_poly, y)
    
    @property
    def cost_history(self):
        """Get the cost history from the underlying linear model."""
        return self.linear_model.cost_history