"""
Ridge Regression implementation with L2 regularization.
"""

import numpy as np


class RidgeRegression:
    """
    Ridge Regression model with L2 regularization.
    
    Parameters:
    -----------
    alpha : float, default=1.0
        Regularization strength. Higher values specify stronger regularization.
    learning_rate : float, default=0.01
        The learning rate for gradient descent.
    max_iterations : int, default=1000
        Maximum number of iterations for gradient descent.
    method : str, default='gradient_descent'
        Method to use: 'gradient_descent' or 'normal_equation'
    tolerance : float, default=1e-6
        Tolerance for convergence in gradient descent.
    """
    
    def __init__(self, alpha=1.0, learning_rate=0.01, max_iterations=1000,
                 method='gradient_descent', tolerance=1e-6):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.method = method
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def fit(self, X, y):
        """
        Fit the ridge regression model to the training data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        """
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        m, n = X.shape
        
        if self.method == 'normal_equation':
            # Normal equation with regularization: 
            # theta = (X^T * X + alpha * I)^(-1) * X^T * y
            X_with_bias = np.column_stack([np.ones(m), X])
            
            # Create identity matrix, but don't regularize bias term
            I = np.eye(n + 1)
            I[0, 0] = 0  # Don't regularize bias
            
            try:
                theta = np.linalg.inv(X_with_bias.T @ X_with_bias + self.alpha * I) @ X_with_bias.T @ y
                self.bias = theta[0]
                self.weights = theta[1:]
            except np.linalg.LinAlgError:
                # If matrix is singular, use pseudo-inverse
                theta = np.linalg.pinv(X_with_bias.T @ X_with_bias + self.alpha * I) @ X_with_bias.T @ y
                self.bias = theta[0]
                self.weights = theta[1:]
        else:
            # Gradient descent with regularization
            self.weights = np.random.normal(0, 0.01, n)
            self.bias = 0
            
            prev_cost = float('inf')
            
            for i in range(self.max_iterations):
                # Forward pass
                y_pred = X @ self.weights + self.bias
                
                # Calculate cost (MSE + L2 regularization)
                mse_cost = np.mean((y_pred - y) ** 2) / 2
                reg_cost = self.alpha * np.sum(self.weights ** 2) / (2 * m)
                cost = mse_cost + reg_cost
                self.cost_history.append(cost)
                
                # Calculate gradients
                dw = (1/m) * X.T @ (y_pred - y) + (self.alpha / m) * self.weights
                db = (1/m) * np.sum(y_pred - y)  # No regularization for bias
                
                # Update parameters
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
                
                # Check for convergence
                if abs(prev_cost - cost) < self.tolerance:
                    break
                prev_cost = cost
    
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
        if self.weights is None:
            raise ValueError("Model must be fitted before making predictions.")
        
        X = np.array(X)
        return X @ self.weights + self.bias
    
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
        y_pred = self.predict(X)
        y_mean = np.mean(y)
        ss_tot = np.sum((y - y_mean) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        return 1 - (ss_res / ss_tot)