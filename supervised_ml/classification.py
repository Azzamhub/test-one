"""
Logistic Regression implementation for binary classification.
"""

import numpy as np


class LogisticRegression:
    """
    Logistic Regression for binary classification using gradient descent.
    
    Attributes:
        coefficients_ (np.ndarray): Model coefficients after fitting
        intercept_ (float): Model intercept after fitting
        learning_rate (float): Learning rate for gradient descent
        max_iter (int): Maximum number of iterations for gradient descent
    """
    
    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.coefficients_ = None
        self.intercept_ = None
        self.is_fitted_ = False
    
    def _sigmoid(self, z):
        """Sigmoid activation function."""
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """
        Fit the logistic regression model using gradient descent.
        
        Args:
            X (array-like): Training features of shape (n_samples, n_features)
            y (array-like): Training targets of shape (n_samples,) with values 0 or 1
        
        Returns:
            self: Returns the fitted estimator
        """
        X = np.array(X)
        y = np.array(y)
        
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.coefficients_ = np.zeros(n_features)
        self.intercept_ = 0
        
        # Gradient descent
        for i in range(self.max_iter):
            # Forward pass
            linear_pred = X @ self.coefficients_ + self.intercept_
            y_pred = self._sigmoid(linear_pred)
            
            # Compute cost (log-likelihood)
            cost = self._compute_cost(y, y_pred)
            
            # Compute gradients
            dw = (1 / n_samples) * X.T @ (y_pred - y)
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.coefficients_ -= self.learning_rate * dw
            self.intercept_ -= self.learning_rate * db
        
        self.is_fitted_ = True
        return self
    
    def _compute_cost(self, y_true, y_pred):
        """Compute logistic regression cost function."""
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        cost = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return cost
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X (array-like): Features to predict on of shape (n_samples, n_features)
        
        Returns:
            np.ndarray: Predicted probabilities of shape (n_samples, 2)
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.array(X)
        linear_pred = X @ self.coefficients_ + self.intercept_
        prob_positive = self._sigmoid(linear_pred)
        prob_negative = 1 - prob_positive
        
        return np.column_stack([prob_negative, prob_positive])
    
    def predict(self, X):
        """
        Make binary predictions.
        
        Args:
            X (array-like): Features to predict on of shape (n_samples, n_features)
        
        Returns:
            np.ndarray: Predicted class labels (0 or 1) of shape (n_samples,)
        """
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] >= 0.5).astype(int)
    
    def score(self, X, y):
        """
        Calculate accuracy score.
        
        Args:
            X (array-like): Test features
            y (array-like): True target values
        
        Returns:
            float: Accuracy score
        """
        from .metrics import accuracy_score
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)