"""
Logistic Regression implementation for binary and multiclass classification.
"""

import numpy as np


class LogisticRegression:
    """
    Logistic Regression classifier using gradient descent.
    
    Parameters:
    -----------
    learning_rate : float, default=0.01
        The learning rate for gradient descent.
    max_iterations : int, default=1000
        Maximum number of iterations for gradient descent.
    tolerance : float, default=1e-6
        Tolerance for convergence in gradient descent.
    multi_class : str, default='ovr'
        Multiclass strategy: 'ovr' (one-vs-rest) or 'binary'
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, 
                 tolerance=1e-6, multi_class='ovr'):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.multi_class = multi_class
        self.weights = None
        self.bias = None
        self.classes = None
        self.cost_history = []
        self.classifiers = {}  # For multiclass
    
    def _sigmoid(self, z):
        """
        Sigmoid activation function with numerical stability.
        """
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _fit_binary(self, X, y):
        """
        Fit binary logistic regression.
        """
        m, n = X.shape
        self.weights = np.random.normal(0, 0.01, n)
        self.bias = 0
        
        prev_cost = float('inf')
        
        for i in range(self.max_iterations):
            # Forward pass
            z = X @ self.weights + self.bias
            y_pred = self._sigmoid(z)
            
            # Calculate cost (binary cross-entropy)
            # Add small epsilon to prevent log(0)
            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            cost = -(1/m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
            self.cost_history.append(cost)
            
            # Calculate gradients
            dw = (1/m) * X.T @ (y_pred - y)
            db = (1/m) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Check for convergence
            if abs(prev_cost - cost) < self.tolerance:
                break
            prev_cost = cost
    
    def fit(self, X, y):
        """
        Fit the logistic regression model to the training data.
        
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
        
        if len(self.classes) == 2:
            # Binary classification
            # Convert labels to 0 and 1
            y_binary = (y == self.classes[1]).astype(int)
            self._fit_binary(X, y_binary)
        else:
            # Multiclass classification using one-vs-rest
            if self.multi_class == 'ovr':
                for class_label in self.classes:
                    # Create binary problem: current class vs all others
                    y_binary = (y == class_label).astype(int)
                    
                    # Create and train binary classifier
                    binary_classifier = LogisticRegression(
                        learning_rate=self.learning_rate,
                        max_iterations=self.max_iterations,
                        tolerance=self.tolerance,
                        multi_class='binary'
                    )
                    binary_classifier._fit_binary(X, y_binary)
                    self.classifiers[class_label] = binary_classifier
    
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
        if self.weights is None and not self.classifiers:
            raise ValueError("Model must be fitted before making predictions.")
        
        X = np.array(X)
        
        if len(self.classes) == 2:
            # Binary classification
            z = X @ self.weights + self.bias
            prob_positive = self._sigmoid(z)
            return np.column_stack([1 - prob_positive, prob_positive])
        else:
            # Multiclass classification
            probabilities = []
            for class_label in self.classes:
                classifier = self.classifiers[class_label]
                z = X @ classifier.weights + classifier.bias
                prob = classifier._sigmoid(z)
                probabilities.append(prob)
            
            probabilities = np.column_stack(probabilities)
            # Normalize probabilities (softmax-like)
            probabilities = probabilities / np.sum(probabilities, axis=1, keepdims=True)
            return probabilities
    
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
        probabilities = self.predict_proba(X)
        predicted_indices = np.argmax(probabilities, axis=1)
        return self.classes[predicted_indices]
    
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