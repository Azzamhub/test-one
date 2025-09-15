"""
Basic tests for the ML library.
"""

import unittest
import numpy as np
import sys
import os

# Add the ml_library to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_library.regression import LinearRegression, PolynomialRegression, RidgeRegression
from ml_library.classification import LogisticRegression, DecisionTree, RandomForest
from ml_library.utils import StandardScaler, MinMaxScaler, train_test_split
from ml_library.utils import accuracy_score, mean_squared_error, r2_score
from ml_library.datasets import make_regression_data, make_classification_data


class TestRegression(unittest.TestCase):
    """Test regression algorithms."""
    
    def setUp(self):
        """Set up test data."""
        self.X, self.y = make_regression_data(n_samples=100, n_features=1, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
    
    def test_linear_regression(self):
        """Test linear regression."""
        lr = LinearRegression(max_iterations=500)
        lr.fit(self.X_train, self.y_train)
        y_pred = lr.predict(self.X_test)
        
        self.assertIsNotNone(lr.weights)
        self.assertIsNotNone(lr.bias)
        self.assertEqual(len(y_pred), len(self.y_test))
        
        r2 = r2_score(self.y_test, y_pred)
        self.assertGreater(r2, 0.2)  # Should have reasonable performance
    
    def test_polynomial_regression(self):
        """Test polynomial regression."""
        pr = PolynomialRegression(degree=2, max_iterations=500)
        pr.fit(self.X_train, self.y_train)
        y_pred = pr.predict(self.X_test)
        
        self.assertEqual(len(y_pred), len(self.y_test))
        
        r2 = r2_score(self.y_test, y_pred)
        self.assertGreater(r2, 0.5)  # Should perform better than linear
    
    def test_ridge_regression(self):
        """Test ridge regression."""
        rr = RidgeRegression(alpha=1.0, max_iterations=500)
        rr.fit(self.X_train, self.y_train)
        y_pred = rr.predict(self.X_test)
        
        self.assertIsNotNone(rr.weights)
        self.assertIsNotNone(rr.bias)
        self.assertEqual(len(y_pred), len(self.y_test))
        
        r2 = r2_score(self.y_test, y_pred)
        self.assertGreater(r2, 0.2)


class TestClassification(unittest.TestCase):
    """Test classification algorithms."""
    
    def setUp(self):
        """Set up test data."""
        self.X, self.y = make_classification_data(n_samples=100, n_features=2, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
    
    def test_logistic_regression(self):
        """Test logistic regression."""
        lr = LogisticRegression(max_iterations=500)
        lr.fit(self.X_train, self.y_train)
        y_pred = lr.predict(self.X_test)
        
        self.assertEqual(len(y_pred), len(self.y_test))
        
        acc = accuracy_score(self.y_test, y_pred)
        self.assertGreater(acc, 0.7)  # Should have reasonable accuracy
    
    def test_decision_tree(self):
        """Test decision tree."""
        dt = DecisionTree(max_depth=5)
        dt.fit(self.X_train, self.y_train)
        y_pred = dt.predict(self.X_test)
        
        self.assertIsNotNone(dt.root)
        self.assertEqual(len(y_pred), len(self.y_test))
        
        acc = accuracy_score(self.y_test, y_pred)
        self.assertGreater(acc, 0.6)
    
    def test_random_forest(self):
        """Test random forest."""
        rf = RandomForest(n_estimators=10, max_depth=5, random_state=42)
        rf.fit(self.X_train, self.y_train)
        y_pred = rf.predict(self.X_test)
        
        self.assertEqual(len(rf.trees), 10)
        self.assertEqual(len(y_pred), len(self.y_test))
        
        acc = accuracy_score(self.y_test, y_pred)
        self.assertGreater(acc, 0.6)


class TestUtils(unittest.TestCase):
    """Test utility functions."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.X = np.random.randn(100, 3)
        self.y = np.random.randint(0, 2, 100)
    
    def test_standard_scaler(self):
        """Test standard scaler."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        
        # Check that mean is approximately 0 and std is approximately 1
        self.assertTrue(np.allclose(np.mean(X_scaled, axis=0), 0, atol=1e-10))
        self.assertTrue(np.allclose(np.std(X_scaled, axis=0), 1, atol=1e-10))
    
    def test_minmax_scaler(self):
        """Test min-max scaler."""
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(self.X)
        
        # Check that min is 0 and max is 1
        self.assertTrue(np.allclose(np.min(X_scaled, axis=0), 0, atol=1e-10))
        self.assertTrue(np.allclose(np.max(X_scaled, axis=0), 1, atol=1e-10))
    
    def test_train_test_split(self):
        """Test train-test split."""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        self.assertEqual(len(X_train), 80)
        self.assertEqual(len(X_test), 20)
        self.assertEqual(len(y_train), 80)
        self.assertEqual(len(y_test), 20)


if __name__ == '__main__':
    unittest.main()