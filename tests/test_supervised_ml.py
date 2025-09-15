"""
Basic tests for the supervised machine learning library.
"""

import numpy as np
import sys
import os

# Add the parent directory to the path so we can import supervised_ml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from supervised_ml import (
    LinearRegression, LogisticRegression, 
    mean_squared_error, accuracy_score, r2_score,
    train_test_split, normalize_features
)
from supervised_ml.utils import generate_regression_data, generate_classification_data


def test_linear_regression():
    """Test Linear Regression functionality."""
    print("Testing Linear Regression...")
    
    # Generate simple data
    X, y = generate_regression_data(n_samples=50, n_features=1, noise=0.1, random_state=42)
    
    # Fit model
    model = LinearRegression()
    model.fit(X, y)
    
    # Check that model was fitted
    assert model.is_fitted_, "Model should be fitted"
    assert model.coefficients_ is not None, "Coefficients should not be None"
    assert model.intercept_ is not None, "Intercept should not be None"
    
    # Make predictions
    y_pred = model.predict(X)
    assert len(y_pred) == len(y), "Predictions should have same length as targets"
    
    # Calculate R² score
    r2 = r2_score(y, y_pred)
    assert r2 > 0.5, f"R² score should be reasonable, got {r2}"
    
    print("  ✓ Linear Regression tests passed")


def test_logistic_regression():
    """Test Logistic Regression functionality."""
    print("Testing Logistic Regression...")
    
    # Generate simple classification data
    X, y = generate_classification_data(n_samples=100, n_features=2, random_state=42)
    
    # Fit model
    model = LogisticRegression(learning_rate=0.1, max_iter=500)
    model.fit(X, y)
    
    # Check that model was fitted
    assert model.is_fitted_, "Model should be fitted"
    assert model.coefficients_ is not None, "Coefficients should not be None"
    assert model.intercept_ is not None, "Intercept should not be None"
    
    # Make predictions
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)
    
    assert len(y_pred) == len(y), "Predictions should have same length as targets"
    assert y_prob.shape == (len(y), 2), "Probabilities should have shape (n_samples, 2)"
    assert np.all((y_pred == 0) | (y_pred == 1)), "Predictions should be 0 or 1"
    assert np.allclose(np.sum(y_prob, axis=1), 1.0), "Probabilities should sum to 1"
    
    # Calculate accuracy
    accuracy = accuracy_score(y, y_pred)
    assert accuracy > 0.6, f"Accuracy should be reasonable, got {accuracy}"
    
    print("  ✓ Logistic Regression tests passed")


def test_metrics():
    """Test evaluation metrics."""
    print("Testing Metrics...")
    
    # Test regression metrics
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 2.9, 3.8, 5.2])
    
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    assert mse > 0, "MSE should be positive"
    assert r2 <= 1, "R² should be <= 1"
    
    # Test classification metrics
    y_true_class = np.array([0, 1, 1, 0, 1])
    y_pred_class = np.array([0, 1, 0, 0, 1])
    
    accuracy = accuracy_score(y_true_class, y_pred_class)
    assert 0 <= accuracy <= 1, "Accuracy should be between 0 and 1"
    
    print("  ✓ Metrics tests passed")


def test_utils():
    """Test utility functions."""
    print("Testing Utilities...")
    
    # Test train_test_split
    X = np.random.randn(100, 3)
    y = np.random.randn(100)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    assert len(X_train) + len(X_test) == len(X), "Split should preserve total samples"
    assert len(y_train) + len(y_test) == len(y), "Split should preserve total targets"
    assert len(X_test) == int(0.2 * len(X)), "Test size should be approximately correct"
    
    # Test normalization
    X_norm, scaler_params = normalize_features(X, method='standardize')
    assert X_norm.shape == X.shape, "Normalized data should have same shape"
    assert 'mean' in scaler_params, "Scaler params should contain mean"
    assert 'std' in scaler_params, "Scaler params should contain std"
    
    # Test data generation
    X_reg, y_reg = generate_regression_data(n_samples=50, n_features=2, random_state=42)
    assert X_reg.shape == (50, 2), "Generated regression data should have correct shape"
    assert len(y_reg) == 50, "Generated targets should have correct length"
    
    X_clf, y_clf = generate_classification_data(n_samples=50, n_features=2, random_state=42)
    assert X_clf.shape == (50, 2), "Generated classification data should have correct shape"
    assert len(y_clf) == 50, "Generated labels should have correct length"
    assert set(y_clf) == {0, 1}, "Generated labels should be binary"
    
    print("  ✓ Utilities tests passed")


def run_tests():
    """Run all tests."""
    print("=== Running Supervised ML Tests ===\n")
    
    try:
        test_linear_regression()
        test_logistic_regression()
        test_metrics()
        test_utils()
        
        print("\n=== All Tests Passed! ===")
        return True
    
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)