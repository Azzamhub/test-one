"""
Simple usage examples for the ML library.
"""

import sys
import os

# Add the ml_library to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_library.regression import LinearRegression
from ml_library.classification import LogisticRegression, RandomForest
from ml_library.utils import train_test_split, StandardScaler, accuracy_score, r2_score
from ml_library.datasets import make_regression_data, make_classification_data, load_iris_like


def simple_regression_example():
    """Simple regression example."""
    print("Simple Regression Example")
    print("-" * 30)
    
    # Generate data
    X, y = make_regression_data(n_samples=100, n_features=1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate
    score = model.score(X_test, y_test)
    print(f"R² Score: {score:.4f}")
    
    return model


def simple_classification_example():
    """Simple classification example."""
    print("\nSimple Classification Example")
    print("-" * 30)
    
    # Load data
    X, y, _, _ = load_iris_like()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    return model


def random_forest_example():
    """Random Forest example."""
    print("\nRandom Forest Example")
    print("-" * 30)
    
    # Generate binary classification data
    X, y = make_classification_data(n_samples=200, n_features=4, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Train Random Forest
    rf = RandomForest(n_estimators=50, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest Accuracy: {accuracy:.4f}")
    
    # Show probability predictions
    probabilities = rf.predict_proba(X_test[:5])
    print("\nProbability predictions for first 5 samples:")
    for i, prob in enumerate(probabilities):
        print(f"Sample {i+1}: Class 0: {prob[0]:.3f}, Class 1: {prob[1]:.3f}")
    
    return rf


if __name__ == "__main__":
    print("ML Library - Simple Usage Examples")
    print("=" * 40)
    
    # Run examples
    regression_model = simple_regression_example()
    classification_model = simple_classification_example()
    rf_model = random_forest_example()
    
    print("\n" + "=" * 40)
    print("All examples completed successfully!")
    print("Check out examples/comprehensive_demo.py for more detailed examples.")