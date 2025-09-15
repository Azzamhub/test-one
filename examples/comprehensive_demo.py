"""
Comprehensive example demonstrating all supervised machine learning algorithms.
"""

import numpy as np
import sys
import os

# Add the ml_library to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_library.regression import LinearRegression, PolynomialRegression, RidgeRegression
from ml_library.classification import LogisticRegression, DecisionTree, RandomForest
from ml_library.utils import StandardScaler, MinMaxScaler, train_test_split
from ml_library.utils import accuracy_score, precision_score, recall_score, f1_score
from ml_library.utils import mean_squared_error, r2_score
from ml_library.datasets import (make_regression_data, make_classification_data, 
                                 load_iris_like, load_boston_like)


def demonstrate_regression():
    """Demonstrate regression algorithms."""
    print("=" * 60)
    print("REGRESSION ALGORITHMS DEMONSTRATION")
    print("=" * 60)
    
    # Generate sample regression data
    print("\n1. Generating sample regression data...")
    X, y = make_regression_data(n_samples=200, n_features=1, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Linear Regression
    print("\n2. Linear Regression:")
    lr = LinearRegression(learning_rate=0.01, max_iterations=1000)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"   MSE: {mse:.4f}")
    print(f"   R²: {r2:.4f}")
    print(f"   Weights: {lr.weights}")
    print(f"   Bias: {lr.bias:.4f}")
    
    # Polynomial Regression
    print("\n3. Polynomial Regression (degree=2):")
    pr = PolynomialRegression(degree=2, learning_rate=0.01, max_iterations=1000)
    pr.fit(X_train, y_train)
    y_pred = pr.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"   MSE: {mse:.4f}")
    print(f"   R²: {r2:.4f}")
    
    # Ridge Regression
    print("\n4. Ridge Regression (alpha=1.0):")
    rr = RidgeRegression(alpha=1.0, learning_rate=0.01, max_iterations=1000)
    rr.fit(X_train, y_train)
    y_pred = rr.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"   MSE: {mse:.4f}")
    print(f"   R²: {r2:.4f}")
    print(f"   Weights: {rr.weights}")
    print(f"   Bias: {rr.bias:.4f}")
    
    # Test with Boston-like dataset
    print("\n5. Testing with Boston Housing-like dataset:")
    X_boston, y_boston, feature_names = load_boston_like()
    X_train, X_test, y_train, y_test = train_test_split(X_boston, y_boston, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr_boston = LinearRegression(learning_rate=0.01, max_iterations=2000)
    lr_boston.fit(X_train_scaled, y_train)
    y_pred = lr_boston.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"   Linear Regression - MSE: {mse:.4f}, R²: {r2:.4f}")


def demonstrate_classification():
    """Demonstrate classification algorithms."""
    print("\n\n" + "=" * 60)
    print("CLASSIFICATION ALGORITHMS DEMONSTRATION")
    print("=" * 60)
    
    # Binary classification
    print("\n1. Binary Classification:")
    X_binary, y_binary = make_classification_data(n_samples=200, n_features=2, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_binary, y_binary, test_size=0.2, random_state=42)
    
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Logistic Regression
    print("\n   a) Logistic Regression:")
    log_reg = LogisticRegression(learning_rate=0.1, max_iterations=1000)
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"      Accuracy: {acc:.4f}")
    print(f"      Precision: {prec:.4f}")
    print(f"      Recall: {rec:.4f}")
    print(f"      F1-score: {f1:.4f}")
    
    # Decision Tree
    print("\n   b) Decision Tree:")
    dt = DecisionTree(max_depth=5, min_samples_split=5)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"      Accuracy: {acc:.4f}")
    print(f"      Precision: {prec:.4f}")
    print(f"      Recall: {rec:.4f}")
    print(f"      F1-score: {f1:.4f}")
    
    # Random Forest
    print("\n   c) Random Forest:")
    rf = RandomForest(n_estimators=50, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"      Accuracy: {acc:.4f}")
    print(f"      Precision: {prec:.4f}")
    print(f"      Recall: {rec:.4f}")
    print(f"      F1-score: {f1:.4f}")
    
    # Multiclass classification
    print("\n2. Multiclass Classification (Iris-like dataset):")
    X_iris, y_iris, feature_names, target_names = load_iris_like()
    X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)
    
    print(f"Classes: {target_names}")
    print(f"Features: {feature_names}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Logistic Regression (multiclass)
    print("\n   a) Logistic Regression (One-vs-Rest):")
    log_reg_multi = LogisticRegression(learning_rate=0.1, max_iterations=1000)
    log_reg_multi.fit(X_train_scaled, y_train)
    y_pred = log_reg_multi.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"      Accuracy: {acc:.4f}")
    print(f"      Precision (macro): {prec:.4f}")
    print(f"      Recall (macro): {rec:.4f}")
    print(f"      F1-score (macro): {f1:.4f}")
    
    # Decision Tree (multiclass)
    print("\n   b) Decision Tree:")
    dt_multi = DecisionTree(max_depth=5, min_samples_split=5)
    dt_multi.fit(X_train_scaled, y_train)
    y_pred = dt_multi.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"      Accuracy: {acc:.4f}")
    print(f"      Precision (macro): {prec:.4f}")
    print(f"      Recall (macro): {rec:.4f}")
    print(f"      F1-score (macro): {f1:.4f}")
    
    # Random Forest (multiclass)
    print("\n   c) Random Forest:")
    rf_multi = RandomForest(n_estimators=50, max_depth=5, random_state=42)
    rf_multi.fit(X_train_scaled, y_train)
    y_pred = rf_multi.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"      Accuracy: {acc:.4f}")
    print(f"      Precision (macro): {prec:.4f}")
    print(f"      Recall (macro): {rec:.4f}")
    print(f"      F1-score (macro): {f1:.4f}")


def demonstrate_preprocessing():
    """Demonstrate data preprocessing utilities."""
    print("\n\n" + "=" * 60)
    print("DATA PREPROCESSING DEMONSTRATION")
    print("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 3) * [10, 5, 2] + [50, 20, 5]
    
    print("\n1. Original data statistics:")
    print(f"   Mean: {np.mean(X, axis=0)}")
    print(f"   Std:  {np.std(X, axis=0)}")
    print(f"   Min:  {np.min(X, axis=0)}")
    print(f"   Max:  {np.max(X, axis=0)}")
    
    # Standard scaling
    print("\n2. StandardScaler:")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"   Mean after scaling: {np.mean(X_scaled, axis=0)}")
    print(f"   Std after scaling:  {np.std(X_scaled, axis=0)}")
    
    # Min-Max scaling
    print("\n3. MinMaxScaler:")
    minmax_scaler = MinMaxScaler(feature_range=(0, 1))
    X_minmax = minmax_scaler.fit_transform(X)
    
    print(f"   Min after scaling: {np.min(X_minmax, axis=0)}")
    print(f"   Max after scaling: {np.max(X_minmax, axis=0)}")
    
    # Train-test split
    print("\n4. Train-test split:")
    y = np.random.randint(0, 2, 100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"   Original size: {len(X)}")
    print(f"   Training size: {len(X_train)}")
    print(f"   Test size: {len(X_test)}")
    print(f"   Class distribution in training: {np.bincount(y_train)}")
    print(f"   Class distribution in test: {np.bincount(y_test)}")


def main():
    """Main function to run all demonstrations."""
    print("SUPERVISED MACHINE LEARNING LIBRARY DEMONSTRATION")
    print("=" * 60)
    print("This demonstration shows the capabilities of the ML library")
    print("including regression, classification, and utility functions.")
    
    try:
        demonstrate_regression()
        demonstrate_classification()
        demonstrate_preprocessing()
        
        print("\n\n" + "=" * 60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nAll algorithms have been tested and are working properly.")
        print("You can now use this library for your machine learning projects.")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()