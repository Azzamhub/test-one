"""
Linear Regression Example

This example demonstrates how to use the LinearRegression class
for supervised learning regression tasks.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to the path so we can import supervised_ml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from supervised_ml import LinearRegression, train_test_split, normalize_features
from supervised_ml.utils import generate_regression_data
from supervised_ml.metrics import mean_squared_error, r2_score


def main():
    print("=== Linear Regression Example ===\n")
    
    # Generate synthetic regression data
    print("1. Generating synthetic regression dataset...")
    X, y = generate_regression_data(n_samples=100, n_features=1, noise=0.2, random_state=42)
    print(f"   Dataset shape: X={X.shape}, y={y.shape}")
    
    # Split data into training and testing sets
    print("\n2. Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"   Training set: X={X_train.shape}, y={y_train.shape}")
    print(f"   Test set: X={X_test.shape}, y={y_test.shape}")
    
    # Create and fit the model
    print("\n3. Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print(f"   Fitted coefficients: {model.coefficients_}")
    print(f"   Fitted intercept: {model.intercept_:.4f}")
    
    # Make predictions
    print("\n4. Making predictions...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Evaluate the model
    print("\n5. Evaluating model performance...")
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"   Training MSE: {train_mse:.4f}")
    print(f"   Test MSE: {test_mse:.4f}")
    print(f"   Training R²: {train_r2:.4f}")
    print(f"   Test R²: {test_r2:.4f}")
    
    # Visualize results (if single feature)
    if X.shape[1] == 1:
        print("\n6. Creating visualization...")
        plt.figure(figsize=(10, 6))
        
        # Plot training data
        plt.scatter(X_train.flatten(), y_train, alpha=0.6, color='blue', label='Training Data')
        plt.scatter(X_test.flatten(), y_test, alpha=0.6, color='red', label='Test Data')
        
        # Plot regression line
        X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_line = model.predict(X_line)
        plt.plot(X_line.flatten(), y_line, color='green', linewidth=2, label='Regression Line')
        
        plt.xlabel('Feature X')
        plt.ylabel('Target y')
        plt.title('Linear Regression Results')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('linear_regression_example.png', dpi=300, bbox_inches='tight')
        print("   Visualization saved as 'linear_regression_example.png'")
    
    print("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    main()