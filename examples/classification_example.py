"""
Logistic Regression Example

This example demonstrates how to use the LogisticRegression class
for supervised learning binary classification tasks.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to the path so we can import supervised_ml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from supervised_ml import LogisticRegression, train_test_split, normalize_features
from supervised_ml.utils import generate_classification_data
from supervised_ml.metrics import accuracy_score, confusion_matrix


def main():
    print("=== Logistic Regression Example ===\n")
    
    # Generate synthetic classification data
    print("1. Generating synthetic classification dataset...")
    X, y = generate_classification_data(n_samples=200, n_features=2, random_state=42)
    print(f"   Dataset shape: X={X.shape}, y={y.shape}")
    print(f"   Class distribution: Class 0: {np.sum(y == 0)}, Class 1: {np.sum(y == 1)}")
    
    # Normalize features
    print("\n2. Normalizing features...")
    X_normalized, scaler_params = normalize_features(X, method='standardize')
    
    # Split data into training and testing sets
    print("\n3. Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y, test_size=0.2, random_state=42
    )
    print(f"   Training set: X={X_train.shape}, y={y_train.shape}")
    print(f"   Test set: X={X_test.shape}, y={y_test.shape}")
    
    # Create and fit the model
    print("\n4. Training Logistic Regression model...")
    model = LogisticRegression(learning_rate=0.1, max_iter=1000)
    model.fit(X_train, y_train)
    print(f"   Fitted coefficients: {model.coefficients_}")
    print(f"   Fitted intercept: {model.intercept_:.4f}")
    
    # Make predictions
    print("\n5. Making predictions...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_prob_test = model.predict_proba(X_test)
    
    # Evaluate the model
    print("\n6. Evaluating model performance...")
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    print(f"   Training Accuracy: {train_accuracy:.4f}")
    print(f"   Test Accuracy: {test_accuracy:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)
    print(f"   Confusion Matrix:")
    print(f"   [[TN={cm[0,0]:2d}, FP={cm[0,1]:2d}]")
    print(f"    [FN={cm[1,0]:2d}, TP={cm[1,1]:2d}]]")
    
    # Show some predictions with probabilities
    print(f"\n7. Sample predictions with probabilities:")
    for i in range(min(5, len(y_test))):
        prob_0, prob_1 = y_prob_test[i]
        print(f"   Sample {i+1}: True={y_test[i]}, Predicted={y_pred_test[i]}, "
              f"P(class=0)={prob_0:.3f}, P(class=1)={prob_1:.3f}")
    
    # Visualize results (if 2D features)
    if X.shape[1] == 2:
        print("\n8. Creating visualization...")
        plt.figure(figsize=(12, 5))
        
        # Plot 1: Original data
        plt.subplot(1, 2, 1)
        colors = ['red', 'blue']
        for class_val in [0, 1]:
            mask = y == class_val
            plt.scatter(X[mask, 0], X[mask, 1], c=colors[class_val], 
                       alpha=0.6, label=f'Class {class_val}')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Original Classification Data')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Decision boundary
        plt.subplot(1, 2, 2)
        
        # Plot training data
        for class_val in [0, 1]:
            mask_train = y_train == class_val
            mask_test = y_test == class_val
            plt.scatter(X_train[mask_train, 0], X_train[mask_train, 1], 
                       c=colors[class_val], alpha=0.6, marker='o', 
                       label=f'Train Class {class_val}')
            plt.scatter(X_test[mask_test, 0], X_test[mask_test, 1], 
                       c=colors[class_val], alpha=0.8, marker='s', 
                       label=f'Test Class {class_val}')
        
        # Create decision boundary
        h = 0.02
        x_min, x_max = X_normalized[:, 0].min() - 1, X_normalized[:, 0].max() + 1
        y_min, y_max = X_normalized[:, 1].min() - 1, X_normalized[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict_proba(mesh_points)[:, 1]
        Z = Z.reshape(xx.shape)
        
        plt.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--', linewidths=2)
        plt.contourf(xx, yy, Z, levels=50, alpha=0.3, cmap='RdYlBu')
        
        plt.xlabel('Feature 1 (normalized)')
        plt.ylabel('Feature 2 (normalized)')
        plt.title('Decision Boundary (Normalized Data)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('logistic_regression_example.png', dpi=300, bbox_inches='tight')
        print("   Visualization saved as 'logistic_regression_example.png'")
    
    print("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    main()