# Supervised Machine Learning Library

A comprehensive Python library for supervised machine learning that implements both regression and classification algorithms from scratch using NumPy. This library provides a clean, scikit-learn-like API for easy use in machine learning projects.

## Features

### Regression Algorithms
- **Linear Regression**: Implementation using gradient descent and normal equation
- **Polynomial Regression**: Extends linear regression with polynomial features  
- **Ridge Regression**: Linear regression with L2 regularization

### Classification Algorithms
- **Logistic Regression**: Binary and multiclass classification using one-vs-rest
- **Decision Tree**: Recursive binary splitting with Gini impurity
- **Random Forest**: Ensemble of decision trees with bootstrap sampling

### Utility Functions
- **Data Preprocessing**: StandardScaler and MinMaxScaler for feature scaling
- **Data Splitting**: train_test_split for data partitioning
- **Evaluation Metrics**: accuracy, precision, recall, F1-score, MSE, R²
- **Sample Datasets**: Built-in dataset generators for testing

## Installation

### Requirements
```bash
pip install -r requirements.txt
```

Required packages:
- numpy>=1.21.0
- matplotlib>=3.5.0 
- scikit-learn>=1.0.0 (for comparison/validation only)
- pandas>=1.3.0

## Quick Start

### Regression Example
```python
from ml_library.regression import LinearRegression
from ml_library.utils import train_test_split, StandardScaler
from ml_library.datasets import make_regression_data

# Generate sample data
X, y = make_regression_data(n_samples=100, n_features=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression(learning_rate=0.01, max_iterations=1000)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)
print(f"R² Score: {model.score(X_test_scaled, y_test):.4f}")
```

### Classification Example
```python
from ml_library.classification import RandomForest
from ml_library.utils import train_test_split, accuracy_score
from ml_library.datasets import load_iris_like

# Load dataset
X, y, feature_names, target_names = load_iris_like()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForest(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
```

## API Reference

### Regression Models

#### LinearRegression
```python
LinearRegression(learning_rate=0.01, max_iterations=1000, method='gradient_descent', tolerance=1e-6)
```
- `learning_rate`: Step size for gradient descent
- `max_iterations`: Maximum training iterations
- `method`: 'gradient_descent' or 'normal_equation'
- `tolerance`: Convergence threshold

#### PolynomialRegression
```python
PolynomialRegression(degree=2, learning_rate=0.01, max_iterations=1000)
```
- `degree`: Polynomial degree for feature expansion

#### RidgeRegression
```python
RidgeRegression(alpha=1.0, learning_rate=0.01, max_iterations=1000)
```
- `alpha`: L2 regularization strength

### Classification Models

#### LogisticRegression
```python
LogisticRegression(learning_rate=0.01, max_iterations=1000, multi_class='ovr')
```
- `multi_class`: 'ovr' for one-vs-rest multiclass strategy

#### DecisionTree
```python
DecisionTree(max_depth=10, min_samples_split=2, min_samples_leaf=1)
```
- `max_depth`: Maximum tree depth
- `min_samples_split`: Minimum samples to split node
- `min_samples_leaf`: Minimum samples in leaf node

#### RandomForest
```python
RandomForest(n_estimators=100, max_depth=10, max_features='sqrt', bootstrap=True, random_state=None)
```
- `n_estimators`: Number of trees
- `max_features`: Features per tree ('sqrt', 'log2', int, or None)
- `bootstrap`: Whether to use bootstrap sampling

### Utility Functions

#### Data Preprocessing
```python
# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

scaler = MinMaxScaler(feature_range=(0, 1))
X_normalized = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

#### Evaluation Metrics
```python
from ml_library.utils import accuracy_score, precision_score, recall_score, f1_score
from ml_library.utils import mean_squared_error, r2_score

# Classification metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

# Regression metrics
mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
```

## Running Examples

### Comprehensive Demo
```bash
python examples/comprehensive_demo.py
```

This runs a complete demonstration of all algorithms with sample datasets.

### Running Tests
```bash
python tests/test_basic.py
```

## Dataset Generators

The library includes several built-in dataset generators:

```python
from ml_library.datasets import make_regression_data, make_classification_data
from ml_library.datasets import load_iris_like, load_boston_like

# Generate synthetic regression data
X, y = make_regression_data(n_samples=100, n_features=3, noise=0.1)

# Generate synthetic classification data  
X, y = make_classification_data(n_samples=100, n_features=2, n_classes=3)

# Load Iris-like dataset (multiclass)
X, y, feature_names, target_names = load_iris_like()

# Load Boston Housing-like dataset (regression)
X, y, feature_names = load_boston_like()
```

## Project Structure

```
ml_library/
├── __init__.py                 # Main package
├── regression/                 # Regression algorithms
│   ├── __init__.py
│   ├── linear_regression.py
│   ├── polynomial_regression.py
│   └── ridge_regression.py
├── classification/             # Classification algorithms  
│   ├── __init__.py
│   ├── logistic_regression.py
│   ├── decision_tree.py
│   └── random_forest.py
├── utils/                      # Utilities
│   ├── __init__.py
│   ├── preprocessing.py
│   └── metrics.py
└── datasets/                   # Sample datasets
    ├── __init__.py
    └── sample_data.py

examples/                       # Example scripts
├── comprehensive_demo.py

tests/                          # Test suite
├── test_basic.py

requirements.txt               # Dependencies
README.md                     # This file
```

## Performance Notes

- All algorithms are implemented in pure NumPy for educational purposes and transparency
- For production use, consider scikit-learn which is highly optimized
- The library prioritizes code clarity and understanding over maximum performance
- Suitable for learning, prototyping, and understanding ML algorithm internals

## Contributing

This library is designed as a learning tool and reference implementation. Feel free to:
- Study the source code to understand how algorithms work
- Extend with additional algorithms 
- Use as a foundation for your own ML implementations
- Compare with scikit-learn implementations

## License

This project is open source and available under the MIT License.
