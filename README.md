# test-one
# Supervised Machine Learning: Regression and Classification

A simple yet comprehensive implementation of fundamental supervised machine learning algorithms in Python. This library provides implementations of Linear Regression and Logistic Regression with supporting utilities for data preprocessing, model evaluation, and visualization.

## Features

### Algorithms
- **Linear Regression**: Ordinary least squares implementation for regression tasks
- **Logistic Regression**: Gradient descent-based binary classification

### Evaluation Metrics
- Mean Squared Error (MSE)
- R² Score (Coefficient of Determination)
- Mean Absolute Error (MAE)
- Classification Accuracy
- Confusion Matrix

### Utilities
- Train/test data splitting
- Feature normalization (standardization and min-max scaling)
- Synthetic dataset generation
- Data visualization tools

## Installation

1. Clone this repository:
```bash
git clone https://github.com/Azzamhub/test-one.git
cd test-one
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Linear Regression Example
```python
from supervised_ml import LinearRegression, train_test_split
from supervised_ml.utils import generate_regression_data
from supervised_ml.metrics import mean_squared_error, r2_score

# Generate synthetic data
X, y = generate_regression_data(n_samples=100, n_features=1, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
```

### Logistic Regression Example
```python
from supervised_ml import LogisticRegression, normalize_features
from supervised_ml.utils import generate_classification_data
from supervised_ml.metrics import accuracy_score

# Generate synthetic data
X, y = generate_classification_data(n_samples=200, n_features=2, random_state=42)

# Normalize features
X_normalized, _ = normalize_features(X, method='standardize')

# Train model
model = LogisticRegression(learning_rate=0.1, max_iter=1000)
model.fit(X_normalized, y)

# Make predictions
y_pred = model.predict(X_normalized)

# Evaluate
print(f"Accuracy: {accuracy_score(y, y_pred):.4f}")
```

## Examples

Run the complete examples:

```bash
# Linear regression with visualization
python examples/regression_example.py

# Logistic regression with decision boundary
python examples/classification_example.py
```

## Testing

Run the test suite:

```bash
python tests/test_supervised_ml.py
```

## Project Structure

```
test-one/
├── supervised_ml/
│   ├── __init__.py          # Main package exports
│   ├── regression.py        # Linear regression implementation
│   ├── classification.py    # Logistic regression implementation
│   ├── metrics.py          # Evaluation metrics
│   └── utils.py            # Utility functions
├── examples/
│   ├── regression_example.py
│   └── classification_example.py
├── tests/
│   └── test_supervised_ml.py
├── requirements.txt
└── README.md
```

## API Reference

### Models

#### LinearRegression
- `fit(X, y)`: Train the model
- `predict(X)`: Make predictions
- `score(X, y)`: Calculate R² score

#### LogisticRegression
- `fit(X, y)`: Train the model  
- `predict(X)`: Make binary predictions
- `predict_proba(X)`: Get prediction probabilities
- `score(X, y)`: Calculate accuracy

### Metrics
- `mean_squared_error(y_true, y_pred)`
- `r2_score(y_true, y_pred)`
- `accuracy_score(y_true, y_pred)`
- `confusion_matrix(y_true, y_pred)`

### Utilities
- `train_test_split(X, y, test_size=0.2)`
- `normalize_features(X, method='standardize')`
- `generate_regression_data(n_samples, n_features)`
- `generate_classification_data(n_samples, n_features)`

## Requirements

- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.5.0
- pandas >= 1.3.0

## License

This project is open source and available under the MIT License.
