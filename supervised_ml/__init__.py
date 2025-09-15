"""
Supervised Machine Learning: Regression and Classification

A simple implementation of fundamental supervised learning algorithms.
"""

from .regression import LinearRegression
from .classification import LogisticRegression
from .metrics import mean_squared_error, accuracy_score, r2_score
from .utils import train_test_split, normalize_features

__version__ = "1.0.0"
__all__ = [
    "LinearRegression",
    "LogisticRegression", 
    "mean_squared_error",
    "accuracy_score",
    "r2_score",
    "train_test_split",
    "normalize_features"
]