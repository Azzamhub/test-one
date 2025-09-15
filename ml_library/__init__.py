"""
Supervised Machine Learning Library
===================================

A comprehensive library for supervised machine learning including regression 
and classification algorithms.

Modules:
--------
- regression: Linear, Polynomial, and Ridge regression implementations
- classification: Logistic regression, Decision Tree, and Random Forest
- utils: Data preprocessing, evaluation metrics, and utilities
- datasets: Sample datasets for testing and demonstration
"""

__version__ = "1.0.0"
__author__ = "ML Library Team"

# Import main modules for easy access
from . import regression
from . import classification
from . import utils
from . import datasets

__all__ = [
    'regression',
    'classification', 
    'utils',
    'datasets'
]