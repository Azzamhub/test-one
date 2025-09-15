"""
Regression algorithms implementation.
"""

from .linear_regression import LinearRegression
from .polynomial_regression import PolynomialRegression
from .ridge_regression import RidgeRegression

__all__ = [
    'LinearRegression',
    'PolynomialRegression', 
    'RidgeRegression'
]