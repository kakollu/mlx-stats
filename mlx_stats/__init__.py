"""mlx-stats: statistical models on the Apple GPU through MLX.

OLS / WLS with statsmodels' results (standard errors, t and p values, confidence intervals, R^2, F, log-likelihood,
AIC/BIC, omnibus, Jarque-Bera, Durbin-Watson, condition number, HC0/HC1 robust covariance) computed from one pass of
the data for X^T X and one for the residuals.
"""
import numpy as np
from .regression import OLS, WLS, RegressionResults, SingularDesignWarning
from .linear_model import LinearRegression

__all__ = ["OLS", "WLS", "LinearRegression", "RegressionResults", "SingularDesignWarning", "add_constant"]


def add_constant(X, prepend=True):
    """A column of ones in front (or at the end), like statsmodels.add_constant. Keeps DataFrames as DataFrames."""
    if hasattr(X, "columns"):
        X = X.copy()
        if prepend:
            X.insert(0, "const", 1.0)
        else:
            X["const"] = 1.0
        return X
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, None]
    ones = np.ones((X.shape[0], 1), dtype=X.dtype if X.dtype in (np.float32, np.float64) else np.float64)
    return np.hstack([ones, X] if prepend else [X, ones])
