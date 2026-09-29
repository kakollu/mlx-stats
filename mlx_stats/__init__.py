"""mlx-stats: statistical models on the Apple GPU through MLX.

OLS / WLS with statsmodels' results (standard errors, t and p values, confidence intervals, R^2, F, log-likelihood,
AIC/BIC, omnibus, Jarque-Bera, Durbin-Watson, condition number, HC0/HC1 robust covariance) computed from one pass of
the data for X^T X and one for the residuals.
"""
import numpy as np
from .regression import OLS, WLS, RegressionResults, SingularDesignWarning
from .linear_model import LinearRegression

__all__ = ["OLS", "WLS", "LinearRegression", "RegressionResults", "SingularDesignWarning", "add_constant"]


def add_constant(X, prepend=True, has_constant="skip"):
    """A column of ones in front (or at the end), like statsmodels.add_constant. With has_constant='skip' (the
    default, as in statsmodels) nothing is added when X already has a nonzero constant column; 'add' always adds;
    'raise' raises. DataFrames and Series come back as DataFrames with a 'const' column."""
    if has_constant not in ("skip", "add", "raise"):
        raise ValueError("has_constant must be 'skip', 'add' or 'raise'")
    if hasattr(X, "to_frame") and not hasattr(X, "columns"):
        X = X.to_frame()
    if hasattr(X, "columns"):
        vals = X.to_numpy(dtype=np.float64, na_value=np.nan) if len(X) else np.zeros((0, X.shape[1]))
    else:
        vals = np.asarray(X, dtype=np.float64)
        if vals.ndim == 1:
            vals = vals[:, None]
    if vals.shape[0]:
        const = (np.nanmax(vals, 0) == np.nanmin(vals, 0)) & (vals[0] != 0)
        if const.any():
            if has_constant == "raise":
                raise ValueError("data already contains a constant column")
            if has_constant == "skip":
                return X if hasattr(X, "columns") else (vals if np.asarray(X).ndim == 1 else X)
    if hasattr(X, "columns"):
        X = X.copy()
        name = "const"
        if prepend:
            X.insert(0, name, 1.0, allow_duplicates=True)
        else:
            X[name] = 1.0
        return X
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, None]
    ones = np.ones((X.shape[0], 1), dtype=X.dtype if X.dtype in (np.float32, np.float64) else np.float64)
    return np.hstack([ones, X] if prepend else [X, ones])
