"""scikit-learn style LinearRegression on the GPU: coefficients from one pass over X for X^T X.

    lr = LinearRegression().fit(X, y)          # X: numpy, pandas or MLX; float64 is rounded to float32 on the GPU
    lr.coef_, lr.intercept_, lr.predict(X_new), lr.score(X, y)

refine: 'auto' adds the residual pass (one more read of X) only when the Gram solve could be off by more than
~1e-7 relative, i.e. when the design is collinear; True always refines; False never.
"""
import numpy as np
from .regression import OLS, WLS


class LinearRegression:
    def __init__(self, fit_intercept=True, refine="auto"):
        self.fit_intercept = fit_intercept
        self.refine = refine

    def fit(self, X, y, sample_weight=None):
        m = (WLS(y, X, weights=sample_weight, add_constant=self.fit_intercept) if sample_weight is not None
             else OLS(y, X, add_constant=self.fit_intercept))
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.refine == "auto":
                r = m.fit(refine=False)
                if r._lam_min < 1e-3:
                    r = m.fit(refine=True)
            else:
                r = m.fit(refine=bool(self.refine))
        b = np.asarray(r.params_arr, np.float64)
        self.intercept_ = float(b[0]) if self.fit_intercept else 0.0
        self.coef_ = b[1:] if self.fit_intercept else b
        self.rank_ = r.rank
        self.n_features_in_ = m.data.p
        self.results_ = r
        return self

    def predict(self, X):
        return self.results_.predict(X)

    def score(self, X, y, sample_weight=None):
        """R^2 of the predictions on (X, y), as sklearn defines it."""
        yh = self.predict(X)
        y = np.asarray(y.to_numpy() if hasattr(y, "to_numpy") else y, np.float64).reshape(-1)
        w = np.ones_like(y) if sample_weight is None else np.asarray(sample_weight, np.float64)
        ybar = np.average(y, weights=w)
        return 1.0 - np.sum(w * (y - yh) ** 2) / np.sum(w * (y - ybar) ** 2)
