"""scikit-learn style LinearRegression on the GPU: coefficients from one pass over X for X^T X.

    lr = LinearRegression().fit(X, y)          # X: numpy, pandas or MLX; float64 is rounded to float32 on the GPU
    lr.coef_, lr.intercept_, lr.predict(X_new), lr.score(X, y)

refine: 'auto' adds the residual pass (one more read of X) only when the design is collinear enough for the Gram
solve to be off by more than ~1e-7 relative; True always refines; False never. For a rank-deficient design the
coefficients are, as in scikit-learn, the minimum-norm solution of the centred problem (the intercept is not part of
the norm), which differs from statsmodels' minimum-norm solution over all parameters.
"""
import warnings
import numpy as np
from .regression import OLS, WLS


class LinearRegression:
    def __init__(self, fit_intercept=True, refine="auto"):
        self.fit_intercept = fit_intercept
        self.refine = refine

    def fit(self, X, y, sample_weight=None):
        y_arr = y.to_numpy() if hasattr(y, "to_numpy") else y
        y2d = getattr(y_arr, "ndim", 1) == 2
        if y2d and y_arr.shape[1] != 1:
            raise ValueError("LinearRegression here fits one target; y with several columns is not supported")
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=np.float64) if not hasattr(sample_weight, "astype") or isinstance(sample_weight, np.ndarray) else sample_weight
            try:
                swn = np.asarray(sw, dtype=np.float64)
                if np.any(swn < 0):
                    raise ValueError("Negative values in data passed to `sample_weight`")
                if not np.any(swn > 0):
                    raise ValueError("Sample weights must contain at least one non-zero number.")
            except TypeError:
                pass
        self.feature_names_in_ = np.asarray(list(X.columns), dtype=object) if hasattr(X, "columns") else None
        ac = "add" if self.fit_intercept else False            # scikit-learn always adds the intercept
        m = (WLS(y, X, weights=sample_weight, add_constant=ac) if sample_weight is not None
             else OLS(y, X, add_constant=ac))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.refine == "auto":
                r = m.fit(refine=False)
                if r._lam_min < 1e-3:
                    r = m.fit(refine=True)
            else:
                r = m.fit(refine=bool(self.refine))
        b = np.asarray(r.params_arr, np.float64).copy()
        vc = r._vc
        if r._null is not None and vc:
            # scikit-learn: minimum norm over the coefficients only (centred problem); move along null(X)
            N = r._null
            t = -np.linalg.lstsq(N[vc:], b[vc:], rcond=None)[0]
            b = b + N @ t
        self.intercept_ = float(b[0]) if vc else 0.0
        self.coef_ = b[vc:]
        if y2d:
            self.coef_ = self.coef_[None, :]
            self.intercept_ = np.array([self.intercept_])
        self._y2d = y2d
        self.rank_ = r.rank
        self.n_features_in_ = m.data.p
        self.results_ = r
        return self

    def _check_names(self, X):
        if self.feature_names_in_ is not None and hasattr(X, "columns"):
            if list(X.columns) != list(self.feature_names_in_):
                raise ValueError("The feature names should match those that were passed during fit.")

    def predict(self, X):
        self._check_names(X)
        import mlx.core as mx
        if isinstance(X, mx.array):
            Xn = np.array(X.astype(mx.float32)).astype(np.float64)
        else:
            Xn = np.asarray(X.to_numpy() if hasattr(X, "to_numpy") else X, dtype=np.float64)
        coef = np.ravel(self.coef_)
        if Xn.ndim == 1:
            Xn = Xn[:, None] if coef.size == 1 else Xn[None, :]
        out = Xn @ coef + float(np.ravel([self.intercept_])[0])
        return out[:, None] if self._y2d else out

    def score(self, X, y, sample_weight=None):
        """R^2 of the predictions on (X, y), as scikit-learn defines it (1.0 for a constant y predicted exactly)."""
        yh = np.ravel(self.predict(X))
        y = np.asarray(y.to_numpy() if hasattr(y, "to_numpy") else y, np.float64).reshape(-1)
        w = np.ones_like(y) if sample_weight is None else np.asarray(sample_weight, np.float64)
        num = np.sum(w * (y - yh) ** 2)
        den = np.sum(w * (y - np.average(y, weights=w)) ** 2)
        if den == 0:                                       # constant y: 1.0 if predicted exactly (to roundoff)
            return 1.0 if num <= 1e-24 * max(np.sum(w * y * y), 1e-300) else 0.0
        return 1.0 - num / den
