"""Linear regression with statsmodels' results, from one GPU pass for X^T X and one for the residuals.

    res = OLS(y, X).fit()            # X as given: add a constant column yourself (add_constant), like statsmodels
    res.params, res.bse, res.pvalues, res.conf_int(), res.rsquared, res.fvalue, res.llf, res.aic, ...
    print(res.summary())
    WLS(y, X, weights=w).fit(cov_type="HC1")

How the numbers are made. Pass 1 (`_gram`) returns G = sum s^2 [a, a_y, 1][a, a_y, 1]^T with a = x - K, a_y = y - K_y,
K near the column means and s = sqrt(weight). Every quantity below is float64 arithmetic on G. The basis for the solve
is B = [a_j for non-constant columns j, 1]: shifted columns are close to centred, so B^T W B is well conditioned even
when the raw columns have large means. X = B W_map exactly (W_map holds 1s and the shifts), and with the eigen-
decomposition B^T W B = V L V^T, X = Q F where Q has orthonormal columns and F = L^(1/2) V^T W_map is small. From F:
statsmodels' minimum-norm parameters pinv(F) Q^T y, normalized covariance pinv(F^T F) = pinv(X^T X), rank, and the
singular values behind the condition number - all the same definitions statsmodels uses, evaluated in a basis where
float32 accumulation is enough.

Pass 2 (`_resid`) computes the residuals with two-float coefficients and returns Q^T e exactly enough for one step of
iterative refinement (so the parameters converge to what a float64 solver gives on the same float32 data), the exact
sum of squared residuals, the residual power sums behind skew, kurtosis, omnibus and Jarque-Bera, and Durbin-Watson.
Robust (HC0/HC1) covariance adds one more Gram pass with each row scaled by its residual.

Precision envelope: X^T X is accurate to ~3e-8 of sqrt(G_ii G_jj) per entry (float32 rounding, independent of n).
Directions whose scaled eigenvalue is below RANK_TOL are treated as exactly collinear; designs that are merely ill
conditioned converge through refinement when cond(scaled X)^2 * 3e-8 is well below 1, i.e. cond(scaled X) up to about
a few thousand. Past that a warning is raised: float32 data cannot resolve the design.
"""
import time
import warnings
import numpy as np
from scipy import stats
from ._data import DeviceData
from ._passes import gram_pass, resid_pass
from . import _tests

EXACT_TOL = 1e-4         # scaled eigenvalue of B^T W B below which the fit recomputes the Gram exactly
RANK_TOL = 1e-6          # ...below which a direction counts as exact collinearity, from the fast (float32) Gram
RANK_TOL_EXACT = 1e-12   # ...the same, from the exact Gram
F_TOL = 1e-10            # relative singular value of the column-scaled F below which a direction counts as zero
DATA_TOL = 1e-10         # scaled eigenvalue below which float32 data itself cannot resolve the design


class SingularDesignWarning(UserWarning):
    pass


class OLS:
    """Ordinary least squares, statsmodels argument order: OLS(endog, exog)."""

    def __init__(self, endog, exog, hasconst=None, resident=None, add_constant=False):
        """add_constant=True fits an intercept without copying X to append a column (statsmodels needs
        sm.add_constant(X), a full copy); the constant is reported first, named 'const'."""
        self.data = DeviceData(exog, endog, None, resident)
        self.hasconst = hasconst
        self.add_constant = add_constant
        self._weighted = False
        self._init_names()

    def _init_names(self):
        d = self.data
        self.yname = d.yname or "y"
        self._pandas = d.xnames is not None
        self.xnames = (["const"] + list(d.xnames)) if (d.xnames is not None and self.add_constant) else d.xnames
        self.nobs = float(d.n)
        self.k_design = d.p + (1 if self.add_constant else 0)

    def fit(self, cov_type="nonrobust", use_t=None, refine=True, precise=None, max_passes=4):
        """cov_type: 'nonrobust', 'HC0' or 'HC1'. refine: run the residual pass (exact SSR, diagnostics, refinement).
        With refine=False only the Gram pass runs: parameters from X^T X, SSR from the Gram, no residual diagnostics.
        precise: None recomputes X^T X exactly when the design is ill conditioned; True always; False never."""
        return RegressionResults(self, cov_type, use_t, refine, max_passes, precise)


class WLS(OLS):
    """Weighted least squares: WLS(endog, exog, weights). Weights are inverse variances, as in statsmodels."""

    def __init__(self, endog, exog, weights=1.0, hasconst=None, resident=None, add_constant=False):
        self.data = DeviceData(exog, endog, weights, resident)
        self.hasconst = hasconst
        self.add_constant = add_constant
        self._weighted = self.data.w is not None
        self._init_names()


def _pinv_sym_psd(A, tol):
    """eigen-pinv of a symmetric PSD matrix after scaling to unit diagonal. -> (V_r, lam_r (scaled), d, rank, lam_all)"""
    d = np.sqrt(np.maximum(np.diag(A), 0.0))
    d = np.where(d > 0, d, 1.0)
    S = A / np.outer(d, d)
    lam, V = np.linalg.eigh((S + S.T) / 2)
    keep = lam > tol * max(lam.max(), 0.0)
    return V[:, keep], lam[keep], d, int(keep.sum()), lam


def _pinv(F, tol):
    """pinv(F) (p x r), rank, and the p singular values of F (zeros for the directions F lacks).

    The rank is decided with the columns scaled to unit norm, so a column's units do not decide whether it is
    collinear; the pseudo-inverse itself is taken in the unscaled metric, because the minimum-norm solution that
    statsmodels (and numpy's pinv) returns for a rank-deficient design is minimum-norm in the original units."""
    p = F.shape[1]
    cn = np.sqrt((F * F).sum(0))
    cn = np.where(cn > 0, cn, 1.0)
    ss = np.linalg.svd(F / cn, compute_uv=False)
    rank = int((ss > tol * (ss.max() if ss.size else 0.0)).sum())
    U, s, Vt = np.linalg.svd(F, full_matrices=False)
    P = (Vt[:rank].T / s[:rank]) @ U[:, :rank].T
    sv = np.zeros(p)
    sv[:min(p, s.size)] = s[:p]
    if rank < p:
        sv[rank:] = 0.0                                # exact collinearity: the missing directions are zero
    return P, rank, sv


class RegressionResults:
    """statsmodels RegressionResults attribute names; arrays are pandas objects when the input had column names."""

    def __init__(self, model, cov_type, use_t, refine, max_passes, precise=None):
        t0 = time.perf_counter()
        self.model = model
        d = model.data
        n, p = d.n, d.p
        self.nobs = float(n)
        K = d.shift()
        self._K = K
        Kx, Ky = K[:p].astype(np.float64), float(K[p])
        G = gram_pass(d, K, exact=bool(precise))
        self._passes = 1
        if not np.all(np.isfinite(G)):
            raise ValueError("exog, endog or weights contain inf or NaN")
        self.exact_gram = bool(precise)
        info = self._setup(G, Kx, Ky, RANK_TOL_EXACT if precise else RANK_TOL)
        if precise is None and info["lam_min"] < EXACT_TOL:
            G = gram_pass(d, K, exact=True)                  # collinear: the float32 Gram cannot decide it
            self._passes += 1
            self.exact_gram = True
            info = self._setup(G, Kx, Ky, RANK_TOL_EXACT)
        beta, ssr, J, F, Fp = info["beta"], info["ssr"], info["J"], info["F"], info["Fp"]
        self._lam_min = info["lam_min"]
        rank, k_constant = self.rank, self.k_constant
        if rank < self.model.k_design:
            warnings.warn("The design matrix is rank-deficient. The model parameters are not uniquely determined.",
                          SingularDesignWarning, stacklevel=3)
        elif info["lam_min"] < DATA_TOL:
            warnings.warn(f"Near-collinear design (smallest scaled eigenvalue {info['lam_min']:.1e}): float32 data "
                          f"cannot resolve it; parameters may differ from a float64 fit on float64 data.",
                          SingularDesignWarning, stacklevel=3)
        self._diag = None
        self._refine_steps = []

        # --- residual pass: exact SSR, refinement, diagnostics ----------------------------------------------------
        if refine:
            for it in range(max_passes):
                c0 = float(self._Kd @ beta - Ky)
                beta_s = beta[self._vc:].copy()
                grad, mom, dwn, _ = resid_pass(d, K, beta_s, c0)
                self._passes += 1
                ssr_pass = mom[1]                                    # sum rho^2, exact to float32 rounding of rho
                su2, swy, sw = mom[5], mom[6], mom[7]               # exact sums of (s a_y)^2, s^2 a_y, s^2
                self.centered_tss = su2 - swy * swy / sw
                self.uncentered_tss = su2 + 2 * Ky * swy + Ky * Ky * sw
                qe = self._Rinv_t.T @ grad[J]                        # Q^T W e
                delta = Fp @ qe
                gain = float(np.sum((F @ delta) ** 2))              # SSR decrease of the refinement step
                beta = beta + delta
                ssr = max(ssr_pass - gain, 0.0)
                self._diag = (mom, dwn)
                # converged when the step moves no parameter by more than 1e-10 of itself or 1e-8 of its standard
                # error: along a collinear direction the SSR is flat, so the SSR decrease alone would stop too early
                se = np.sqrt(np.maximum(np.diag(self.normalized_cov_params), 0.0) * ssr / max(n - self.rank, 1))
                tol = 1e-10 * np.abs(beta) + 1e-8 * se + 1e-300
                # the next step would be about `contraction` times this one: the Gram used for the solve differs from
                # the exact X^T X by ~1e-7 of its scale, which the smallest scaled eigenvalue amplifies
                contraction = min(1.0, 1e-6 / max(info["lam_min"], 1e-300))
                self._refine_steps.append((float(np.max(np.abs(delta) / np.maximum(np.abs(beta), 1e-300))), gain / max(ssr_pass, 1e-300)))
                converged = np.all(np.abs(delta) <= tol) or np.all(contraction * np.abs(delta) <= tol)
                # the residual moments come from this pass, before the step: keep them only if the step changed the
                # residuals by less than a millionth of their size (||X delta||^2 = gain)
                if converged and gain <= 1e-12 * max(ssr_pass, 1e-300):
                    break
        self.params_arr = beta
        self.ssr = float(ssr)
        self.df_model = float(rank - k_constant)
        self.df_resid = float(n - rank)
        self.cov_type = cov_type
        self.use_t = (cov_type == "nonrobust") if use_t is None else use_t
        self._cov = None
        if cov_type in ("HC0", "HC1"):
            self._cov = self._hc(cov_type, beta, K, J)
        elif cov_type != "nonrobust":
            raise ValueError("cov_type must be 'nonrobust', 'HC0' or 'HC1'")
        self._weights_log = None
        self.fit_seconds = time.perf_counter() - t0

    def _setup(self, G, Kx, Ky, rank_tol):
        """Everything that follows from the Gram: constants, basis, rank, solution, normalized covariance."""
        model = self.model
        p = model.data.p
        xi = list(range(p)) + [p + 1]                     # B columns: shifted x, then the constant
        M = G[np.ix_(xi, xi)]
        b = G[xi, p]
        gyy, gy1, g11 = G[p, p], G[p, p + 1], G[p + 1, p + 1]
        # design columns: [virtual constant (add_constant)] + the columns of X. A constant column's shift is its
        # value and its shifted values are all exactly 0; the virtual one behaves the same way with value 1.
        vc = 1 if model.add_constant else 0
        zero_real = np.diag(M)[:p] == 0.0
        zero_var = np.concatenate([[True] * vc, zero_real])
        Kd = np.concatenate([[1.0] * vc, Kx])
        pd_ = p + vc
        const_cols = np.where(zero_var)[0]
        self._name_const = int(const_cols[0]) if const_cols.size else None
        k_constant, const_idx = 0, None
        check_implicit = False
        if model.hasconst is not False:
            if const_cols.size == 1:
                if Kd[const_cols[0]] != 0:
                    k_constant, const_idx = 1, int(const_cols[0])
                else:
                    check_implicit = True
            elif const_cols.size > 1:
                vals = Kd[const_cols]
                ones = np.where(vals == 1)[0]
                if ones.size:
                    k_constant, const_idx = 1, int(const_cols[ones[0]])
                elif (vals != 0).any():
                    k_constant, const_idx = 1, int(const_cols[np.argmax(vals != 0)])
                else:
                    check_implicit = True
            else:
                check_implicit = True
        # the basis: non-constant shifted columns of X and the constant
        J = [j for j in range(p) if not zero_real[j]] + [p]
        MJ = M[np.ix_(J, J)]
        bJ = b[J]
        V, lam, dsc, rJ, lam_all = _pinv_sym_psd(MJ, rank_tol)
        m = len(J)
        Wmap = np.zeros((m, pd_))                         # design column c = B_J @ Wmap[:, c]
        for r, j in enumerate(J[:-1]):
            Wmap[r, j + vc] = 1.0
        Wmap[m - 1, :] = Kd
        Rq = (np.sqrt(lam)[:, None] * V.T) * dsc[None, :]
        Rinv_t = V / np.sqrt(lam)[None, :] / dsc[:, None]
        F = Rq @ Wmap
        Fp, rank, svals = _pinv(F, F_TOL)
        if check_implicit and not model.hasconst:
            Faug = Rq @ np.column_stack([np.eye(m)[:, m - 1], Wmap])
            k_constant = int(_pinv(Faug, F_TOL)[1] == rank)
            const_idx = None
        elif model.hasconst:
            k_constant = 1
        self.k_constant, self._const_idx, self.rank = k_constant, const_idx, rank
        e1 = np.zeros(m); e1[m - 1] = 1.0
        z = Rinv_t.T @ bJ + Ky * (Rq @ e1)
        beta = Fp @ z
        self.normalized_cov_params = Fp @ Fp.T
        self._Fp, self._F, self._Rinv_t, self._J = Fp, F, Rinv_t, J
        self.centered_tss = gyy - gy1 * gy1 / g11
        self.uncentered_tss = gyy + 2 * Ky * gy1 + Ky * Ky * g11
        theta = Wmap @ beta - Ky * e1
        ssr = gyy - 2 * theta @ bJ + theta @ MJ @ theta
        self._ssr_gram = ssr
        self.wexog_singular_values = svals
        lam_min = float(lam_all.min() / lam_all.max()) if lam_all.size and lam_all.max() > 0 else 0.0
        self._Kd, self._vc = Kd, vc
        return {"beta": beta, "ssr": ssr, "J": J, "F": F, "Fp": Fp, "lam_min": lam_min}

    # --- robust covariance: one more Gram pass with rows scaled by w |r| --------------------------------------------
    def _hc(self, kind, beta, K, J):
        import mlx.core as mx
        from ._passes import gram_pass
        d = self.model.data
        p = d.p
        c0 = float(self._Kd @ beta - float(K[p]))
        _, _, _, routs = resid_pass(d, K, beta[self._vc:], c0, write_r=True)
        self._passes += 1
        scales = []
        for (Xb, yb, sb), r in zip(d.blocks(), routs):
            scales.append(mx.abs(r) if sb is None else mx.abs(r) * sb * sb)
        Gm = gram_pass(d, K, with_y=False, scales=scales, exact=self.exact_gram)
        self._passes += 1
        meatJ = Gm[np.ix_(J, J)]
        meatQ = self._Rinv_t.T @ meatJ @ self._Rinv_t
        cov = self._Fp @ meatQ @ self._Fp.T
        if kind == "HC1":
            cov = cov * self.nobs / self.df_resid
        return cov

    # --- statsmodels attributes -------------------------------------------------------------------------------------
    def _wrap(self, v, kind="params"):
        names = self.model.xnames
        if self.model._pandas and names is not None:
            import pandas as pd
            if kind == "params":
                return pd.Series(v, index=names)
            if kind == "cov":
                return pd.DataFrame(v, index=names, columns=names)
        return v

    @property
    def exog_names(self):
        if self.model.xnames is not None:
            return list(self.model.xnames)
        p = self.model.k_design
        if self._name_const is not None:                    # statsmodels: the first zero-variance column is 'const'
            names = [f"x{i}" for i in range(1, p)]
            names.insert(self._name_const, "const")
            return names
        return [f"x{i}" for i in range(1, p + 1)]

    @property
    def params(self):
        return self._wrap(self.params_arr)

    @property
    def scale(self):
        return self.ssr / self.df_resid

    def cov_params(self):
        c = self._cov if self._cov is not None else self.scale * self.normalized_cov_params
        return self._wrap(c, "cov")

    @property
    def _cov_arr(self):
        return self._cov if self._cov is not None else self.scale * self.normalized_cov_params

    @property
    def bse(self):
        return self._wrap(np.sqrt(np.maximum(np.diag(self._cov_arr), 0.0)))

    @property
    def tvalues(self):
        se = np.sqrt(np.maximum(np.diag(self._cov_arr), 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            return self._wrap(self.params_arr / se)

    @property
    def pvalues(self):
        t = np.asarray(self.tvalues)
        if self.use_t:
            return self._wrap(2 * stats.t.sf(np.abs(t), self.df_resid))
        return self._wrap(2 * stats.norm.sf(np.abs(t)))

    def conf_int(self, alpha=0.05):
        se = np.sqrt(np.maximum(np.diag(self._cov_arr), 0.0))
        q = stats.t.ppf(1 - alpha / 2, self.df_resid) if self.use_t else stats.norm.ppf(1 - alpha / 2)
        ci = np.column_stack([self.params_arr - q * se, self.params_arr + q * se])
        if self.model._pandas and self.model.xnames is not None:
            import pandas as pd
            return pd.DataFrame(ci, index=self.model.xnames, columns=[0, 1])
        return ci

    @property
    def ess(self):
        return (self.centered_tss if self.k_constant else self.uncentered_tss) - self.ssr

    @property
    def rsquared(self):
        return 1 - self.ssr / (self.centered_tss if self.k_constant else self.uncentered_tss)

    @property
    def rsquared_adj(self):
        return 1 - (self.nobs - self.k_constant) / self.df_resid * (1 - self.rsquared)

    @property
    def mse_model(self):
        return self.ess / self.df_model if self.df_model else np.nan

    @property
    def mse_resid(self):
        return self.ssr / self.df_resid

    @property
    def mse_total(self):
        return (self.centered_tss if self.k_constant else self.uncentered_tss) / (self.df_resid + self.df_model)

    def _wald_f(self):
        k = len(self.params_arr)
        idx = [i for i in range(k) if i != self._const_idx] if self.k_constant == 1 else list(range(k))
        if self.k_constant == 1 and self._const_idx is None:
            return np.nan, np.nan
        if not idx:
            return np.nan, np.nan
        Rb = self.params_arr[idx]
        C = self._cov_arr[np.ix_(idx, idx)]
        q = np.linalg.matrix_rank(C)
        f = float(Rb @ np.linalg.pinv(C) @ Rb) / q
        return f, float(stats.f.sf(f, q, self.df_resid))

    @property
    def fvalue(self):
        if self.cov_type != "nonrobust":
            return self._wald_f()[0]
        return self.mse_model / self.mse_resid

    @property
    def f_pvalue(self):
        if self.cov_type != "nonrobust":
            return self._wald_f()[1]
        return float(stats.f.sf(self.fvalue, self.df_model, self.df_resid))

    @property
    def llf(self):
        n2 = self.nobs / 2.0
        if self.model._weighted:
            if self._weights_log is None:
                w = self.model.data.w
                import mlx.core as mx
                self._weights_log = float(mx.sum(mx.log(w.astype(mx.float32))).item()) if isinstance(w, mx.array) \
                    else float(np.sum(np.log(np.asarray(w, np.float64))))
            return float(-np.log(self.ssr) * n2 - (1 + np.log(np.pi / n2)) * n2 + 0.5 * self._weights_log)
        return float(-n2 * np.log(2 * np.pi) - n2 * np.log(self.ssr / self.nobs) - n2)

    @property
    def aic(self):
        return -2 * self.llf + 2 * (self.df_model + self.k_constant)

    @property
    def bic(self):
        return -2 * self.llf + np.log(self.nobs) * (self.df_model + self.k_constant)

    @property
    def eigenvals(self):
        return np.sort(self.wexog_singular_values ** 2)[::-1]

    @property
    def condition_number(self):
        e = self.eigenvals
        return float(np.sqrt(e[0] / e[-1])) if e[-1] > 0 else np.inf

    @property
    def diagn(self):
        """skew, kurtosis, omnibus, Jarque-Bera, Durbin-Watson of the (whitened) residuals; needs the residual pass."""
        if self._diag is None:
            return None
        mom, dwn = self._diag
        return _tests.diagnostics(mom[0], mom[1], mom[2], mom[3], self.nobs, dwn)

    def predict(self, exog=None):
        """Predictions X @ params for new exog (same columns as the fit; without the virtual constant). With no
        argument, the fitted values."""
        if exog is None:
            return self.fittedvalues
        import mlx.core as mx
        if isinstance(exog, mx.array):
            b = mx.array(self.params_arr[self._vc:].astype(np.float32))
            out = exog.astype(mx.float32) @ b + float(self.params_arr[0] if self._vc else 0.0)
            return np.array(out)
        X = np.asarray(exog.to_numpy() if hasattr(exog, "to_numpy") else exog, dtype=np.float64)
        if X.ndim == 1:
            X = X[:, None] if self.model.data.p == 1 else X[None, :]
        out = X @ self.params_arr[self._vc:]
        return out + (self.params_arr[0] if self._vc else 0.0)

    @property
    def resid(self):
        """y - X params for every row (one more pass over the data, then cached)."""
        if getattr(self, "_resid", None) is None:
            import mlx.core as mx
            d = self.model.data
            c0 = float(self._Kd @ self.params_arr - float(self._K[d.p]))
            _, _, _, routs = resid_pass(d, self._K, self.params_arr[self._vc:], c0, write_r=True)
            self._resid = np.concatenate([np.array(r).astype(np.float64) for r in routs])
        return self._resid

    @property
    def wresid(self):
        if not self.model._weighted:
            return self.resid
        w = self.model.data.w
        w = np.array(w) if not isinstance(w, np.ndarray) else w
        return np.sqrt(np.asarray(w, np.float64)) * self.resid

    @property
    def fittedvalues(self):
        y = self.model.data.y
        y = np.array(y).astype(np.float64) if not isinstance(y, np.ndarray) else y.astype(np.float64)
        return y - self.resid

    def summary(self, alpha=0.05):
        from ._summary import summary
        return summary(self, alpha)
