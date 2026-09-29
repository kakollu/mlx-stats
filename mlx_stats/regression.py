"""Linear regression with statsmodels' results, from one GPU pass for X^T X and one for the residuals.

    res = OLS(y, X).fit()            # X as given: add a constant column yourself (add_constant), like statsmodels
    res = OLS(y, X, add_constant=True).fit()      # or let the fit add one without copying X
    res.params, res.bse, res.pvalues, res.conf_int(), res.rsquared, res.fvalue, res.llf, res.aic, ...
    print(res.summary())
    WLS(y, X, weights=w).fit(cov_type="HC1")

How the numbers are made. The data layer shifts every column by a value K near its mean and, when a column's spread is
far from 1, scales it by a power of two (undone exactly at the end). Pass 1 (`_gram`) returns
G = sum w [a, a_y, 1][a, a_y, 1]^T with a = x - K, a_y = y - K_y. Every quantity below is float64 arithmetic on G.
The basis for the solve is B = [a_j for non-constant columns j, 1]: shifted columns are close to centred, so B^T W B is
well conditioned even when the raw columns have large means. X = B W_map exactly (W_map holds 1s and the shifts), and
with the eigendecomposition B^T W B = V L V^T, X = Q F where Q has orthonormal columns and F = L^(1/2) V^T W_map is
small. From F: statsmodels' minimum-norm parameters, the normalized covariance pinv(X^T X), the rank, and the singular
values behind the condition number - the definitions statsmodels uses, in a basis where float32 accumulation is enough.

Pass 2 (`_resid`) computes the residuals with two-float coefficients: the exact sum of squared residuals and total sums
of squares, the residual power sums behind skew, kurtosis, omnibus and Jarque-Bera, Durbin-Watson, and an exact
X^T W r for iterative refinement, so the parameters converge to what a float64 solver gives on the same float32 data.
Robust (HC0/HC1) covariance adds one more Gram pass with each row scaled by its residual.

Precision. The fast Gram is accurate to ~3e-8 of sqrt(G_ii G_jj) per entry, which fixes standard errors to ~3e-8 /
lambda_min, lambda_min the smallest eigenvalue of the scaled B^T W B. Below EXACT_TOL (1e-2) the Gram is recomputed
exactly (`_gram_exact`, ~1e-14), so standard errors stay within ~3e-6 of a float64 fit without it and ~1e-9 with it.
Parameters are refined to ~1e-9..1e-14 either way. A warning is raised when the design itself (column-scaled) has a
squared singular-value ratio below DATA_TOL (1e-10, cond ~1e5): float32 data cannot resolve such a design.
"""
import time
import warnings
import numpy as np
from scipy import stats
from ._data import DeviceData
from ._passes import gram_pass, resid_pass
from . import _tests

EXACT_TOL = 1e-2         # scaled eigenvalue of B^T W B below which the fit recomputes the Gram exactly
PRECISE_OFF_WARN = 1e-4  # ...below which precise=False warns that standard errors are limited by float32
RANK_TOL = 1e-6          # ...below which a direction counts as exact collinearity, from the fast (float32) Gram
RANK_TOL_EXACT = 1e-12   # ...the same, from the exact Gram
F_TOL = 1e-10            # relative singular value of the column-scaled F below which a direction counts as zero
DATA_TOL = 1e-10         # squared singular-value ratio of the column-scaled design below which float32 data cannot resolve it


class SingularDesignWarning(UserWarning):
    pass


class ConvergenceWarning(UserWarning):
    pass


class OLS:
    """Ordinary least squares, statsmodels argument order: OLS(endog, exog)."""
    _kind = "OLS"

    def __init__(self, endog, exog, hasconst=None, resident=None, add_constant=False):
        """add_constant=True fits an intercept without copying X to append a column (statsmodels needs
        sm.add_constant(X), a full copy); the constant is reported first, named 'const'. Like sm.add_constant, it is
        skipped when X already has a nonzero constant column."""
        self.data = DeviceData(exog, endog, None, resident, ones_if_empty=add_constant)
        self.hasconst = hasconst
        self.add_constant = add_constant
        self._weighted = False
        self._init_names()

    def _init_names(self):
        d = self.data
        self.yname = str(d.yname) if d.yname is not None else "y"
        self._pandas = d.xnames is not None or d.row_index is not None
        self.nobs = float(d.n)

    def fit(self, cov_type="nonrobust", use_t=None, refine=True, precise=None, max_passes=4):
        """cov_type: 'nonrobust', 'HC0' or 'HC1'. refine: run the residual pass (exact SSR, diagnostics, refinement).
        With refine=False only the Gram pass runs (parameters from X^T X, SSR from the Gram, no residual diagnostics),
        unless the fit is so tight that the Gram's SSR would be unreliable; then the residual pass runs anyway.
        precise: None recomputes X^T X exactly when the design is ill conditioned; True always; False never."""
        return RegressionResults(self, cov_type, use_t, refine, max_passes, precise)


class WLS(OLS):
    """Weighted least squares: WLS(endog, exog, weights). Weights are inverse variances, as in statsmodels."""
    _kind = "WLS"

    def __init__(self, endog, exog, weights=1.0, hasconst=None, resident=None, add_constant=False):
        self.data = DeviceData(exog, endog, weights, resident, ones_if_empty=add_constant)
        self.hasconst = hasconst
        self.add_constant = add_constant
        self._weighted = self.data.w is not None
        self._init_names()


def _pinv_sym_psd(A, tol, dropped=False):
    """eigen-pinv of a symmetric PSD matrix after scaling to unit diagonal. -> (V_r, lam_r (scaled), d, rank, lam_all),
    plus the dropped (null) eigenvectors when dropped=True."""
    d = np.sqrt(np.maximum(np.diag(A), 0.0))
    d = np.where(d > 0, d, 1.0)
    S = A / np.outer(d, d)
    lam, V = np.linalg.eigh((S + S.T) / 2)
    keep = lam > tol * max(lam.max(), 0.0)
    if dropped:
        return V[:, keep], lam[keep], d, int(keep.sum()), lam, V[:, ~keep]
    return V[:, keep], lam[keep], d, int(keep.sum()), lam


def _pinv(F, tol, N_hint=None, colscale=None):
    """The Moore-Penrose pseudo-inverse of F (p x r), its rank, the column-scaled singular values, and a basis of the
    null space of F (or None).

    Everything is computed with the columns scaled to unit norm, so a column's units do not affect accuracy. For a
    full-rank F that is simply the (unique) solution. For a rank-deficient F the scaled inverse gives a generalized
    inverse G; the Moore-Penrose inverse is G projected onto the orthogonal complement of null(F) in the original
    metric, which is what makes the solution minimum-norm in the original units, as statsmodels' pinv is."""
    p = F.shape[1]
    cn = np.sqrt((F * F).sum(0)) if colscale is None else colscale
    cn = np.where(cn > 0, cn, 1.0)
    Fs = F / cn
    U, s, Vt = np.linalg.svd(Fs, full_matrices=True)
    rank = int((s > tol * (s.max() if s.size else 0.0)).sum())
    G = (Vt[:rank].T / s[:rank]) @ U[:, :rank].T / cn[:, None]
    N = None
    if rank < p:
        N = Vt[rank:].T / cn[:, None]                    # null(F) = D^-1 null(F D^-1)
        if N_hint is not None and N_hint.shape[1] == p - rank:
            N = N_hint                                   # the same space, computed where it is well conditioned
        Qn, _ = np.linalg.qr(N)
        G = G - Qn @ (Qn.T @ G)
    return G, rank, s, N


def _design_null(Ub, zero_var, Kd):
    """A basis of null(X) in design coordinates, from a basis Ub of null(B_J) (rows: the non-constant columns in
    order, then the constant 1). X = B_J Wmap where Wmap copies the non-constant coefficients and puts Kd . beta on
    the constant, so beta is null exactly when beta_nonconst = Ub_nc t and Kd_const . beta_const = Ub_1 t - Kd_nc .
    Ub_nc t. Solved directly - no factorization of a matrix holding the column means next to O(1) entries - and the
    one sum that cancels (Ub_1 - Kd_nc . Ub_nc) is taken exactly rounded (math.fsum)."""
    import math
    pd_ = len(zero_var)
    cc = [c for c in range(pd_) if zero_var[c]]
    nc = [c for c in range(pd_) if not zero_var[c]]
    U_nc, U_1 = Ub[:-1, :], Ub[-1, :]
    Kc, Kn = Kd[cc], Kd[nc]
    kB = Ub.shape[1]
    r = np.array([math.fsum([U_1[i]] + [-float(Kn[j]) * float(U_nc[j, i]) for j in range(len(nc))]) for i in range(kB)])
    vecs = []
    if cc and np.any(Kc != 0):
        for i in range(kB):
            v = np.zeros(pd_)
            v[nc] = U_nc[:, i]
            v[cc] = r[i] * Kc / (Kc @ Kc)
            vecs.append(v)
        if len(cc) > 1:                                  # combinations of constant columns that cancel
            _, _, Vt = np.linalg.svd(Kc[None, :], full_matrices=True)
            for row in Vt[1:]:
                v = np.zeros(pd_); v[cc] = row; vecs.append(v)
    else:
        if kB:                                           # no constant column: need r . t = 0
            _, sv, Vt = np.linalg.svd(r[None, :], full_matrices=True)
            T = Vt[1:] if (sv.size and sv[0] > 0) else Vt
            for t in T:
                v = np.zeros(pd_); v[nc] = U_nc @ t; vecs.append(v)
        for c in cc:                                     # all-zero columns
            v = np.zeros(pd_); v[c] = 1.0; vecs.append(v)
    return np.column_stack(vecs) if vecs else None


def _safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(np.float64(a), np.float64(b))


class RegressionResults:
    """statsmodels RegressionResults attribute names; arrays are pandas objects when the input was pandas."""

    def __init__(self, model, cov_type, use_t, refine, max_passes, precise=None):
        t0 = time.perf_counter()
        self.model = model
        d = model.data
        n, p = d.n, d.p
        self.nobs = float(n)
        cov_type = {"nonrobust": "nonrobust", "hc0": "HC0", "hc1": "HC1"}.get(str(cov_type).lower())
        if cov_type is None:
            raise ValueError("cov_type must be 'nonrobust', 'HC0' or 'HC1'")
        K = d.shift()
        self._K = K
        Kx, Ky = K[:p].astype(np.float64), float(K[p])
        G = gram_pass(d, K, exact=bool(precise))
        self._passes = 1
        d.check_weights()
        if not np.all(np.isfinite(G)):
            raise ValueError("exog, endog or weights contain inf or NaN (or values beyond the float32 range, ~3e38)")
        # constant columns are judged on the unweighted data, as statsmodels does; for OLS the Gram's zero diagonal
        # says it exactly, for WLS a zero-weight row could hide a column's variation
        self._zero_real = d.unweighted_constant() if model._weighted else (np.diag(G)[:p] == 0.0)
        self._G_unweighted = None
        if model._weighted and d.has_zero_weight:
            self._G_unweighted = gram_pass(d, K, scales=[None] * len(list(d.blocks())), exact=True)
            self._passes += 1
        self.exact_gram = bool(precise)
        info = self._setup(G, Kx, Ky, RANK_TOL_EXACT if precise else RANK_TOL)
        if precise is None and info["lam_min"] < EXACT_TOL:
            G = gram_pass(d, K, exact=True)                  # collinear: the float32 Gram limits the covariance
            self._passes += 1
            self.exact_gram = True
            info = self._setup(G, Kx, Ky, RANK_TOL_EXACT)
        if precise is False and info["lam_min"] < PRECISE_OFF_WARN:
            warnings.warn(f"precise=False on a collinear design (smallest scaled eigenvalue {info['lam_min']:.1e}): "
                          "standard errors are limited by the float32 Gram; use precise=None or True.",
                          SingularDesignWarning, stacklevel=3)
        beta, ssr, J, F, Fp = info["beta"], info["ssr"], info["J"], info["F"], info["Fp"]
        self._lam_min = info["lam_min"]
        rank, k_constant = self.rank, self.k_constant
        kd = p + self._vc
        if rank < kd:
            warnings.warn("The design matrix is rank-deficient. The model parameters are not uniquely determined.",
                          SingularDesignWarning, stacklevel=3)
        elif info["lam_design"] < DATA_TOL:
            warnings.warn(f"Near-collinear design (squared singular-value ratio of the scaled design "
                          f"{info['lam_design']:.1e}): float32 data cannot resolve it; parameters may differ from a "
                          f"float64 fit on float64 data.", SingularDesignWarning, stacklevel=3)
        self._diag = None
        self._refine_steps = []
        tss_ref = self.centered_tss_w if k_constant else self.uncentered_tss_w
        # the Gram's SSR is a difference of sums known to ~3e-8 of the total: useless when the fit is tight
        gram_ssr_ok = np.isfinite(ssr) and ssr > 1e-4 * max(tss_ref, 1e-300)
        run_resid = refine or not gram_ssr_ok
        if run_resid:
            sig = np.sqrt(max(ssr if gram_ssr_ok else 1e-6 * tss_ref, 1e-300) / max(n, 1))
            rscale = float(2.0 ** -np.round(np.log2(sig))) if np.isfinite(sig) and sig > 0 else 1.0
            converged = False
            for it in range(max_passes):
                c0 = float(self._Kd @ beta - Ky)
                grad, sums, _ = resid_pass(d, K, beta[self._vc:], c0, rscale)
                self._passes += 1
                ssr_pass = sums[8]                                     # sum w r^2, exact
                self.centered_tss_w = sums[5] - sums[6] * sums[6] / sums[7]
                self.uncentered_tss_w = sums[5] + 2 * Ky * sums[6] + Ky * Ky * sums[7]
                qe = self._Rinv_t.T @ grad[J]                          # Q^T W e
                delta = Fp @ qe
                gain = float(np.sum((F @ delta) ** 2))                 # SSR decrease of the refinement step
                beta = beta + delta
                ssr = max(ssr_pass - gain, 0.0)
                self._diag = (sums[:4] / rscale ** np.arange(1, 5), sums[4] / rscale ** 2)
                se = np.sqrt(np.maximum(np.diag(self._ncov_w), 0.0) * ssr / max(n - rank, 1))
                tol = 1e-10 * np.abs(beta) + 1e-8 * se + 1e-300
                contraction = min(1.0, 1e-6 / max(info["lam_min"], 1e-300))
                self._refine_steps.append((float(np.max(np.abs(delta) / np.maximum(np.abs(beta), 1e-300), initial=0)),
                                           gain / max(ssr_pass, 1e-300)))
                moved = np.all(np.abs(delta) <= tol) or np.all(contraction * np.abs(delta) <= tol)
                if moved and gain <= 1e-12 * max(ssr_pass, 1e-300):
                    converged = True
                    break
                if not refine and gain <= 1e-12 * max(ssr_pass, 1e-300):
                    converged = True                                   # SSR was all that was needed
                    break
            if not converged:
                warnings.warn(f"refinement did not converge in {max_passes} passes (last relative step "
                              f"{self._refine_steps[-1][0]:.1e}); results may be imprecise", ConvergenceWarning, stacklevel=3)
            if not refine:
                self._diag = None                                      # diagnostics were not asked for
        self.cov_type = cov_type
        self.use_t = (cov_type == "nonrobust") if use_t is None else use_t
        self.df_model = float(rank - k_constant)
        self.df_resid = float(n - rank)
        # --- back to the units of the data: X = X' / c, y = y' / c_y (powers of two: exact) -------------------------
        c = np.concatenate([[1.0] * self._vc, d.xscale])
        cy = d.yscale
        self._c, self._cy = c, cy
        self._beta_w = beta
        self.params_arr = beta * c / cy
        self.normalized_cov_params = self._ncov_w * np.outer(c, c)
        self.ssr = float(ssr) / cy ** 2
        self.centered_tss = self.centered_tss_w / cy ** 2
        self.uncentered_tss = self.uncentered_tss_w / cy ** 2
        Fu = F / c[None, :]
        sv = np.linalg.svd(Fu, compute_uv=False) if Fu.size else np.zeros(0)
        m = min(n, kd)
        self.wexog_singular_values = np.concatenate([np.sort(sv)[::-1][:m], np.zeros(max(0, m - sv.size))])
        if rank < kd:
            self.wexog_singular_values[rank:] = 0.0
        self._null = None if info["N"] is None else info["N"] * c[:, None]
        self._cov = None
        if cov_type in ("HC0", "HC1"):
            self._cov = self._hc(cov_type, beta, K, J) * np.outer(c, c) / cy ** 2
        self._weights_log = None
        self.fit_seconds = time.perf_counter() - t0

    def _setup(self, G, Kx, Ky, rank_tol):
        """Everything that follows from the Gram: constants, basis, rank, solution, normalized covariance."""
        model = self.model
        d = model.data
        p = d.p
        xi = list(range(p)) + [p + 1]                     # B columns: shifted x, then the constant
        M = G[np.ix_(xi, xi)]
        b = G[xi, p]
        gyy, gy1, g11 = G[p, p], G[p, p + 1], G[p + 1, p + 1]
        zero_real = self._zero_real
        Kx_true = Kx / d.xscale                            # the constant columns' values in the data's units
        # design columns: [virtual constant] + the columns of X. Like sm.add_constant, the virtual constant is skipped
        # when X already has a nonzero constant column. A constant column's shifted values are all exactly 0.
        # add_constant="add" (LinearRegression, as scikit-learn) always adds it
        vc = 1 if (model.add_constant == "add" or (model.add_constant and not np.any(zero_real & (Kx_true != 0)))) else 0
        zero_var = np.concatenate([[True] * vc, zero_real])
        Kd = np.concatenate([[1.0] * vc, Kx])               # shifts in working units (the virtual constant: 1)
        Kd_true = np.concatenate([[1.0] * vc, Kx_true])
        pd_ = p + vc
        const_cols = np.where(zero_var)[0]
        self._name_const = int(const_cols[0]) if const_cols.size else None
        k_constant, const_idx = 0, None
        check_implicit = False
        if model.hasconst is not False:
            if const_cols.size == 1:
                if Kd_true[const_cols[0]] != 0:
                    k_constant, const_idx = 1, int(const_cols[0])
                else:
                    check_implicit = True
            elif const_cols.size > 1:
                vals = Kd_true[const_cols]
                ones = np.where(vals == 1)[0]
                if ones.size:
                    k_constant, const_idx = 1, int(const_cols[ones[0]])
                elif (vals != 0).any():
                    k_constant, const_idx = 1, int(const_cols[np.argmax(vals != 0)])
                else:
                    check_implicit = True
            else:
                check_implicit = True
        J = [j for j in range(p) if not zero_real[j]] + [p]
        MJ = M[np.ix_(J, J)]
        bJ = b[J]
        V, lam, dsc, rJ, lam_all, Vnull = _pinv_sym_psd(MJ, rank_tol, dropped=True)
        m = len(J)
        Wmap = np.zeros((m, pd_))                         # design column c = B_J @ Wmap[:, c]
        for r, j in enumerate(J[:-1]):
            Wmap[r, j + vc] = 1.0
        Wmap[m - 1, :] = Kd
        Rq = (np.sqrt(lam)[:, None] * V.T) * dsc[None, :]
        Rinv_t = V / np.sqrt(lam)[None, :] / dsc[:, None]
        F = Rq @ Wmap
        # scale F's columns by each design column's size in the shifted basis, not by F's own column norms: a column
        # that is exactly zero in the (weighted) problem shows up in F as roundoff, and normalising it by itself would
        # turn that roundoff into a full direction
        colscale = np.sqrt(((dsc[:, None] * Wmap) ** 2).sum(0))
        Fp, rank, ss, N = _pinv(F, F_TOL, colscale=colscale)
        if rank < pd_:
            # null(X) = {beta : Wmap beta in null(B_J)}. null(B_J) comes from the shifted (near-centred) basis, where it
            # is well conditioned; Wmap is exact. The SVD of F above gets the same space, but F carries the column
            # means, so with large means its null vectors lose digits - and they decide the minimum-norm solution.
            NB = _design_null(Vnull / dsc[:, None], zero_var, Kd)
            if NB is not None and NB.shape[1] == pd_ - rank and np.linalg.matrix_rank(NB) == pd_ - rank:
                Fp, rank, ss, N = _pinv(F, F_TOL, N_hint=NB, colscale=colscale)
        if check_implicit and not model.hasconst:
            if getattr(self, "_G_unweighted", None) is not None:
                # zero weights change ranks: statsmodels decides the implicit constant on the unweighted exog
                k_constant = int(self._implicit_const(self._G_unweighted, Kx, rank_tol, vc))
            else:
                WA = np.column_stack([np.eye(m)[:, m - 1], Wmap])
                k_constant = int(_pinv(Rq @ WA, F_TOL, colscale=np.sqrt(((dsc[:, None] * WA) ** 2).sum(0)))[1] == rank)
            const_idx = None
        elif model.hasconst:
            k_constant = 1
        self.k_constant, self._const_idx, self.rank, self._vc = k_constant, const_idx, rank, vc
        e1 = np.zeros(m); e1[m - 1] = 1.0
        z = Rinv_t.T @ bJ + Ky * (Rq @ e1)
        beta = Fp @ z
        self._ncov_w = Fp @ Fp.T
        self._Fp, self._F, self._Rinv_t, self._J, self._Kd = Fp, F, Rinv_t, J, Kd
        self.centered_tss_w = gyy - gy1 * gy1 / g11
        self.uncentered_tss_w = gyy + 2 * Ky * gy1 + Ky * Ky * g11
        theta = Wmap @ beta - Ky * e1
        ssr = gyy - 2 * theta @ bJ + theta @ MJ @ theta
        lam_min = float(lam_all.min() / lam_all.max()) if lam_all.size and lam_all.max() > 0 else 0.0
        ssn = ss[:rank]
        lam_design = float((ssn.min() / ssn.max()) ** 2) if ssn.size and ssn.max() > 0 else 0.0
        return {"beta": beta, "ssr": ssr, "J": J, "F": F, "Fp": Fp, "lam_min": lam_min, "lam_design": lam_design, "N": N}

    def _implicit_const(self, G, Kx, rank_tol, vc):
        """rank([1, X]) == rank(X) on the Gram G (unweighted)."""
        p = self.model.data.p
        xi = list(range(p)) + [p + 1]
        M = G[np.ix_(xi, xi)]
        J = [j for j in range(p) if not self._zero_real[j]] + [p]
        V, lam, dsc, _, _ = _pinv_sym_psd(M[np.ix_(J, J)], rank_tol)
        m = len(J)
        Wmap = np.zeros((m, p + vc))
        for r, j in enumerate(J[:-1]):
            Wmap[r, j + vc] = 1.0
        Wmap[m - 1, :] = np.concatenate([[1.0] * vc, Kx])
        Rq = (np.sqrt(lam)[:, None] * V.T) * dsc[None, :]
        WA = np.column_stack([np.eye(m)[:, m - 1], Wmap])
        rank = _pinv(Rq @ Wmap, F_TOL, colscale=np.sqrt(((dsc[:, None] * Wmap) ** 2).sum(0)))[1]
        return _pinv(Rq @ WA, F_TOL, colscale=np.sqrt(((dsc[:, None] * WA) ** 2).sum(0)))[1] == rank

    # --- robust covariance: one more Gram pass with rows scaled by w |r| (working units) ------------------------------
    def _hc(self, kind, beta, K, J):
        import mlx.core as mx
        d = self.model.data
        p = d.p
        c0 = float(self._Kd @ beta - float(K[p]))
        _, _, routs = resid_pass(d, K, beta[self._vc:], c0, 1.0, write_r=True)
        self._passes += 1
        scales = []
        for (Xb, yb, sb, wb), r in zip(d.blocks(), routs):
            scales.append(mx.abs(r) if wb is None else mx.abs(r) * wb)
        Gm = gram_pass(d, K, with_y=False, scales=scales, exact=self.exact_gram)
        self._passes += 1
        meatJ = Gm[np.ix_(J, J)]
        meatQ = self._Rinv_t.T @ meatJ @ self._Rinv_t
        cov = self._Fp @ meatQ @ self._Fp.T
        if kind == "HC1":
            cov = cov * _safe_div(self.nobs, self.df_resid)
        return cov

    # --- statsmodels attributes -------------------------------------------------------------------------------------
    @property
    def _labels(self):
        """Parameter labels: the data's column labels (types kept, as pandas has them) or generated names."""
        d = self.model.data
        if d.xnames is not None:
            return (["const"] if self._vc else []) + list(d.xnames)
        return self.exog_names

    @property
    def exog_names(self):
        d = self.model.data
        if d.xnames is not None:
            return [str(v) for v in self._labels]
        k = d.p + self._vc
        if self._name_const is not None:                    # statsmodels: the first zero-variance column is 'const'
            names = [f"x{i}" for i in range(1, k)]
            names.insert(self._name_const, "const")
            return names
        return [f"x{i}" for i in range(1, k + 1)]

    def _wrap(self, v, kind="params"):
        if self.model._pandas:
            import pandas as pd
            names = self._labels
            if kind == "params":
                return pd.Series(v, index=names)
            if kind == "cov":
                return pd.DataFrame(v, index=names, columns=names)
        return v

    def _wrap_rows(self, v):
        idx = self.model.data.row_index
        if idx is not None:
            import pandas as pd
            return pd.Series(v, index=idx)
        return v

    @property
    def params(self):
        return self._wrap(self.params_arr)

    @property
    def scale(self):
        if self.df_resid == 0:
            return np.inf                                  # saturated: statsmodels' ssr/0 on roundoff is inf too
        return _safe_div(self.ssr, self.df_resid)

    @property
    def _cov_arr(self):
        return self._cov if self._cov is not None else self.scale * self.normalized_cov_params

    def cov_params(self):
        return self._wrap(self._cov_arr, "cov")

    @property
    def bse(self):
        return self._wrap(np.sqrt(np.maximum(np.diag(self._cov_arr), 0.0)))

    @property
    def tvalues(self):
        se = np.sqrt(np.maximum(np.diag(self._cov_arr), 0.0))
        return self._wrap(_safe_div(self.params_arr, se))

    @property
    def pvalues(self):
        t = np.asarray(self.tvalues, float)
        if self.use_t:
            return self._wrap(2 * stats.t.sf(np.abs(t), self.df_resid))
        return self._wrap(2 * stats.norm.sf(np.abs(t)))

    def conf_int(self, alpha=0.05):
        se = np.sqrt(np.maximum(np.diag(self._cov_arr), 0.0))
        q = stats.t.ppf(1 - alpha / 2, self.df_resid) if self.use_t else stats.norm.ppf(1 - alpha / 2)
        ci = np.column_stack([self.params_arr - q * se, self.params_arr + q * se])
        if self.model._pandas:
            import pandas as pd
            return pd.DataFrame(ci, index=self._labels, columns=[0, 1])
        return ci

    @property
    def ess(self):
        return (self.centered_tss if self.k_constant else self.uncentered_tss) - self.ssr

    @property
    def rsquared(self):
        return 1 - _safe_div(self.ssr, self.centered_tss if self.k_constant else self.uncentered_tss)

    @property
    def rsquared_adj(self):
        return 1 - _safe_div(self.nobs - self.k_constant, self.df_resid) * (1 - self.rsquared)

    @property
    def mse_model(self):
        return _safe_div(self.ess, self.df_model) if self.df_model else np.nan

    @property
    def mse_resid(self):
        return _safe_div(self.ssr, self.df_resid) if self.df_resid else np.nan

    @property
    def mse_total(self):
        if self.df_resid + self.df_model == 0:
            return np.nan
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
        if q == 0:
            return np.nan, np.nan
        f = float(Rb @ np.linalg.pinv(C) @ Rb) / q
        return f, float(stats.f.sf(f, q, self.df_resid))

    @property
    def fvalue(self):
        if self.cov_type != "nonrobust":
            return self._wald_f()[0]
        return _safe_div(self.mse_model, self.mse_resid)

    @property
    def f_pvalue(self):
        if self.cov_type != "nonrobust":
            return self._wald_f()[1]
        return float(stats.f.sf(self.fvalue, self.df_model, self.df_resid))

    @property
    def llf(self):
        n2 = self.nobs / 2.0
        with np.errstate(divide="ignore", invalid="ignore"):
            if self.model._weighted:
                if self._weights_log is None:
                    w = self.model.data.w
                    import mlx.core as mx
                    w = np.array(w.astype(mx.float32)) if isinstance(w, mx.array) else np.asarray(w)
                    self._weights_log = float(np.sum(np.log(w.astype(np.float64))))
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
        return float(np.sqrt(e[0] / e[-1])) if e.size and e[-1] > 0 else np.inf

    @property
    def diagn(self):
        """skew, kurtosis, omnibus, Jarque-Bera, Durbin-Watson of the (whitened) residuals; needs the residual pass."""
        if self._diag is None:
            return None
        mom, dwn = self._diag
        return _tests.diagnostics(mom[0], mom[1], mom[2], mom[3], self.nobs, dwn)

    def predict(self, exog=None):
        """Predictions X @ params for new exog (the columns of the fit, without the virtual constant). With no
        argument, the fitted values. Computed in float64 on the host."""
        if exog is None:
            return self.fittedvalues
        import mlx.core as mx
        idx = getattr(exog, "index", None) if hasattr(exog, "to_numpy") else None
        if isinstance(exog, mx.array):
            X = np.array(exog.astype(mx.float32)).astype(np.float64)
        else:
            X = np.asarray(exog.to_numpy() if hasattr(exog, "to_numpy") else exog, dtype=np.float64)
        b = self.params_arr[self._vc:]
        if X.ndim == 1:
            X = X[:, None] if b.size == 1 else X[None, :]
        out = X @ b + (self.params_arr[0] if self._vc else 0.0)
        if idx is not None:
            import pandas as pd
            return pd.Series(out, index=idx)
        return out

    def _resid_w(self):
        """Residuals in the data's units, one more pass over the data (cached)."""
        if getattr(self, "_resid", None) is None:
            d = self.model.data
            c0 = float(self._Kd @ self._beta_w - float(self._K[d.p]))
            _, _, routs = resid_pass(d, self._K, self._beta_w[self._vc:], c0, 1.0, write_r=True)
            self._resid = np.concatenate([np.array(r).astype(np.float64) for r in routs]) / self._cy
        return self._resid

    @property
    def resid(self):
        """y - X params for every row (one more pass over the data, then cached)."""
        return self._wrap_rows(self._resid_w())

    @property
    def wresid(self):
        r = self._resid_w()
        if self.model._weighted:
            import mlx.core as mx
            w = self.model.data.w
            w = np.array(w.astype(mx.float32)) if isinstance(w, mx.array) else np.asarray(w)
            r = np.sqrt(w.astype(np.float64)) * r
        return self._wrap_rows(r)

    @property
    def fittedvalues(self):
        import mlx.core as mx
        y = self.model.data.y
        y = np.array(y.astype(mx.float32)).astype(np.float64) if isinstance(y, mx.array) else np.asarray(y, np.float64)
        return self._wrap_rows(y - self._resid_w())

    def summary(self, alpha=0.05):
        from ._summary import summary
        return summary(self, alpha)
