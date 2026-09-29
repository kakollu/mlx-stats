#!/usr/bin/env python3
"""Every case fits the same float32-valued data with mlx-stats (GPU) and statsmodels (float64, CPU) and compares
every reported quantity. Run: .venv/bin/python tests/test_vs_statsmodels.py [-v]"""
import sys, warnings, numpy as np, pandas as pd, statsmodels.api as sm, mlx.core as mx
import mlx_stats as ms

VERBOSE = "-v" in sys.argv
rng = np.random.default_rng(12345)
f32 = lambda a: np.asarray(a, np.float64).astype(np.float32).astype(np.float64)
SCALARS = ["rsquared", "rsquared_adj", "fvalue", "f_pvalue", "llf", "aic", "bic", "ssr", "ess", "centered_tss",
           "uncentered_tss", "mse_model", "mse_resid", "mse_total", "condition_number"]
EXACT = ["df_model", "df_resid", "k_constant", "nobs"]
results = []


def close(a, b, rtol, atol=0.0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    both_nan = np.isnan(a) & np.isnan(b)
    both_inf = np.isinf(a) & np.isinf(b) & (np.sign(a) == np.sign(b))
    ok = both_nan | both_inf | (np.abs(a - b) <= atol + rtol * np.abs(b))
    return bool(ok.all()), float(np.max(np.where(both_nan | both_inf, 0, np.abs(a - b) / np.maximum(np.abs(b), 1e-300)), initial=0))


def compare(name, y, X, sm_kw=None, ms_kw=None, fit_kw=None, model="OLS", rtol_p=1e-7, rtol_se=1e-5, rtol_s=1e-6,
            ms_X=None, ms_y=None, skip=()):
    sm_kw, ms_kw, fit_kw = sm_kw or {}, ms_kw or {}, fit_kw or {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r0 = getattr(sm, model)(y, X, **sm_kw).fit(**fit_kw)
        r1 = getattr(ms, model)(ms_y if ms_y is not None else y, ms_X if ms_X is not None else X, **ms_kw).fit(**fit_kw)
    fails, worst = [], {}
    se0 = np.asarray(r0.bse, float)
    # parameters: relative, or a millionth of a standard error when the parameter is ~0
    ok, w = close(r1.params, r0.params, rtol_p, atol=0.0)
    ok2 = np.all(np.abs(np.asarray(r1.params, float) - np.asarray(r0.params, float)) <= rtol_p * np.abs(np.asarray(r0.params, float)) + 1e-6 * np.nan_to_num(se0))
    worst["params"] = w
    if not ok2: fails.append(f"params rel {w:.1e}")
    for a, tol in [("bse", rtol_se), ("tvalues", rtol_se)]:
        ok, w = close(getattr(r1, a), getattr(r0, a), tol, atol=1e-12); worst[a] = w
        if not ok: fails.append(f"{a} rel {w:.1e}")
    ok, w = close(r1.pvalues, r0.pvalues, 1e-4, atol=1e-9); worst["pvalues"] = w
    if not ok: fails.append(f"pvalues rel {w:.1e}")
    ok, w = close(r1.conf_int(), np.asarray(r0.conf_int()), rtol_se, atol=1e-6 * float(np.nanmax(se0)) if se0.size else 0); worst["conf_int"] = w
    if not ok: fails.append(f"conf_int rel {w:.1e}")
    for a in SCALARS:
        if a in skip: continue
        v0, v1 = float(getattr(r0, a)), float(getattr(r1, a))
        if a == "condition_number" and v0 > 1e12 and v1 == np.inf:
            continue                                   # exact collinearity: statsmodels' roundoff ~1e15, ours inf
        tol = 1e-4 if a == "condition_number" else (1e-4 if a in ("f_pvalue",) else rtol_s)
        ok, w = close(v1, v0, tol, atol=1e-300 if a == "f_pvalue" else 1e-12); worst[a] = w
        if not ok: fails.append(f"{a} {v1:.10g} vs {v0:.10g}")
    for a in EXACT:
        if float(getattr(r1, a)) != float(getattr(r0, a)):
            fails.append(f"{a} {getattr(r1, a)} vs {getattr(r0, a)}")
    if r1.diagn is not None and "diagn" not in skip:
        from statsmodels.stats.stattools import jarque_bera, omni_normtest, durbin_watson
        wres = r0.wresid
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            jb, jbp, sk, ku = jarque_bera(wres); om, omp = omni_normtest(wres); dw = durbin_watson(wres)
        d = r1.diagn
        for k, v in [("jarque_bera", jb), ("jarque_bera_pvalue", jbp), ("skew", sk), ("kurtosis", ku), ("omnibus", om), ("omnibus_pvalue", omp), ("durbin_watson", dw)]:
            ok, w = close(d[k], v, 1e-5, atol=1e-8); worst[k] = w
            if not ok: fails.append(f"{k} {d[k]:.8g} vs {v:.8g}")
    if "summary" not in skip:
        a = [l for l in r0.summary().as_text().splitlines() if not l.startswith(("Date", "Time"))]
        b = [l for l in r1.summary().as_text().splitlines() if not l.startswith(("Date", "Time"))]
        b = [l for l in b if not l.startswith("[") or not "Computed on the GPU" in l]
        bad = [i for i, (x, z) in enumerate(zip(a, b)) if not same_line(x, z)]
        if bad or len(a) != len(b):
            fails.append(f"summary text differs at lines {bad[:3]} ({len(a)} vs {len(b)} lines)")
            if VERBOSE:
                for i in bad[:3]: print("   sm:", repr(a[i])); print("   ms:", repr(b[i]))
    status = "PASS" if not fails else "FAIL"
    results.append((name, status))
    top = ", ".join(f"{k} {v:.0e}" for k, v in sorted(worst.items(), key=lambda kv: -kv[1])[:3])
    print(f"{status}  {name:<44} passes={r1._passes}  worst: {top}" + ("" if not fails else "\n      " + "\n      ".join(fails)), flush=True)
    return r0, r1


import re
_num = re.compile(r"[-+]?(?:\d+\.?\d*(?:e[-+]?\d+)?|inf|nan)")


def same_line(x, z):
    """Identical labels and layout; numbers equal, or within a unit in the last printed digit, or both 'singular'."""
    if x == z or ("smallest eigenvalue is" in x and "smallest eigenvalue is" in z):
        return True                                    # statsmodels prints its roundoff (~1e-27); ours is exactly 0
    squash = lambda t: re.sub(r"\s+", " ", _num.sub("#", t))
    if squash(x) != squash(z) or (len(x) != len(z) and "smallest eigenvalue" not in x):
        return False
    for a, b in zip(_num.findall(x), _num.findall(z)):
        fa, fb = float(a), float(b)
        if fa == fb or (abs(fa) > 1e12 and fb == np.inf) or (abs(fa) < 1e-10 and "smallest eigenvalue" in x):
            continue
        digits = a.split("e")[0]
        ulp = 10.0 ** (-(len(digits.split(".")[1]) if "." in digits else 0)) * (10.0 ** int(a.split("e")[1]) if "e" in a else 1)
        if abs(fa - fb) > 1.01 * ulp:
            return False
    return True


def data(n, k, const=True, scale=None, shift=None, noise=1.0, seed=None):
    g = np.random.default_rng(seed) if seed is not None else rng
    X = g.standard_normal((n, k))
    if scale is not None: X = X * scale
    if shift is not None: X = X + shift
    beta = g.uniform(-2, 2, k)
    y = X @ beta + noise * g.standard_normal(n) + (3.0 if const else 0.0)
    X = f32(X); y = f32(y)
    return (sm.add_constant(X) if const else X), y


# 1-3: plain models
X, y = data(2000, 5)
df = pd.DataFrame(X[:, 1:], columns=[f"feat_{i}" for i in range(5)]); df.insert(0, "const", 1.0); ys = pd.Series(y, name="price")
compare("DataFrame, const first", ys, df)
compare("ndarray, const first", y, X)
compare("ndarray, const last", y, np.column_stack([X[:, 1:], X[:, :1]]))
Xn, yn = data(2000, 4, const=False)
compare("no constant (uncentered R^2)", yn, Xn)
compare("hasconst=False with a const column", y, X, sm_kw={"hasconst": False}, ms_kw={"hasconst": False})

# 4-8: constants and rank
cat = rng.integers(0, 3, 3000); D = np.eye(3)[cat]; x = f32(rng.standard_normal(3000))
yd = f32(D @ [1.0, 2.0, -1.0] + 0.5 * x + rng.standard_normal(3000))
compare("implicit constant (all dummies, no const)", yd, np.column_stack([D, x]))
compare("dummy trap: const + all dummies", yd, sm.add_constant(np.column_stack([D, x])))
compare("duplicate column", y, np.column_stack([X, X[:, 2]]))
compare("zero column + const", y, np.column_stack([X, np.zeros(len(y))]))
compare("two constant columns (1 and 2)", y, np.column_stack([X, 2 * np.ones(len(y))]))

# 9-11: conditioning
Xl, yl = data(5000, 3, shift=np.array([1e4, -3e3, 500.0]))
compare("large column means", f32(yl + 1e5), Xl)
Xc = X.copy(); Xc[:, 2] = f32(Xc[:, 1] + 0.01 * rng.standard_normal(len(y)))
yc = f32(Xc @ rng.uniform(-1, 1, Xc.shape[1]) + rng.standard_normal(len(y)))
compare("collinear pair (corr 0.99995)", yc, Xc, rtol_se=1e-4)
Xp, yp = data(4000, 4, noise=1e-5)
compare("near-exact fit (R^2 ~ 1 - 1e-11)", yp, Xp, rtol_se=1e-4)

# 12-15: weights and robust covariance
w = f32(rng.uniform(0.2, 5.0, len(y)))
compare("WLS", y, X, sm_kw={"weights": w}, ms_kw={"weights": w}, model="WLS")
compare("OLS HC0", y, X, fit_kw={"cov_type": "HC0"})
compare("OLS HC1", y, X, fit_kw={"cov_type": "HC1"})
compare("WLS HC1", y, X, sm_kw={"weights": w}, ms_kw={"weights": w}, model="WLS", fit_kw={"cov_type": "HC1"})

# 16-17: small n
Xs, ys_ = data(12, 2)
compare("n = 12", ys_, Xs)
Xs, ys_ = data(6, 1)
compare("n = 6 (omnibus undefined)", ys_, Xs, skip=("summary",))

# 18-21: input types and blocks
compare("float32 ndarray input", y, X, ms_X=X.astype(np.float32), ms_y=y.astype(np.float32))
compare("MLX array input", y, X, ms_X=mx.array(X.astype(np.float32)), ms_y=mx.array(y.astype(np.float32)))
compare("integer exog", f32(y), np.column_stack([np.ones(len(y), dtype=np.int64), rng.integers(0, 50, (len(y), 3))]))
Xb, yb = data(3_000_000, 20, seed=7)
r0, r1 = compare("3M rows x 20 (several blocks)", yb, Xb, skip=("summary",))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    r2 = ms.OLS(yb, Xb, resident=False).fit()
same = np.array_equal(r1.params, r2.params) and r1.ssr == r2.ssr
print(f"{'PASS' if same else 'FAIL'}  {'streaming (not resident) == resident':<44} identical params and SSR: {same}")
results.append(("streaming", "PASS" if same else "FAIL"))

# 22-25: the virtual constant, residuals, predictions, the scikit-learn API
compare("add_constant=True (no copy of X) vs sm.add_constant", y, X, ms_X=X[:, 1:], ms_kw={"add_constant": True})
compare("WLS add_constant=True", y, X, sm_kw={"weights": w}, ms_kw={"weights": w, "add_constant": True}, model="WLS", ms_X=X[:, 1:])
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    r0 = sm.WLS(y, X, weights=w).fit(); r1 = ms.WLS(y, X, weights=w).fit()
    Xnew = f32(rng.standard_normal((50, X.shape[1])))
    checks = {"resid": close(r1.resid, r0.resid, 0, atol=1e-6 * np.abs(r0.resid).max()),
              "wresid": close(r1.wresid, r0.wresid, 0, atol=1e-6 * np.abs(r0.wresid).max()),
              "fittedvalues": close(r1.fittedvalues, r0.fittedvalues, 1e-9, atol=1e-6 * np.abs(r0.resid).max()),
              "predict": close(r1.predict(Xnew), r0.predict(Xnew), 1e-9, atol=1e-7 * np.abs(r0.predict(Xnew)).max())}
ok = all(v[0] for v in checks.values())
print(f"{'PASS' if ok else 'FAIL'}  {'WLS resid / wresid / fittedvalues / predict':<44} worst: " + ", ".join(f"{k} {v[1]:.0e}" for k, v in checks.items()))
results.append(("resid", "PASS" if ok else "FAIL"))
from sklearn.linear_model import LinearRegression as SkLR
for fi, sw in [(True, None), (False, None), (True, w)]:
    a = SkLR(fit_intercept=fi).fit(X[:, 1:], y, sample_weight=sw); b = ms.LinearRegression(fit_intercept=fi).fit(X[:, 1:], y, sample_weight=sw)
    se = np.asarray(ms.OLS(y, X[:, 1:], add_constant=fi).fit().bse)[1 if fi else 0:] if sw is None else np.ones(X.shape[1] - 1)
    okc = np.all(np.abs(b.coef_ - a.coef_) <= 1e-7 * np.abs(a.coef_) + 1e-6 * se)
    oki = abs(b.intercept_ - a.intercept_) <= 1e-6 * max(1, abs(a.intercept_))
    oks = abs(b.score(X[:, 1:], y) - a.score(X[:, 1:], y)) <= 1e-7
    ok = okc and oki and oks
    tag = f"LinearRegression fit_intercept={fi}" + (" weighted" if sw is not None else "")
    print(f"{'PASS' if ok else 'FAIL'}  {tag:<44} coef rel {np.max(np.abs(b.coef_ - a.coef_) / np.abs(a.coef_)):.0e}, intercept {abs(b.intercept_ - a.intercept_):.0e}, score {abs(b.score(X[:, 1:], y) - a.score(X[:, 1:], y)):.0e}")
    results.append((tag, "PASS" if ok else "FAIL"))

# ---- regressions found by the independent review (2026-09-28): each was a reproduced defect -----------------------
def check(name, ok, detail=""):
    results.append((name, "PASS" if ok else "FAIL"))
    print(f"{'PASS' if ok else 'FAIL'}  {name:<44} {detail}", flush=True)


def raises(f, exc=ValueError, text=None):
    try:
        f()
    except exc as e:
        return text is None or text in str(e)
    return False


def params_close(r1, r0, rtol=1e-7):
    a, b = np.asarray(r1.params, float), np.asarray(r0.params, float)
    return bool(np.all(np.abs(a - b) <= rtol * np.abs(b) + 1e-6 * np.nan_to_num(np.asarray(r0.bse, float)))), float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    g = np.random.default_rng(99)
    # tiny inputs (MLX binds < 8 elements in constant memory) and a one-row last block
    for n_, p_ in [(5, 1), (3, 2), (1, 1)]:
        Xt = f32(g.standard_normal((n_, p_))); yt = f32(g.standard_normal(n_))
        ok, w_ = params_close(ms.OLS(yt, Xt).fit(), sm.OLS(yt, Xt).fit())
        check(f"tiny input n={n_} p={p_}", ok, f"params rel {w_:.0e}")
    from mlx_stats import _data as _d
    old_bb = _d.BLOCK_BYTES; _d.BLOCK_BYTES = 4096
    Xt = f32(g.standard_normal((5122, 3))); yt = f32(Xt @ [1, 2, 3] + g.standard_normal(5122))
    ok, w_ = params_close(ms.OLS(yt, Xt).fit(), sm.OLS(yt, Xt).fit()); _d.BLOCK_BYTES = old_bb
    check("last block of one row", ok, f"params rel {w_:.0e}")
    yt = f32(g.standard_normal(200)); r1 = ms.OLS(yt, np.empty((200, 0)), add_constant=True).fit()
    check("intercept-only model (empty X + add_constant)", abs(r1.params_arr[0] - yt.mean()) < 1e-12)
    # wide designs: threadgroup memory
    Xt = f32(g.standard_normal((3000, 230))); yt = f32(Xt[:, :5].sum(1) + g.standard_normal(3000))
    ok, w_ = params_close(ms.OLS(yt, Xt).fit(), sm.OLS(yt, Xt).fit()); check("230 columns", ok, f"params rel {w_:.0e}")
    # saturated: df_resid = 0 gives statsmodels' inf / nan, not ZeroDivisionError
    Xt = f32(g.standard_normal((4, 4))); yt = f32(g.standard_normal(4)); r1 = ms.OLS(yt, Xt).fit()
    check("df_resid = 0", np.all(np.isinf(np.asarray(r1.bse))) and np.isnan(r1.mse_resid) and len(str(r1.summary())) > 0)
    # WLS: constants and implicit constants decided on the unweighted data; a column live only on zero-weight rows
    n_ = 4000; Z = f32(g.standard_normal((n_, 2))); yt = f32(Z @ [1.0, -1.0] + 3.0 + g.standard_normal(n_))
    wt = f32(g.uniform(0.5, 2.0, n_)); wt[:50] = 0.0; dm = np.zeros(n_); dm[:50] = 1.0
    for tag, Xt in [("no const", np.column_stack([Z, dm])), ("const", sm.add_constant(np.column_stack([Z, dm])))]:
        r0 = sm.WLS(yt, Xt, weights=wt).fit(); r1 = ms.WLS(yt, Xt, weights=wt).fit()
        ok = r1.rank == r0.model.rank and r1.k_constant == r0.k_constant and abs(r1.ssr - r0.ssr) < 1e-8 * r0.ssr and abs(r1.params_arr[-1]) < 1e-8
        check(f"WLS column live only on zero-weight rows, {tag}", ok, f"rank {r1.rank}/{r0.model.rank} k_const {r1.k_constant}/{r0.k_constant}")
    xc = Z[:, 0].copy(); xc[50:] = 1.0                                  # constant only where the weight is positive
    Xt = np.column_stack([Z[:, 1], xc]); r0 = sm.WLS(yt, Xt, weights=wt).fit(); r1 = ms.WLS(yt, Xt, weights=wt).fit()
    check("WLS constant only on positive weights", r1.k_constant == r0.k_constant and abs(r1.rsquared - r0.rsquared) < 1e-9)
    # an exact linear relation under weights (the exact Gram must not break it)
    Zi = g.integers(-1000, 1000, (3000, 2)).astype(float); wt = f32(10 ** g.uniform(-3, 3, 3000))
    Xt = np.column_stack([Zi, Zi[:, 0] + Zi[:, 1]]); yt = f32(Xt[:, :2] @ [1.0, 0.001] + g.standard_normal(3000))
    r0 = sm.WLS(yt, Xt, weights=wt).fit(); r1 = ms.WLS(yt, Xt, weights=wt).fit()
    check("WLS exact relation x3 = x1 + x2 (no const)", r1.rank == 2 and np.max(np.abs(r1.params_arr - np.asarray(r0.params))) < 1e-6)
    # units far apart: the pseudo-inverse must not depend on column units
    Xt = f32(np.column_stack([g.standard_normal(5000) * 1e-8, g.standard_normal(5000), g.standard_normal(5000) * 1e8]))
    yt = f32(Xt @ [1e8, 1.0, 1e-8] + g.standard_normal(5000))
    ref = np.linalg.lstsq(Xt / Xt.std(0), yt, rcond=None)[0] / Xt.std(0)     # statsmodels truncates the 1e-8 column here
    b_ = ms.OLS(yt, Xt).fit().params_arr; w_ = float(np.max(np.abs(b_ - ref) / np.abs(ref)))
    check("columns in units 1e-8 / 1 / 1e8 (vs exact)", w_ < 1e-7, f"params rel {w_:.0e}")
    # float32 overflow / underflow of sums of squares: power-of-two scaling
    for sc in (1e17, 1e-20):
        Xt = f32(g.standard_normal((20000, 2)) * sc); yt = f32(Xt @ [1 / sc, 2 / sc] + g.standard_normal(20000))
        ok, w_ = params_close(ms.OLS(yt, Xt).fit(), sm.OLS(yt, Xt).fit(), rtol=1e-6); check(f"column scale {sc:g}", ok, f"params rel {w_:.0e}")
    yt2 = f32(g.standard_normal(20000) * 1e10 + 5); Xt = sm.add_constant(f32(g.standard_normal((20000, 2))))
    r1 = ms.OLS(yt2, Xt).fit(); r0 = sm.OLS(yt2, Xt).fit()
    from statsmodels.stats.stattools import jarque_bera
    check("residual sd 1e10: moments do not overflow", abs(r1.diagn["kurtosis"] - jarque_bera(r0.wresid)[3]) < 1e-6)
    # refine=False on a tight fit: the Gram SSR is replaced by the exact one
    Xt = sm.add_constant(f32(g.standard_normal((20000, 3)))); yt = f32(Xt @ [1, 2, 3, 4] + 1e-4 * g.standard_normal(20000))
    r1 = ms.OLS(yt, Xt).fit(refine=False); r0 = sm.OLS(yt, Xt).fit()
    check("refine=False on R^2 = 1 - 3e-9", abs(r1.ssr - r0.ssr) < 1e-6 * r0.ssr, f"ssr {r1.ssr:.6e} vs {r0.ssr:.6e}")
    # input handling
    Xt = f32(g.standard_normal((300, 2))); yt = f32(Xt @ [1, 2] + g.standard_normal(300))
    check("negative weights raise", raises(lambda: ms.WLS(yt, Xt, weights=np.r_[-1.0, np.ones(299)]).fit()))
    check("all-zero weights raise", raises(lambda: ms.WLS(yt, Xt, weights=np.zeros(300)).fit()))
    dfi = pd.DataFrame(Xt); r1 = ms.OLS(pd.Series(yt), dfi, add_constant=True).fit()
    check("integer column labels kept", list(r1.params.index) == ["const", 0, 1] and abs(r1.params[1] - r1.params_arr[2]) == 0)
    Xc = sm.add_constant(Xt); r1 = ms.OLS(yt, Xc, add_constant=True).fit()
    check("add_constant=True skips an existing constant", len(r1.params_arr) == 3 and ms.add_constant(Xc).shape == (300, 3))
    dfb = pd.DataFrame({"a": Xt[:, 0], "flag": Xt[:, 1] > 0, "k": pd.array(np.arange(300) % 5, dtype="Int64")})
    r0 = sm.OLS(yt, sm.add_constant(dfb.astype(float))).fit(); r1 = ms.OLS(yt, dfb, add_constant=True).fit()
    ok, w_ = params_close(r1, r0); check("DataFrame with bool and Int64 columns", ok, f"params rel {w_:.0e}")
    ys = pd.Series(yt, index=np.arange(300)[::-1])
    check("misaligned pandas indices raise", raises(lambda: ms.OLS(ys, pd.DataFrame(Xt)), text="not aligned"))
    check("MLX endog of shape (n, 2) raises", raises(lambda: ms.OLS(mx.array(np.stack([yt, yt], 1).astype(np.float32)), Xt), text="1-D"))
    from sklearn.linear_model import LinearRegression as SkLR
    Xr = np.column_stack([Xt, np.full(300, 5.0)]); a = SkLR().fit(Xr, yt); b = ms.LinearRegression().fit(Xr, yt)
    check("LinearRegression rank-deficient == scikit-learn", np.allclose(b.coef_, a.coef_, atol=1e-6) and abs(b.intercept_ - a.intercept_) < 1e-6)
    dfn = pd.DataFrame(Xt, columns=["a", "b"]); lr = ms.LinearRegression().fit(dfn, yt)
    check("LinearRegression feature-name mismatch raises", raises(lambda: lr.predict(dfn[["b", "a"]]), text="feature names"))

# ---- the formula interface and the command line ---------------------------------------------------------------------
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statsmodels.formula.api as smf
    g = np.random.default_rng(7); n_ = 5000
    dff = pd.DataFrame({"price": f32(g.standard_normal(n_)), "size": f32(g.uniform(1, 5, n_)), "age": g.integers(0, 50, n_).astype(float),
                        "region": g.choice(["n", "s", "w"], n_), "kind": g.choice(["a", "b"], n_)})
    dff.loc[[3, 17], "size"] = np.nan; dff.loc[40, "region"] = None
    for fml in ["price ~ size + age + C(region)", "price ~ C(region) * C(kind) + size", "price ~ size + age - 1",
                "price ~ 0 + C(region) + size", "price ~ size:C(region) + age"]:
        r0 = smf.ols(fml, dff).fit(); r1 = ms.ols(fml, dff).fit()
        ok = list(r0.params.index) == list(r1.params.index) and r1.nobs == r0.nobs
        d_ = float(np.max(np.abs(r1.params.to_numpy() - r0.params.to_numpy()) / np.maximum(np.abs(r0.params.to_numpy()), 1e-12)))
        check(f"formula {fml}", ok and d_ < 1e-7, f"names/order/nobs equal {ok}, params rel {d_:.0e}")
    wf = g.uniform(0.5, 2, n_)
    r0 = smf.wls("price ~ size + C(region)", dff, weights=wf).fit(); r1 = ms.wls("price ~ size + C(region)", dff, weights=wf).fit()
    d_ = float(np.max(np.abs(r1.params.to_numpy() - r0.params.to_numpy()) / np.abs(r0.params.to_numpy())))
    check("formula WLS with missing rows dropped", r1.nobs == r0.nobs and d_ < 1e-7, f"params rel {d_:.0e}")
    import subprocess, tempfile, os
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "d.csv"); dff.to_csv(path, index=False)
        out = subprocess.run([sys.executable, "-m", "mlx_stats.cli", "fit", path, "price ~ size + C(region)"], capture_output=True, text=True)
        check("command line: mlx-stats fit file.csv formula", out.returncode == 0 and "C(region)[T.s]" in out.stdout and "rows used" in out.stdout,
              out.stderr.strip().splitlines()[-1] if out.returncode else "")
        out = subprocess.run([sys.executable, "-m", "mlx_stats.cli", "demo", "--rows", "20000"], capture_output=True, text=True)
        check("command line: mlx-stats demo", out.returncode == 0 and "OLS Regression Results" in out.stdout)

# bad input
Xbad = X.copy(); Xbad[5, 2] = np.nan
try:
    ms.OLS(y, Xbad).fit(); ok = False
except ValueError:
    ok = True
print(f"{'PASS' if ok else 'FAIL'}  {'NaN in exog raises ValueError':<44}")
results.append(("nan", "PASS" if ok else "FAIL"))

nf = sum(s == "FAIL" for _, s in results)
print(f"\n{len(results) - nf}/{len(results)} passed")
sys.exit(1 if nf else 0)
