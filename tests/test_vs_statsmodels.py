#!/usr/bin/env python3
"""Every case fits the same float32-valued data with mlx-stats (GPU) and statsmodels (float64, CPU) and compares
every reported quantity. Run: .venv/bin/python tests/test_vs_statsmodels.py [-v]"""
import sys, warnings, numpy as np, pandas as pd, statsmodels.api as sm, mlx.core as mx
sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])
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

# 22: bad input
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
