#!/usr/bin/env python3
"""Linear regression against the standard libraries on this machine, from NumPy float64 input (what a DataFrame
holds). Usage: bench_ols.py n p [n p ...]   e.g. bench_ols.py 1_000_000 10 10_000_000 10 10_000_000 50"""
import sys, time, gc, warnings, numpy as np, mlx.core as mx
sys.path.insert(0, __file__.rsplit("/bench/", 1)[0])
import mlx_stats as ms
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression as SkLR
warnings.simplefilter("ignore")


def best(f, r):
    ts = []
    for _ in range(r):
        gc.collect(); t = time.perf_counter(); out = f(); ts.append(time.perf_counter() - t)
    return out, min(ts)


args = [int(a.replace("_", "")) for a in sys.argv[1:]] or [1_000_000, 10]
for n, p in zip(args[::2], args[1::2]):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, p)); X[:, 1::3] = X[:, 1::3] * 5 + 20
    Xc = np.column_stack([np.ones(n), X])
    y = Xc @ rng.uniform(-1, 1, p + 1) + rng.standard_normal(n)
    gb64 = Xc.nbytes / 1e9 + y.nbytes / 1e9
    heavy = n * p > 3e8
    print(f"\n== n = {n:,}, p = {p} (+ const): {gb64:.2f} GB of float64 input ==", flush=True)
    rows = []
    # --- ours ---
    ms.OLS(y[:1000], Xc[:1000]).fit()                    # compile kernels outside the timing
    ms.LinearRegression().fit(X[:1000], y[:1000]); ms.OLS(y[:1000], Xc[:1000]).fit(precise=True)
    gc.collect(); t0 = time.perf_counter(); ms.OLS(y, Xc).fit(); t_first = time.perf_counter() - t0
    r_ms, t = best(lambda: ms.OLS(y, Xc).fit(), 3); s = r_ms.summary()
    rows.append(("mlx-stats OLS.fit(), first call on this data (fresh MLX memory)", t_first, r_ms.params_arr, "cold"))
    rows.append(("mlx-stats OLS.fit(): statsmodels results + diagnostics", t, r_ms.params_arr, f"{r_ms._passes} passes"))
    r_g, t = best(lambda: ms.OLS(y, Xc).fit(refine=False), 3)
    rows.append(("mlx-stats OLS.fit(refine=False): one pass, no residual diagnostics", t, r_g.params_arr, "1 pass"))
    lr, t = best(lambda: ms.LinearRegression().fit(X, y), 3)
    rows.append(("mlx-stats LinearRegression (scikit-learn style)", t, np.concatenate([[lr.intercept_], lr.coef_]), ""))
    import pandas as pd
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(p)]); ys = pd.Series(y, name="y")
    r_df, t = best(lambda: ms.OLS(ys, df, add_constant=True).fit(), 3)
    rows.append(("mlx-stats OLS(y, DataFrame, add_constant=True).fit()", t, np.asarray(r_df.params), "pandas"))
    del df
    Xm, ym = mx.array(Xc.astype(np.float32)), mx.array(y.astype(np.float32)); mx.eval(Xm, ym)
    r_m, t = best(lambda: ms.OLS(ym, Xm).fit(), 3)
    rows.append(("mlx-stats OLS.fit() on data already in MLX (float32)", t, r_m.params_arr, "resident"))
    del Xm, ym
    # --- the standard libraries ---
    r_sm, t = best(lambda: sm.OLS(y, Xc).fit(), 1 if heavy else 2)
    rows.append(("statsmodels OLS.fit()", t, np.asarray(r_sm.params), ""))
    _, t2 = best(lambda: sm.OLS(y, Xc).fit().summary(), 1)
    rows.append(("statsmodels OLS.fit().summary()", t2, np.asarray(r_sm.params), "diagnostics"))
    sk, t = best(lambda: SkLR().fit(X, y), 1 if heavy else 2)
    rows.append(("scikit-learn LinearRegression", t, np.concatenate([[sk.intercept_], sk.coef_]), ""))
    b, t = best(lambda: np.linalg.solve(Xc.T @ Xc, Xc.T @ y), 3)
    rows.append(("NumPy normal equations (Accelerate X^T X, float64)", t, b, "no inference"))
    b2, t = best(lambda: np.linalg.lstsq(Xc, y, rcond=None)[0], 1 if heavy else 2)
    rows.append(("NumPy lstsq", t, b2, "no inference"))
    ref = np.asarray(r_sm.params)
    tref = [t for name, t, _, _ in rows if name == "statsmodels OLS.fit()"][0]
    for name, t, b, note in rows:
        err = np.max(np.abs(b - ref) / np.maximum(np.abs(ref), 1e-12))
        print(f"  {name:<68} {t:8.3f} s  {tref / t:7.1f}x statsmodels   max rel diff vs statsmodels {err:.1e}  {note}", flush=True)
    print(f"  floor: copy of the float64 input into MLX at ~117 GB/s {gb64 / 117:.3f} s; one GPU read of it as float32 at ~500 GB/s {gb64 / 2 / 500:.4f} s", flush=True)
    del X, Xc, y
