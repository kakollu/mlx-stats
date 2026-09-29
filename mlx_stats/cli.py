"""Command line: fit a model from a data file, or run a demo.

    mlx-stats fit sales.csv "price ~ size + age + C(region)"
    mlx-stats fit sales.parquet "price ~ size" --weights w --cov HC1
    mlx-stats demo                     # a 5M-row fit on synthetic data, timed (and against statsmodels if installed)
"""
import argparse
import sys
import time


def _read(path, sheet=None):
    import pandas as pd
    p = path.lower()
    try:
        if p.endswith((".csv", ".txt")):
            return pd.read_csv(path)
        if p.endswith((".tsv", ".tab")):
            return pd.read_csv(path, sep="\t")
        if p.endswith((".parquet", ".pq")):
            return pd.read_parquet(path)
        if p.endswith((".xlsx", ".xls")):
            return pd.read_excel(path, sheet_name=sheet or 0)
        if p.endswith(".feather"):
            return pd.read_feather(path)
    except ImportError as e:
        sys.exit(f"reading {path} needs an extra package: {e}. Try: pip install 'mlx-stats[files]'")
    sys.exit(f"unsupported file type: {path} (use .csv, .tsv, .parquet, .xlsx or .feather)")


def _fit(args):
    import mlx_stats as ms
    df = _read(args.file, args.sheet)
    t = time.perf_counter()
    model = ms.wls(args.formula, df, weights=args.weights) if args.weights else ms.ols(args.formula, df)
    res = model.fit(cov_type=args.cov)
    print(res.summary())
    print(f"\n{int(res.nobs):,} rows used of {len(df):,}; fitted in {time.perf_counter() - t:.3f} s")


def _demo(args):
    import numpy as np
    import pandas as pd
    import mlx_stats as ms
    n, p = args.rows, args.cols
    print(f"Synthetic data: {n:,} rows x {p} columns (float64), y = 1 + X b + noise ...", flush=True)
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.standard_normal((n, p)), columns=[f"x{i + 1}" for i in range(p)])
    df["y"] = 1.0 + df.to_numpy() @ np.linspace(-1, 1, p) + rng.standard_normal(n)
    formula = "y ~ " + " + ".join(df.columns[:-1])
    ms.ols(formula, df.head(1000)).fit()                       # compile the GPU kernels once
    t = time.perf_counter(); res = ms.ols(formula, df).fit(); t_ms = time.perf_counter() - t
    print(res.summary())
    print(f"\nmlx-stats: {t_ms:.3f} s (formula, design matrix and a full statsmodels-style fit)")
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("statsmodels is not installed, so there is no comparison run (pip install statsmodels to see one).")
        return
    t = time.perf_counter(); r0 = smf.ols(formula, df).fit(); t_sm = time.perf_counter() - t
    d = float(np.max(np.abs(res.params_arr - r0.params.to_numpy()) / np.abs(r0.params.to_numpy())))
    print(f"statsmodels: {t_sm:.3f} s on the same formula and data; largest relative parameter difference {d:.1e}")


def main(argv=None):
    import mlx_stats
    ap = argparse.ArgumentParser(prog="mlx-stats", description="Linear regression on the Apple GPU with statsmodels' results.")
    ap.add_argument("--version", action="version", version=f"mlx-stats {mlx_stats.__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="fit an OLS (or WLS) model from a data file and print the summary")
    f.add_argument("file", help=".csv, .tsv, .parquet, .xlsx or .feather")
    f.add_argument("formula", help='e.g. "price ~ size + age + C(region)"')
    f.add_argument("--weights", help="column with WLS weights (inverse variances)")
    f.add_argument("--cov", default="nonrobust", choices=["nonrobust", "HC0", "HC1"], help="covariance type")
    f.add_argument("--sheet", help="Excel sheet name")
    f.set_defaults(func=_fit)
    d = sub.add_parser("demo", help="fit a model to synthetic data and time it")
    d.add_argument("--rows", type=int, default=5_000_000)
    d.add_argument("--cols", type=int, default=10)
    d.set_defaults(func=_demo)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
