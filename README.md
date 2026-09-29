# mlx-stats

Linear regression on the GPU of an Apple Silicon Mac, with statsmodels' results: parameters, standard errors, t and p
values, confidence intervals, R², F, log-likelihood, AIC/BIC, omnibus, Jarque-Bera, skew, kurtosis, Durbin-Watson,
condition number, and HC0/HC1 robust covariance. Plus a scikit-learn style `LinearRegression`.

**Requires an Apple Silicon Mac (M1 or later).** The fit runs on the Mac's GPU through Apple's MLX and Metal; it does not
run on Windows, Linux or Intel Macs.

## Quick start

```bash
pip install git+https://github.com/kakollu/mlx-stats
```

The install refuses to build anywhere but an Apple Silicon Mac with a native (arm64) Python.

```python
import pandas as pd
import mlx_stats as ms

df = pd.read_csv("houses.csv")
res = ms.ols("price ~ size + age + C(region)", data=df).fit()
print(res.summary())                 # the same table statsmodels prints
res.params, res.pvalues, res.conf_int()
```

Formulas work as in `statsmodels.formula.api`: categorical terms `C(x)`, interactions `a * b` and `a:b`, transforms
`np.log(x)` and `I(x ** 2)`, no intercept with `- 1`, and rows with missing values dropped. Parameter names and order
match statsmodels. Weighted least squares and robust standard errors:

```python
ms.wls("price ~ size + age", data=df, weights="w").fit(cov_type="HC1")
```

From the command line, without writing Python:

```bash
mlx-stats fit houses.csv "price ~ size + age + C(region)"
```

To see it run on synthetic data (5 million rows, timed, and compared with statsmodels if it is installed):

```bash
mlx-stats demo
```

### Without formulas

```python
res = ms.OLS(y, X, add_constant=True).fit()      # arrays or DataFrames, statsmodels argument order
ms.WLS(y, X, weights=w).fit(cov_type="HC1")
ms.LinearRegression().fit(X, y).coef_            # scikit-learn style
```

X and y can be NumPy arrays (float64 or float32), pandas objects or MLX arrays.

## How it works

A linear model needs X^T X, X^T y and y^T y — one read of the data — and then a p x p solve. On unified memory the GPU
reads the data at ~500 GB/s, so the fit costs about as much as getting the data into GPU memory.

1. **One pass for X^T X** (`_gram.py`). A Metal kernel multiplies 8 x 8 tiles on the GPU's matrix unit and folds its
   float32 partial sums into two-float (hi, lo) accumulators every 16-128 rows, so each entry is accurate to float32
   rounding (~3e-8 of its scale) however many rows there are. Columns are shifted by a value near their mean, so a large
   mean does not cost precision.
2. **A small solve in float64** on the CPU, in a basis where the shifted columns plus a constant are well conditioned:
   statsmodels' minimum-norm parameters, rank, normalized covariance and condition number.
3. **One pass for the residuals** (`_resid.py`) with the coefficients in two-float precision: the exact sum of squared
   residuals, the residual moments behind the normality tests, Durbin-Watson, and an exact X^T r that refines the
   parameters to what a float64 solver gives on the same float32 data.
4. **An exact X^T X** (`_gram_exact.py`, every product formed exactly) replaces step 1 when the design is collinear.

Input from NumPy float64 is copied into MLX memory as raw bits on eight threads and rounded to float32 on the GPU
(bit-exact with NumPy's cast); DataFrames are transposed on the GPU.

## Results

On an M5 Max, from NumPy float64 input, against `statsmodels.OLS(...).fit()` (full details in `bench/RESULTS.md`):

| shape | mlx-stats full fit | statsmodels | scikit-learn | NumPy normal equations (no inference) |
|---|---:|---:|---:|---:|
| 1M x 10 | 5 ms | 160 ms | 63 ms | 4 ms |
| 10M x 10 | 20 ms | 1.49 s | 630 ms | 33 ms |
| 10M x 50 | 68 ms | 9.97 s | 5.40 s | 187 ms |
| 100M x 10 | 144 ms | 26.4 s | 8.33 s | 332 ms |

With the data already in MLX memory the full fit of 100M x 10 takes 30 ms. On the same float32-valued data the
parameters match statsmodels to 1e-9 to 1e-14 relative and the summary text is identical (`tests/test_vs_statsmodels.py`,
56 cases, including 25 regression tests from an independent review).

## Precision

The GPU works in float32, so float64 inputs are rounded to float32 first (values beyond ~3e38 cannot be represented);
everything after that is exact or float64 (two-float accumulation, float64 statistics). Columns whose spread is far
from 1 are scaled by powers of two, exactly, so units from 1e-20 to 1e17 fit the same way. Designs whose scaled columns
are nearly collinear switch to the exact Gram automatically, which keeps standard errors within ~1e-9 of a float64 fit.
A warning is raised when the design is too collinear for float32 data to resolve (squared singular-value ratio of the
scaled design below 1e-10). Weights are used as given (statsmodels' convention); constants are detected on the
unweighted data, as statsmodels does.

Rank-deficient designs get statsmodels' minimum-norm parameters. When such a design also has column means around
1e5-1e6, the non-identified parameter components can differ from statsmodels' by ~1e-4 relative (more with extreme
weights); fitted values, residuals and every fit statistic still agree.

## Limits

- Up to 255 columns (the residual pass); X^T X slows past ~100 columns, where the multiply rather than the read is the limit.
- Missing values are dropped through the formula interface only (`OLS`/`WLS` on arrays reject NaN); no HC2/HC3,
  cluster or HAC covariance yet.
- Apple silicon with MLX only.

## Install from source (for development)

Apple Silicon Mac, Python 3.11 or later:

```bash
git clone https://github.com/kakollu/mlx-stats && cd mlx-stats
python3 -m venv .venv && .venv/bin/pip install -e ".[test]"
```

Runtime dependencies are `mlx`, `numpy`, `scipy`, `pandas` and `formulaic`. The `files` extra adds Parquet and Excel
reading for the command line; the `test` extra adds statsmodels and scikit-learn for the comparison suite. Tested on an M5 Max with
Python 3.11 and 3.14 and MLX 0.32.2-0.32.3; other Apple Silicon chips have not been tested yet.

## Reproduce

```bash
.venv/bin/python tests/test_vs_statsmodels.py
.venv/bin/python bench/bench_ols.py 1_000_000 10 10_000_000 10 10_000_000 50 100_000_000 10
```

## License

MIT, see `LICENSE`.
