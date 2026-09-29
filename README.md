# mlx-stats

Linear regression on the Apple GPU through MLX, with statsmodels' results: parameters, standard errors, t and p values,
confidence intervals, R², F, log-likelihood, AIC/BIC, omnibus, Jarque-Bera, skew, kurtosis, Durbin-Watson, condition
number, and HC0/HC1 robust covariance. Plus a scikit-learn style `LinearRegression`.

```python
import mlx_stats as ms

res = ms.OLS(y, X, add_constant=True).fit()      # or ms.OLS(y, ms.add_constant(X)), statsmodels style
print(res.summary())                             # the same text statsmodels prints
res.params, res.bse, res.pvalues, res.conf_int(), res.rsquared, res.fvalue

ms.WLS(y, X, weights=w).fit(cov_type="HC1")
ms.LinearRegression().fit(X, y).coef_
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
| 1M x 10 | 5 ms | 155 ms | 62 ms | 4 ms |
| 10M x 10 | 21 ms | 1.41 s | 639 ms | 33 ms |
| 10M x 50 | 67 ms | 10.1 s | 5.49 s | 189 ms |
| 100M x 10 | 144 ms | 25.2 s | 7.53 s | 331 ms |

With the data already in MLX memory the full fit of 100M x 10 takes 31 ms. On the same float32-valued data the
parameters match statsmodels to 1e-9 to 1e-14 relative and the summary text is identical (`tests/test_vs_statsmodels.py`,
31 cases).

## Precision

The GPU works in float32, so float64 inputs are rounded to float32 first; everything after that is exact or float64
(two-float accumulation, float64 statistics). Designs whose scaled columns are nearly collinear switch to the exact Gram
automatically. A warning is raised when the design is too collinear for float32 data to resolve (smallest scaled
eigenvalue below 1e-10), and rank-deficient designs get statsmodels' minimum-norm solution.

## Limits

- Up to 255 columns (the residual pass); X^T X slows past ~100 columns, where the multiply rather than the read is the limit.
- No formula interface, missing-data handling (`missing='drop'`), HC2/HC3, cluster or HAC covariance yet.
- Apple silicon with MLX only.

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install mlx numpy scipy statsmodels scikit-learn pandas
.venv/bin/python tests/test_vs_statsmodels.py
.venv/bin/python bench/bench_ols.py 1_000_000 10 10_000_000 10 10_000_000 50 100_000_000 10
```
