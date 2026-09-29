# Results

Last updated: 2026-09-28. Apple M5 Max (40 GPU cores, 128 GB), macOS 27, MLX 0.32.2, NumPy 2.4.6, statsmodels 0.15.0,
scikit-learn 1.9.1. Timings are best of 3 unless marked "first call"; the machine had no other heavy load (registered as
`STATS` in the hub's `MACHINE.md`).

## Linear regression against the standard libraries (`bench/bench_ols.py`, log `bench-ols-3.log`)

Input: NumPy float64 X (n x p, a third of the columns with mean 20 and sd 5) plus a constant column, y = X b + noise.
"Full fit" is `mlx_stats.OLS(y, X).fit()`: parameters, standard errors, t/p values, confidence intervals, R^2, F,
log-likelihood, AIC/BIC, omnibus, Jarque-Bera, skew, kurtosis, Durbin-Watson and condition number, i.e. everything in
statsmodels' summary. Speed-ups are against `statsmodels.OLS(y, X).fit()`, which computes the same numbers except the
residual diagnostics (those are computed when `summary()` is called; that row is listed too).

| shape (float64 input) | 1M x 10 | 10M x 10 | 10M x 50 | 100M x 10 |
|---|---:|---:|---:|---:|
| **mlx-stats full fit** | **5 ms** (33x) | **21 ms** (68x) | **67 ms** (152x) | **144 ms** (175x) |
| mlx-stats full fit, first call on the data | 10 ms | 37 ms | 99 ms | 203 ms |
| mlx-stats full fit from a pandas DataFrame | 5 ms | 20 ms | 71 ms | 149 ms |
| mlx-stats full fit, data already in MLX (float32) | 3 ms (58x) | 5 ms (283x) | 17 ms (595x) | 31 ms (807x) |
| mlx-stats one pass (`refine=False`, no residual diagnostics) | 3 ms | 17 ms | 55 ms | 121 ms |
| mlx-stats `LinearRegression` (scikit-learn style) | 5 ms | 17 ms | 55 ms | 113 ms |
| statsmodels `OLS.fit()` | 155 ms | 1.41 s | 10.1 s | 25.2 s |
| statsmodels `OLS.fit().summary()` | 202 ms | 1.96 s | 10.8 s | 22.1 s |
| scikit-learn `LinearRegression` | 62 ms | 639 ms | 5.49 s | 7.53 s |
| NumPy normal equations (Accelerate X^T X, float64; no inference) | 4 ms | 33 ms | 189 ms | 331 ms |
| NumPy `lstsq` (no inference) | 50 ms | 495 ms | 5.02 s | 5.19 s |
| floor: copying the float64 input into MLX memory at ~117 GB/s | 1 ms | 8 ms | 36 ms | 82 ms |

Agreement with statsmodels (largest relative parameter difference): full fit 2.6e-9 / 1.3e-9 / 1.7e-8 / 3.1e-9, one pass
1.3e-7 / 4.3e-8 / 4.7e-8 / 1.9e-8 (the residual pass refines the parameters; on float32-valued data they then agree to
~1e-14, see the tests). The benchmark data are float64 draws, so the GPU sees them rounded to float32; the differences
above are that rounding, not the solver.

Reading it:
- **Against statsmodels and scikit-learn** the gain is 33-175x (full statsmodels-type fit) and grows with size; most of
  it is algorithmic (normal equations + a small p x p solve instead of an SVD or QR of X), the rest is the GPU.
- **Against the same algorithm on the CPU** (NumPy normal equations on Accelerate, which computes no inference) the full
  fit is 1.6x faster at 10M x 10, 2.8x at 10M x 50 and 2.3x at 100M x 10, and the one-pass fit is 1.9-3.4x faster. At 1M
  rows NumPy is as fast or faster: fixed costs (two host round trips per pass, ~0.15 ms each) dominate there.
- **From NumPy input the fit costs about the copy into GPU memory**: at 100M x 10 the one-pass fit is 121 ms against an
  82 ms copy floor and the full fit adds 23 ms for the residual pass on the already-copied data. With the data already in
  MLX memory the full fit is 31 ms for 100M x 10 (two reads of 4.4 GB of float32).
- **First call costs 1.3-2x the warm time**: fresh MLX buffers are page-faulted as they are filled. Warm numbers are the
  steady state of refitting on the same process; a one-off fit on new data pays the first-call number.

## The X^T X kernel (`mlx_stats/_gram.py`, `bench/tune_gram.py`)

Upper-triangle 8x8 tiles on the matrix unit (`simdgroup_multiply_accumulate`), 2x2 blocks of tiles sharing four operand
loads, stages of 16-128 rows staged in threadgroup memory with the next stage prefetched into registers, float32
accumulation folded into a two-float (hi, lo) sum after every stage.

Read rate on 3 GB MLX-resident inputs (X and y), best of 5, by width p, after tuning stage rows / simdgroups / blocks:

| p | 4 | 8 | 16 | 32 | 48 | 64 | 100 | 128 | 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GB/s | 282 | 410 | 386 | 356 | 268 | 293 | 192 | 156 | 85 |

The read floor is ~500 GB/s. Up to ~64 columns the kernel reads at 55-82% of it; past ~100 columns the multiply is the
limit (the matrix unit runs at 30-45% of its 15.7 TFLOP/s ceiling there - each 16x16 block still loads its operands from
threadgroup memory every 8 rows).

Accuracy against a float64 Gram of the same float32 data: 2.3e-8 to 9e-8 of sqrt(G_ii G_jj) per entry at n from 1,000 to
2,000,000 - float32 rounding, not growing with n. The row count comes out exact.

## The exact Gram (`mlx_stats/_gram_exact.py`)

Every product formed exactly (TwoProd) and added into a two-float accumulator: 1e-13 to 1e-11 of sqrt(G_ii G_jj) against
a float64 sum of the same float32 values. It runs on the scalar units at 62-138 GB/s (2.7-4.6x the fast kernel's time)
and is used only when the fast Gram shows a scaled eigenvalue below 1e-4 (collinear designs), or with `fit(precise=True)`.
Before it existed, a pair of columns correlated at 0.99995 put 5e-4 relative error into the standard errors; with it they
match statsmodels to 5e-7.

## Getting data into GPU memory (`mlx_stats/_data.py`, `mlx_stats/_fp64.py`)

| path | rate |
|---|---:|
| `mx.array(numpy float32)` | 68-71 GB/s |
| NumPy `astype(float32)` of float64 | 21 GB/s |
| 8-thread `np.copyto` into a NumPy view of MLX memory | 117 GB/s (1 thread: 57) |
| float64 -> float32 rounding on the GPU (bit-exact with NumPy on 27M values incl. subnormals, ties, inf, NaN) | memory speed |
| GPU read of MLX-resident float32 | 510-528 GB/s |

float64 input is copied as raw bits through a ring of three staging buffers and rounded on the GPU; column-major input
(what `DataFrame.to_numpy()` returns) is copied column by column on eight threads and transposed on the GPU. DataFrame
fits went from 0.913 s to 0.149 s at 100M x 10 with that change (a single-threaded transposing copy per block had been
most of the time).

## Correctness (`tests/test_vs_statsmodels.py`)

31/31 cases match statsmodels (and scikit-learn for `LinearRegression`) on float32-valued data: constant first/last/none,
`hasconst=False`, implicit constant (full dummies), dummy trap, duplicate column, zero column, two constant columns, large
column means, a collinear pair (corr 0.99995), a near-exact fit (R^2 = 1 - 1e-11), WLS, HC0/HC1 (OLS and WLS), n = 12,
n = 6, float32 / MLX / integer / DataFrame input, 3M rows in several blocks, streaming == resident, NaN input, the virtual
constant, residuals / fitted values / predictions, and `LinearRegression` with and without an intercept and with sample
weights. Parameters agree to 1e-9-1e-14 relative, statistics to ~1e-8, and the summary text is identical apart from date,
time and one added note.
