# Results

Last updated: 2026-09-28. Apple M5 Max (40 GPU cores, 128 GB), macOS 27, MLX 0.32.2 (tests also pass on 0.32.3), NumPy 2.4.6, statsmodels 0.15.0,
scikit-learn 1.9.1. Timings are best of 3 unless marked "first call"; the machine had no other heavy load (registered as
`STATS` in the hub's `MACHINE.md`).

## Linear regression against the standard libraries (`bench/bench_ols.py`, log `bench-ols-4.log`)

Final run on the published code (after the independent review's fixes), 2026-09-28. Input: NumPy float64 X (n x p, a
third of the columns with mean 20 and sd 5) plus a constant column, y = X b + noise. "Full fit" is
`mlx_stats.OLS(y, X).fit()`: parameters, standard errors, t/p values, confidence intervals, R^2, F, log-likelihood,
AIC/BIC, omnibus, Jarque-Bera, skew, kurtosis, Durbin-Watson and condition number, i.e. everything in statsmodels'
summary. Speed-ups are against `statsmodels.OLS(y, X).fit()`, which computes the same numbers except the residual
diagnostics (those are computed when `summary()` is called; that row is listed too).

| shape (float64 input) | 1M x 10 | 10M x 10 | 10M x 50 | 100M x 10 |
|---|---:|---:|---:|---:|
| **mlx-stats full fit** | **5 ms** (33x) | **20 ms** (73x) | **68 ms** (147x) | **144 ms** (183x) |
| mlx-stats full fit, first call on the data | 11 ms | 41 ms | 97 ms | 201 ms |
| mlx-stats full fit from a pandas DataFrame | 6 ms | 22 ms | 69 ms | 160 ms |
| mlx-stats full fit, data already in MLX (float32) | 3 ms (57x) | 4 ms (346x) | 16 ms (624x) | 30 ms (873x) |
| mlx-stats one pass (`refine=False`, no residual diagnostics) | 4 ms | 15 ms | 57 ms | 118 ms |
| mlx-stats `LinearRegression` (scikit-learn style) | 5 ms | 16 ms | 57 ms | 115 ms |
| statsmodels `OLS.fit()` | 160 ms | 1.49 s | 9.97 s | 26.4 s |
| statsmodels `OLS.fit().summary()` | 213 ms | 1.95 s | 10.7 s | 23.2 s |
| scikit-learn `LinearRegression` | 63 ms | 630 ms | 5.40 s | 8.33 s |
| NumPy normal equations (Accelerate X^T X, float64; no inference) | 4 ms | 33 ms | 187 ms | 332 ms |
| NumPy `lstsq` (no inference) | 50 ms | 493 ms | 5.00 s | 5.34 s |
| floor: copying the float64 input into MLX memory at ~117 GB/s | 1 ms | 8 ms | 36 ms | 82 ms |

Agreement with statsmodels (largest relative parameter difference): full fit 2.6e-9 / 1.3e-9 / 1.7e-8 / 3.1e-9, one pass
1.3e-7 / 4.3e-8 / 4.7e-8 / 1.9e-8 (the residual pass refines the parameters; on float32-valued data they then agree to
~1e-14, see the tests). The benchmark data are float64 draws, so the GPU sees them rounded to float32; the differences
above are that rounding, not the solver. Earlier runs (`bench-ols-1..3.log`) were made before the ingest improvements
and the review's fixes; the timings of run 3 match this one within ~5%.

Reading it:
- **Against statsmodels and scikit-learn** the gain is 33-183x (full statsmodels-type fit) and grows with size; most of
  it is algorithmic (normal equations + a small p x p solve instead of an SVD or QR of X), the rest is the GPU.
- **Against the same algorithm on the CPU** (NumPy normal equations on Accelerate, which computes no inference) the full
  fit is 1.7x faster at 10M x 10, 2.8x at 10M x 50 and 2.3x at 100M x 10, and the one-pass fit is 2.2-3.3x faster. At 1M
  rows NumPy is as fast: fixed costs (two host round trips per pass, ~0.15 ms each) dominate there.
- **From NumPy input the fit costs about the copy into GPU memory**: at 100M x 10 the one-pass fit is 118 ms against an
  82 ms copy floor and the full fit adds 26 ms for the residual pass on the already-copied data. With the data already in
  MLX memory the full fit is 30 ms for 100M x 10 (two reads of 4.4 GB of float32).
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

## Independent review (2026-09-28)

An independent review in four areas - the Metal kernels, statsmodels parity, numerical robustness, and input handling /
API - looked for defects on small inputs, and every finding had to be reproduced with a script. They reported
~45 findings (overlapping); every one was re-run before fixing. Fixed, each now a test in the suite:

- tiny inputs (< 8 values, including a one-row last block) failed to compile: MLX binds them in constant memory
- 202-255 columns overflowed threadgroup memory in the residual pass and the exact Gram
- the "exact" Gram rounded x - K (and s(x - K) for weights) to float32, which broke exact linear relations: WLS with
  x3 = x1 + x2 came back rank 3 with parameters of 1e5-1e6. Now the shifted values are exact two-floats and weights
  multiply the exact products; the kernel is exact to ~1e-13 against a float64 sum
- the pseudo-inverse depended on column units: columns in units of 1e-8 and 1e8 gave standard errors off by 10-220%
- float32 sums of squares overflowed around column spreads of 1e17 and underflowed around 1e-19: columns and y are now
  scaled by powers of two when needed (exact) and the results unscaled
- residual fourth powers overflowed float32 for residual sd ~1e9 (kurtosis, omnibus, JB came back NaN)
- WLS decided constants on the weighted data; zero-weight rows could fake a constant or an intercept
- standard errors off by up to 8e-5 just above the exact-Gram threshold (now 1e-2: within ~3e-6 before, 1e-9 after)
- `refine=False` could report a negative SSR on a tight fit; df_resid = 0 raised ZeroDivisionError
- negative weights were silently zeroed; integer DataFrame labels were turned into strings; misaligned pandas indices,
  bool/nullable columns, `add_constant` on data that already has a constant; `LinearRegression` on rank-deficient
  designs (scikit-learn's convention differs from statsmodels'), feature-name checks, constant-y `score`

Documented rather than fixed:
- rank-deficient designs with column means around 1e5-1e6: the non-identified components of the minimum-norm
  parameters differ from the exact minimum-norm solution by ~1e-4 (OLS) and up to ~3e-2 with weights spanning 1e6
  (statsmodels: 3e-5 and 6e-4). Fitted values, residuals, SSR, R^2, F and every estimable combination agree to ~1e-9.
- `precise=False` on a design the float32 Gram cannot resolve drops rank (it warns first).
- in one dummy-trap case (a category with 3 rows among 100) statsmodels' own SVD cutoff drops a real direction; there
  mlx-stats reaches the lower, correct SSR and the two disagree by design.

Speed after the fixes is the table at the top (run 4, made on the fixed code); it matches the pre-fix run 3 within ~5%.
WLS costs ~1.4x OLS (the weights are copied too, and constants need one unweighted check).

## Correctness (`tests/test_vs_statsmodels.py`)

56/56 cases (31 comparisons plus 25 regression tests from the review) match statsmodels (and scikit-learn for `LinearRegression`) on float32-valued data: constant first/last/none,
`hasconst=False`, implicit constant (full dummies), dummy trap, duplicate column, zero column, two constant columns, large
column means, a collinear pair (corr 0.99995), a near-exact fit (R^2 = 1 - 1e-11), WLS, HC0/HC1 (OLS and WLS), n = 12,
n = 6, float32 / MLX / integer / DataFrame input, 3M rows in several blocks, streaming == resident, NaN input, the virtual
constant, residuals / fitted values / predictions, and `LinearRegression` with and without an intercept and with sample
weights. Parameters agree to 1e-9-1e-14 relative, statistics to ~1e-8, and the summary text is identical apart from date,
time and one added note.
