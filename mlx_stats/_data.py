"""Getting X, y and weights onto the GPU as float32 row blocks.

NumPy float64 (what a DataFrame usually holds) is copied as raw bits and rounded on the GPU (`_fp64`), about 55 GB/s
end to end on an M5 Max against ~22 GB/s through NumPy's own cast; float32 NumPy is copied at ~60 GB/s; MLX arrays are
used in place. Blocks are ~128 MB so the copy of one overlaps the GPU work on the previous. When the float32 copy
fits comfortably in memory it is kept, so later passes (residuals, robust covariance) do not copy again.
"""
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import mlx.core as mx
from ._fp64 import to_f32, bits_to_f32

BLOCK_BYTES = 128 << 20
COPY_THREADS = 8           # one copy thread reaches ~57 GB/s into MLX memory, eight ~117 (M5 Max)
_pool = None


def _parallel_copy(dst, src):
    """np.copyto in row slices on COPY_THREADS threads (NumPy releases the GIL while copying)."""
    global _pool
    if _pool is None:
        _pool = ThreadPoolExecutor(COPY_THREADS)
    n = len(src)
    if n * src.itemsize * (src.shape[1] if src.ndim > 1 else 1) < (8 << 20):
        np.copyto(dst, src)
        return
    cuts = np.linspace(0, n, COPY_THREADS + 1).astype(int)
    list(_pool.map(lambda k: np.copyto(dst[cuts[k]:cuts[k + 1]], src[cuts[k]:cuts[k + 1]]), range(COPY_THREADS)))


def _parallel_copy_cols(dst, srcT, m):
    """srcT: (p, m) with contiguous rows (one per column of X) -> dst[:, :m], columns split over threads."""
    global _pool
    if _pool is None:
        _pool = ThreadPoolExecutor(COPY_THREADS)
    p = srcT.shape[0]
    if p * m * srcT.itemsize < (8 << 20) or p == 1:
        np.copyto(dst[:, :m], srcT)
        return
    if p >= COPY_THREADS:
        cuts = np.linspace(0, p, COPY_THREADS + 1).astype(int)
        list(_pool.map(lambda k: np.copyto(dst[cuts[k]:cuts[k + 1], :m], srcT[cuts[k]:cuts[k + 1]]), range(COPY_THREADS)))
    else:                                               # few columns: split each column's run instead
        cuts = np.linspace(0, m, COPY_THREADS // p + 2).astype(int)
        jobs = [(c, cuts[k], cuts[k + 1]) for c in range(p) for k in range(len(cuts) - 1)]
        list(_pool.map(lambda t: np.copyto(dst[t[0], t[1]:t[2]], srcT[t[0], t[1]:t[2]]), jobs))


def _mlx_buffer(shape, dtype):
    """An evaluated MLX array and a writable NumPy view of the same memory (unified memory: no copy)."""
    m = mx.zeros(shape, dtype)
    mx.eval(m)
    return m, np.asarray(m)
SAMPLE_ROWS = 2048          # enough to put the shift near the means; constancy is decided by the full pass


def _physmem():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        return 16 << 30


def _is_frame(X):
    return hasattr(X, "columns") and hasattr(X, "to_numpy") and hasattr(X, "index")


def _is_series(X):
    return hasattr(X, "to_numpy") and hasattr(X, "name") and hasattr(X, "index") and not hasattr(X, "columns")


def _frame_to_numpy(df, what):
    """A DataFrame's values as float32/float64 without a copy when every column shares one float dtype; otherwise
    converted to float64 (bool, integer, nullable and numeric categorical columns), NA becoming NaN."""
    dts = set(str(d) for d in df.dtypes)
    if dts in ({"float64"}, {"float32"}):
        return df.to_numpy()
    try:
        return df.to_numpy(dtype=np.float64, na_value=np.nan)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{what} must be numeric: {e}") from None


def _as_2d(X, what):
    """-> (array: numpy or MLX, column labels or None, row index or None). Labels keep their type (as pandas does)."""
    names, index = None, None
    if _is_frame(X):
        names, index = list(X.columns), X.index
        if names and isinstance(names[0], tuple):                        # MultiIndex columns: 'a_b', as statsmodels
            names = ["_".join(str(v) for v in t) for t in names]
        X = _frame_to_numpy(X, what)
    elif _is_series(X):
        names, index = ([X.name] if X.name else None), X.index            # statsmodels: a falsy name is no name
        X = _frame_to_numpy(X.to_frame(), what)
    if isinstance(X, mx.array):
        if X.ndim == 1:
            X = X[:, None]
        if X.ndim != 2:
            raise ValueError(f"{what} must be 2-D")
        return X, names, index
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2:
        raise ValueError(f"{what} must be 2-D")
    if X.dtype not in (np.float32, np.float64):
        if X.dtype == object or not (np.issubdtype(X.dtype, np.number) or X.dtype == np.bool_):
            raise ValueError(f"{what} must be numeric, got {X.dtype}")
        X = X.astype(np.float64)
    return X, names, index


def _as_1d(v, what):
    """-> (array: numpy or MLX 1-D, name or None, row index or None)."""
    name, index = None, None
    if _is_frame(v):
        if v.shape[1] != 1:
            raise ValueError(f"{what} must be one column")
        name, index = v.columns[0], v.index
        v = _frame_to_numpy(v, what).reshape(-1)
    elif _is_series(v):
        name, index = v.name, v.index
        v = _frame_to_numpy(v.to_frame(), what).reshape(-1)
    if isinstance(v, mx.array):
        if v.ndim == 2 and 1 in v.shape:
            v = v.reshape(-1)
        if v.ndim != 1:
            raise ValueError(f"{what} must be 1-D")
        return v, name, index
    v = np.asarray(v)
    if v.ndim == 2 and 1 in v.shape:
        v = v.reshape(-1)
    if v.ndim != 1:
        raise ValueError(f"{what} must be 1-D")
    if v.dtype not in (np.float32, np.float64):
        if v.dtype == object or not (np.issubdtype(v.dtype, np.number) or v.dtype == np.bool_):
            raise ValueError(f"{what} must be numeric, got {v.dtype}")
        v = v.astype(np.float64)
    return v, name, index


def _to_dev(a):
    """numpy or MLX block -> float32 MLX, C-contiguous."""
    if isinstance(a, mx.array):
        return a if a.dtype == mx.float32 else a.astype(mx.float32)
    if a.dtype == np.float64:
        return to_f32(a)                              # copies to a contiguous buffer if needed
    return mx.array(np.ascontiguousarray(a, dtype=np.float32))


def _pow2_scale(spread):
    """A power of two that brings `spread` near 1 when it is far from it (float32 squares and sums overflow past
    ~1e19 and underflow below ~1e-19); 1 otherwise. Powers of two scale float32 values exactly."""
    if not np.isfinite(spread) or spread == 0 or 2.0 ** -20 <= spread <= 2.0 ** 20:
        return 1.0
    return float(2.0 ** -np.round(np.log2(spread)))


class DeviceData:
    """X (n x p), y (n,) and optional weights (n,) as float32 MLX row blocks, shifted near the column means and, when
    a column's spread is far from 1, scaled by a power of two (undone exactly in float64 by the caller)."""

    def __init__(self, X, y, weights=None, resident=None, ones_if_empty=False):
        if not mx.metal.is_available():
            raise RuntimeError("mlx-stats runs its kernels on the GPU of an Apple Silicon Mac (Metal), "
                               "and no Metal device is available here")
        self.X, self.xnames, xindex = _as_2d(X, "exog")
        if self.X.shape[1] == 0:
            if not ones_if_empty:
                raise ValueError("exog has no columns")
            # an intercept-only model: one column of ones stands in for the empty exog (and is the constant)
            self.X = np.ones((self.X.shape[0], 1), np.float32)
            self.xnames = None if self.xnames is None else ["const"]
        self.y, self.yname, yindex = _as_1d(y, "endog")
        self.n, self.p = self.X.shape
        if self.n == 0:
            raise ValueError("no observations: exog has 0 rows")
        if self.y.shape[0] != self.n:
            raise ValueError(f"endog has {self.y.shape[0]} rows but exog has {self.n}")
        if xindex is not None and yindex is not None and not xindex.equals(yindex):
            raise ValueError("The indices for endog and exog are not aligned")
        self.row_index = yindex if yindex is not None else xindex
        self.w = None
        if weights is not None and not (np.ndim(weights) == 0 and float(np.asarray(weights)) == 1.0):
            if np.ndim(weights) == 0 and not isinstance(weights, mx.array):
                weights = np.full(self.n, float(np.asarray(weights)))
            self.w, _, windex = _as_1d(weights, "weights")
            if self.w.shape[0] != self.n:
                raise ValueError("weights must have one value per row")
            if windex is not None and self.row_index is not None and not windex.equals(self.row_index):
                raise ValueError("The indices for weights and exog are not aligned")
        self.rows = max(128, (BLOCK_BYTES // (8 * max(1, self.p + 2))) // 128 * 128)
        dev_bytes = self.n * (self.p + 2) * 4
        self.resident = resident if resident is not None else dev_bytes <= 0.4 * _physmem()
        self._cache = None
        self._wstats = []
        self.has_zero_weight = False
        self._prepare()

    def _prepare(self):
        """Shift K (sample means, or the value itself where the sample is constant) and power-of-two scales."""
        xs, ys = self.sample()
        lo, hi = xs.min(0), xs.max(0)
        K = np.where(lo == hi, lo, xs.mean(0))
        Ky = ys[0] if ys.min() == ys.max() else ys.mean()
        K = K.astype(np.float32).astype(np.float64); Ky = float(np.float32(Ky))
        self.xscale = np.array([_pow2_scale(np.max(np.abs(xs[:, j] - K[j])) if xs.shape[0] else 0.0) for j in range(self.p)])
        self.yscale = _pow2_scale(np.max(np.abs(ys - Ky)) if ys.size else 0.0)
        self.scaled = bool(np.any(self.xscale != 1.0) or self.yscale != 1.0)
        self._shift = np.concatenate([K * self.xscale, [Ky * self.yscale]]).astype(np.float32)
        self._xs_mx = mx.array(self.xscale.astype(np.float32))

    def shift(self):
        return self._shift

    def _finish(self, Xb, yb, wb):
        """Apply the scales, form sqrt(w), and keep the per-block weight range for check_weights()."""
        if self.scaled:
            Xb = Xb * self._xs_mx
            yb = yb * np.float32(self.yscale)
        sb = None
        if wb is not None:
            self._wstats.append(mx.stack([mx.min(wb), mx.max(wb)]))
            sb = mx.sqrt(mx.maximum(wb, 0.0))
        return Xb, yb, sb, wb

    def check_weights(self):
        """Raise on negative or all-zero weights (after the first pass has evaluated the block statistics)."""
        if self.w is None or not self._wstats:
            return
        st = np.array(mx.stack(self._wstats))
        self._wstats = []
        if np.nanmin(st[:, 0]) < 0:
            raise ValueError("weights must be non-negative")
        if np.nanmax(st[:, 1]) <= 0:
            raise ValueError("weights are all zero")
        self.has_zero_weight = bool(np.nanmin(st[:, 0]) == 0)

    def unweighted_constant(self):
        """Per column: every value equals its shift exactly (a constant column, judged on the unweighted data as
        statsmodels does). One reduction over the blocks; used for WLS, where the weighted Gram cannot tell."""
        K = mx.array(self._shift[:self.p])
        m = None
        for Xb, _, _, _ in self.blocks():
            v = mx.max(mx.abs(Xb - K), axis=0)
            m = v if m is None else mx.maximum(m, v)
        return np.array(m) == 0.0

    def _make(self, i, j):
        Xb = _to_dev(self.X[i:j])
        yb = _to_dev(self.y[i:j])
        wb = _to_dev(self.w[i:j]) if self.w is not None else None
        return self._finish(Xb, yb, wb)

    def _stream_numpy(self, keep):
        """NumPy X in blocks, copied on several threads straight into MLX memory. float64 goes through a ring of
        three reused staging buffers as raw bits and is rounded on the GPU; float32 is copied into its block."""
        X = self.X
        f64 = X.dtype == np.float64
        colmajor = not X.flags.c_contiguous and X.flags.f_contiguous       # what a DataFrame usually hands over
        if not X.flags.c_contiguous and not colmajor:
            X = np.ascontiguousarray(X)
        ring, pending = [], [None, None, None]
        for b, i in enumerate(range(0, self.n, self.rows)):
            j = min(self.n, i + self.rows)
            if colmajor:
                # each column's rows i..j are contiguous: copy them into a (p, rows) block on several threads,
                # then transpose on the GPU instead of on one CPU core
                srcT = X.T[:, i:j]
                k = b % 3
                if len(ring) <= k:
                    ring.append(_mlx_buffer((self.p, (2 if f64 else 1) * self.rows), mx.uint32 if f64 else mx.float32))
                if pending[k] is not None:
                    mx.eval(pending[k])
                buf, view = ring[k]
                m = j - i
                if f64:
                    _parallel_copy_cols(view, srcT.view(np.uint32), 2 * m)    # rows stay contiguous: no copy
                    colsT = bits_to_f32(buf[:, :2 * m] if m < self.rows else buf, (self.p, m))
                else:
                    _parallel_copy_cols(view, srcT.astype(np.float32, copy=False), m)
                    colsT = buf[:, :m] if m < self.rows else buf
                Xb = mx.contiguous(mx.transpose(colsT))
                pending[k] = Xb
                blk = self._finish(Xb, _to_dev(self.y[i:j]), _to_dev(self.w[i:j]) if self.w is not None else None)
                mx.async_eval(blk[0])
                if keep is not None:
                    keep.append(blk)
                yield blk
                continue
            src = X[i:j]
            if f64:
                k = b % 3
                if len(ring) <= k:
                    ring.append(_mlx_buffer((self.rows, 2 * self.p), mx.uint32))
                if pending[k] is not None:
                    mx.eval(pending[k])                 # the GPU has finished reading this staging buffer
                buf, view = ring[k]
                _parallel_copy(view[:j - i], src.view(np.uint32).reshape(j - i, 2 * self.p))
                Xb = bits_to_f32(buf[:j - i] if j - i < self.rows else buf, (j - i, self.p))
                pending[k] = Xb
            else:
                Xb, view = _mlx_buffer((j - i, self.p), mx.float32)
                _parallel_copy(view, src.astype(np.float32, copy=False))
            blk = self._finish(Xb, _to_dev(self.y[i:j]), _to_dev(self.w[i:j]) if self.w is not None else None)
            mx.async_eval(blk[0])
            if keep is not None:
                keep.append(blk)
            yield blk

    def blocks(self):
        """Yield (X, y, s, w) float32 MLX blocks in row order; s = sqrt(weights) and w the weights, or None."""
        if self._cache is not None:
            yield from self._cache
            return
        if isinstance(self.X, mx.array) and isinstance(self.y, mx.array) and (self.w is None or isinstance(self.w, mx.array)):
            b = self._make(0, self.n)
            if self.scaled or self.X.dtype != mx.float32:
                self._cache = [b]                        # a scaled or converted copy: make it once
            yield b
            return
        if self._cache is not None:
            yield from self._cache
            return
        keep = [] if self.resident else None
        if isinstance(self.X, np.ndarray):
            yield from self._stream_numpy(keep)
        else:
            for i in range(0, self.n, self.rows):
                b = self._make(i, min(self.n, i + self.rows))
                if keep is not None:
                    keep.append(b)
                yield b
        if keep is not None:
            self._cache = keep

    def sample(self):
        """A strided sample of up to SAMPLE_ROWS rows as float64 numpy, rounded to float32 first (the GPU sees
        float32 values, so a constant column's shift must be its float32 value exactly)."""
        step = max(1, self.n // SAMPLE_ROWS)
        def take(a):
            s = a[::step][:SAMPLE_ROWS]
            s = np.array(s.astype(mx.float32)) if isinstance(s, mx.array) else np.asarray(s).astype(np.float32)
            return s.astype(np.float64)
        return take(self.X), take(self.y)
