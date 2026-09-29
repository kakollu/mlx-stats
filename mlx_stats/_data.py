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


def _as_2d(X, what):
    """-> (array: numpy or MLX, names or None). DataFrames give names; Series/1-D become one column."""
    names = None
    if hasattr(X, "columns") and hasattr(X, "to_numpy"):                 # pandas DataFrame
        names = [str(c) for c in X.columns]
        X = X.to_numpy()
    elif hasattr(X, "to_numpy") and hasattr(X, "name"):                   # pandas Series
        names = [str(X.name) if X.name is not None else "x1"]
        X = X.to_numpy()[:, None]
    if isinstance(X, mx.array):
        if X.ndim == 1:
            X = X[:, None]
        if X.ndim != 2:
            raise ValueError(f"{what} must be 2-D")
        return X, names
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2:
        raise ValueError(f"{what} must be 2-D")
    if X.dtype not in (np.float32, np.float64):
        if X.dtype == object or not np.issubdtype(X.dtype, np.number) and X.dtype != np.bool_:
            raise ValueError(f"{what} must be numeric, got {X.dtype}")
        X = X.astype(np.float64)
    return X, names


def _as_1d(v, what):
    name = None
    if hasattr(v, "to_numpy"):
        name = getattr(v, "name", None)
        if hasattr(v, "columns"):
            if v.shape[1] != 1:
                raise ValueError(f"{what} must be one column")
            name = v.columns[0]
        v = v.to_numpy().reshape(-1)
    if isinstance(v, mx.array):
        return v.reshape(-1), name
    v = np.asarray(v)
    if v.ndim == 2 and 1 in v.shape:
        v = v.reshape(-1)
    if v.ndim != 1:
        raise ValueError(f"{what} must be 1-D")
    if v.dtype not in (np.float32, np.float64):
        v = v.astype(np.float64)
    return v, (str(name) if name is not None else None)


def _to_dev(a):
    """numpy or MLX block -> float32 MLX, C-contiguous."""
    if isinstance(a, mx.array):
        return a if a.dtype == mx.float32 else a.astype(mx.float32)
    if a.dtype == np.float64:
        return to_f32(a)                              # copies to a contiguous buffer if needed
    return mx.array(np.ascontiguousarray(a, dtype=np.float32))


class DeviceData:
    """X (n x p), y (n,) and optional weights (n,) as float32 MLX row blocks."""

    def __init__(self, X, y, weights=None, resident=None):
        self.X, self.xnames = _as_2d(X, "exog")
        self.y, self.yname = _as_1d(y, "endog")
        self.n, self.p = self.X.shape
        if self.y.shape[0] != self.n:
            raise ValueError(f"endog has {self.y.shape[0]} rows but exog has {self.n}")
        self.w = None
        if weights is not None and not (np.isscalar(weights) and weights == 1):
            if np.isscalar(weights):
                weights = np.full(self.n, float(weights))
            self.w, _ = _as_1d(weights, "weights")
            if self.w.shape[0] != self.n:
                raise ValueError("weights must have one value per row")
        self.rows = max(128, (BLOCK_BYTES // (8 * max(1, self.p + 2))) // 128 * 128)
        dev_bytes = self.n * (self.p + 2) * 4
        self.resident = resident if resident is not None else dev_bytes <= 0.4 * _physmem()
        self._cache = None

    def _make(self, i, j):
        Xb = _to_dev(self.X[i:j])
        yb = _to_dev(self.y[i:j])
        sb = None
        if self.w is not None:
            wb = _to_dev(self.w[i:j])
            sb = mx.sqrt(mx.maximum(wb, 0.0))
        return Xb, yb, sb

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
                yb = _to_dev(self.y[i:j])
                sb = None if self.w is None else mx.sqrt(mx.maximum(_to_dev(self.w[i:j]), 0.0))
                mx.async_eval(Xb)
                blk = (Xb, yb, sb)
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
            yb = _to_dev(self.y[i:j])
            sb = None
            if self.w is not None:
                sb = mx.sqrt(mx.maximum(_to_dev(self.w[i:j]), 0.0))
            mx.async_eval(Xb)
            blk = (Xb, yb, sb)
            if keep is not None:
                keep.append(blk)
            yield blk

    def blocks(self):
        """Yield (X, y, s) float32 MLX blocks in row order; s = sqrt(weights) or None."""
        if isinstance(self.X, mx.array) and isinstance(self.y, mx.array) and (self.w is None or isinstance(self.w, mx.array)):
            yield self._make(0, self.n)
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

    def shift(self):
        """Shift K for (x columns, y): the sample mean, or the value itself where the sample is constant."""
        xs, ys = self.sample()
        lo, hi = xs.min(0), xs.max(0)
        K = np.where(lo == hi, lo, xs.mean(0))
        Ky = ys[0] if ys.min() == ys.max() else ys.mean()
        return np.concatenate([K, [Ky]]).astype(np.float32)
