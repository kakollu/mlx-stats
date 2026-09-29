"""Run the GPU passes over a DeviceData's blocks and bring back float64 sums."""
import numpy as np
import mlx.core as mx
from ._gram import gram_launch, unpack
from ._gram_exact import gram_exact_launch, unpack_exact
from ._resid import resid_launch, resid_unpack


def gram_pass(data, shift, with_y=True, scales=None, exact=False):
    """sum over rows of w [x - K, (y - K_y), 1][...]^T as float64, w the row weights (1 for OLS).
    scales: optional per-block MLX vectors s replacing sqrt(w), so the weight is s^2 (robust covariance meat).
    exact: every product formed and summed in two-float arithmetic (`_gram_exact`) instead of the matrix unit."""
    K = mx.array(shift if with_y else shift[:-1])
    reds, lay = [], None
    for b, (Xb, yb, sb, wb) in enumerate(data.blocks()):
        yy = yb if with_y else None
        if exact:
            if scales is None:
                w = wb
            else:
                w = None if scales[b] is None else scales[b] * scales[b]          # None: unweighted
            red, lay = gram_exact_launch(Xb, yy, w, K)
        else:
            red, lay = gram_launch(Xb, yy, scales[b] if scales is not None else sb, K)
        mx.async_eval(red)
        reds.append(red)
    mx.eval(*reds)
    return sum((unpack_exact if exact else unpack)(r, lay) for r in reds)


def resid_pass(data, shift, beta_s, c0, rscale=1.0, write_r=False):
    """-> (gradient sum w [a, 1] r (p + 1), sums (9, see resid_unpack), list of r blocks or Nones)."""
    K = mx.array(shift)
    parts, routs = [], []
    for Xb, yb, sb, wb in data.blocks():
        part, rout, G, nout = resid_launch(Xb, yb, sb, wb, K, beta_s, c0, rscale, write_r)
        mx.async_eval(part if rout is None else [part, rout])
        parts.append((part, G, nout)); routs.append(rout)
    mx.eval(*[p for p, _, _ in parts])
    grad, sums = resid_unpack(parts, data.p)
    return grad, sums, routs
