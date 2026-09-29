"""Run the GPU passes over a DeviceData's blocks and bring back float64 sums."""
import numpy as np
import mlx.core as mx
from ._gram import gram_launch, unpack
from ._gram_exact import gram_exact_launch, unpack_exact
from ._resid import resid_launch, resid_unpack


def gram_pass(data, shift, with_y=True, scales=None, exact=False):
    """sum over rows of s^2 [x - K, (y - K_y), 1][...]^T as float64. scales: optional list of per-block MLX scale
    vectors replacing sqrt(weights) (used for robust covariance meat). exact: products and sums in two-float
    arithmetic (`_gram_exact`) instead of the matrix unit."""
    launch, unp = (gram_exact_launch, unpack_exact) if exact else (gram_launch, unpack)
    K = mx.array(shift if with_y else shift[:-1])
    reds, lay = [], None
    for b, (Xb, yb, sb) in enumerate(data.blocks()):
        s = scales[b] if scales is not None else sb
        red, lay = launch(Xb, yb if with_y else None, s, K)
        mx.async_eval(red)
        reds.append(red)
    mx.eval(*reds)
    return sum(unp(r, lay) for r in reds)


def resid_pass(data, shift, beta_s, c0, write_r=False):
    """-> (gradient sum w [a, 1] r (p + 1), power sums of rho (4), Durbin-Watson numerator, list of r blocks)."""
    K = mx.array(shift)
    parts, routs = [], []
    for Xb, yb, sb in data.blocks():
        part, rout, G, nout = resid_launch(Xb, yb, sb, K, beta_s, c0, write_r)
        mx.async_eval(part if rout is None else [part, rout])
        parts.append((part, G, nout)); routs.append(rout)
    mx.eval(*[p for p, _, _ in parts])
    grad, mom, dw = resid_unpack(parts, data.p)
    return grad, mom, dw, routs
