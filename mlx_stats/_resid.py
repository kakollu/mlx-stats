"""The residual pass: r = y - x.beta for every row, with the coefficients in two-float precision, and the sums a
linear model needs from the residuals, in one read of X and y.

Per row, in shifted coordinates a = x - K, a_y = y - K_y: r = a_y - c0 - sum_j a_j beta_j, the dot product carried
as an exact two-float sum (TwoProd on the high part of each coefficient, the low part added in), so r is accurate
to float32 rounding of r itself even when the fit is close to exact. With row weights s = sqrt(w) (WLS), the
whitened residual is rho = s r.

Per threadgroup, as (hi, lo) two-float sums: the gradient sum_i s_i^2 [a_i, 1] r_i (which is X^T W r in shifted
coordinates: it drives one step of iterative refinement), sum rho, rho^2, rho^3, rho^4 (skew, kurtosis, the omnibus
and Jarque-Bera tests, and the exact sum of squared residuals), sum (rho_t - rho_{t-1})^2 inside the range and the
first and last rho of the range (Durbin-Watson across range boundaries). Optionally r itself is written out.
"""
import numpy as np
import mlx.core as mx
from ._gram import _HEADER

_SRC = """
    uint tid = thread_position_in_threadgroup.x;
    uint g = threadgroup_position_in_grid.x;
    threadgroup float Za[{R} * {LDA}];
    threadgroup float RHO[{R}], VH[{R}], VL[{R}];
    threadgroup float CF[2 * {P} + 2];
    threadgroup float SH[{P} + 1];
    threadgroup float LAST[1];
    threadgroup float RED[16 * {R}];
    for (uint j = tid; j < 2 * {P} + 2; j += {NT}) CF[j] = coef[j];
    for (uint j = tid; j < {P} + 1; j += {NT}) SH[j] = shift[j];
    ulong n = (ulong)meta[0] | ((ulong)meta[1] << 32);
    ulong per = (ulong)meta[2] | ((ulong)meta[3] << 32);
    ulong r0 = (ulong)g * per, r1 = min(r0 + per, n);
    float m1h = 0, m1l = 0, m2h = 0, m2l = 0, m3h = 0, m3l = 0, m4h = 0, m4l = 0, dwh = 0, dwl = 0;
    float u2h = 0, u2l = 0, wyh = 0, wyl = 0, swh = 0, swl = 0;
    float gh = 0, gl = 0;
    float first = 0.0f;
    if (tid == 0) LAST[0] = 0.0f;
    const device float4 *X4 = (const device float4 *)X;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (ulong rb = r0; rb < r1; rb += {R}) {{
        uint nr = (uint)min((ulong){R}, r1 - rb);
        if (nr == {R}) {{
            for (uint f = tid; f < {F4}; f += {NT}) {{
                float4 v = X4[(rb >> 2) * {P} + f];
                #pragma unroll
                for (uint c = 0; c < 4; c++) {{ uint e = 4 * f + c, i = e / {P}, j = e % {P}; Za[i * {LDA} + j] = v[c]; }}
            }}
        }} else {{
            for (uint e = tid; e < nr * {P}; e += {NT}) {{ uint i = e / {P}, j = e % {P}; Za[i * {LDA} + j] = X[(rb + i) * {P} + j]; }}
        }}
        float yv = 0.0f, sv = 1.0f;
        if (tid < nr) {{ yv = Y[rb + tid]; {SLOAD} }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < nr) {{
            // r = (y - K_y) - c0 - sum_j (x_j - K_j) beta_j, every step an exact two-float operation:
            // the shifted values are split exactly (TwoSum), products exactly (TwoProd), and only r is rounded
            float sh = 0.0f, sl = 0.0f;
            for (uint j = 0; j < {P}; j++) {{
                float ah, al; two_sum(Za[tid * {LDA} + j], -SH[j], ah, al);
                float bh = CF[j], bl = CF[{P} + j];
                float p = ah * bh; float pe = fma(ah, bh, -p);
                two_sum_acc(sh, sl, p); sl += pe + (ah * bl + al * bh);
            }}
            float yh, yl; two_sum(yv, -SH[{P}], yh, yl);
            float dh, dl; two_sum(yh, -sh, dh, dl); dl += yl - sl;
            float eh, el; two_sum(dh, -CF[2 * {P}], eh, el); el += dl - CF[2 * {P} + 1];
            float rh = eh + el, rl = el - (rh - eh);                // r as an exact two-float (rh, rl)
            float r = rh;
            // exact sums for the total sum of squares: u = s a_y, sum u^2, sum s u (= sum w a_y), sum s^2 (= sum w)
            float uh = sv * yh, ul = fma(sv, yh, -uh) + sv * yl;
            float q = uh * uh; u2l += fma(uh, uh, -q) + 2.0f * uh * ul; two_sum_acc(u2h, u2l, q);
            float c = sv * uh; wyl += fma(sv, uh, -c) + sv * ul; two_sum_acc(wyh, wyl, c);
            float ww = sv * sv; swl += fma(sv, sv, -ww); two_sum_acc(swh, swl, ww);
            float rho = sv * r;
            {ROUT}
            // v = w r = s (s r), kept as a two-float so X^T W r is not limited by the rounding of r
            float th = sv * rh, tl = fma(sv, rh, -th) + sv * rl;
            float vh = sv * th, vl = fma(sv, th, -vh) + sv * tl;
            RHO[tid] = rho; VH[tid] = vh; VL[tid] = vl;
            float rho2 = rho * rho;
            two_sum_acc(m1h, m1l, rho); two_sum_acc(m2h, m2l, rho2);
            two_sum_acc(m3h, m3l, rho2 * rho); two_sum_acc(m4h, m4l, rho2 * rho2);
            if (rb == r0 && tid == 0) first = rho;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < nr && !(rb == r0 && tid == 0)) {{
            float d = RHO[tid] - (tid > 0 ? RHO[tid - 1] : LAST[0]);
            two_sum_acc(dwh, dwl, d * d);
        }}
        if (tid <= {P}) {{
            // X^T W r: the terms cancel to near zero at the solution, so each product goes in exactly (TwoProd)
            if (tid < {P}) {{
                float k = SH[tid];
                for (uint i = 0; i < nr; i++) {{
                    float ah, al; two_sum(Za[i * {LDA} + tid], -k, ah, al);     // x - K exactly
                    float vh = VH[i], vl = VL[i];
                    float p = ah * vh; gl += fma(ah, vh, -p) + (ah * vl + al * vh); two_sum_acc(gh, gl, p);
                }}
            }} else {{ for (uint i = 0; i < nr; i++) {{ two_sum_acc(gh, gl, VH[i]); gl += VL[i]; }} }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) LAST[0] = RHO[nr - 1];
    }}
    // reduce the per-thread moment sums through threadgroup memory
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 0; k < 8; k++) {{
        float vh = k == 0 ? m1h : k == 1 ? m2h : k == 2 ? m3h : k == 3 ? m4h : k == 4 ? dwh : k == 5 ? u2h : k == 6 ? wyh : swh;
        float vl = k == 0 ? m1l : k == 1 ? m2l : k == 2 ? m3l : k == 3 ? m4l : k == 4 ? dwl : k == 5 ? u2l : k == 6 ? wyl : swl;
        if (tid < {R}) {{ RED[(2 * k) * {R} + tid] = vh; RED[(2 * k + 1) * {R} + tid] = vl; }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ulong o = (ulong)g * {NOUT};
    if (tid < 8) {{
        float h = 0.0f, l = 0.0f;
        for (uint i = 0; i < {R}; i++) {{ two_sum_acc(h, l, RED[(2 * tid) * {R} + i]); l += RED[(2 * tid + 1) * {R} + i]; }}
        uint k = tid < 5 ? 2 * tid : 2 * tid + 3;            // slots 0-9: moments and DW; 13-18: TSS sums (10-12 used below)
        part[o + 2 * ({P} + 1) + k] = h; part[o + 2 * ({P} + 1) + k + 1] = l;
    }}
    if (tid <= {P}) {{ part[o + 2 * tid] = gh; part[o + 2 * tid + 1] = gl; }}
    if (tid == 0) {{ part[o + 2 * ({P} + 1) + 10] = first; part[o + 2 * ({P} + 1) + 11] = LAST[0];
                     part[o + 2 * ({P} + 1) + 12] = (float)(r1 > r0); }}
"""

_kernels = {}
NT = 256


def _config(p):
    r = 128 if p <= 32 else 64 if p <= 64 else 32
    lda = p | 1                                       # odd stride: a thread per row reads conflict-free
    return r, lda


def _kernel(p, has_s, write_r):
    key = (p, has_s, write_r)
    if key in _kernels:
        return _kernels[key]
    r, lda = _config(p)
    nout = 2 * (p + 1) + 19
    src = _SRC.format(R=r, LDA=lda, P=p, NT=NT, F4=r * p // 4, NOUT=nout,
                      SLOAD="sv = S[rb + tid];" if has_s else "",
                      ROUT="rout[rb + tid] = r;" if write_r else "")
    ins = ["X", "Y"] + (["S"] if has_s else []) + ["shift", "coef", "meta"]
    outs = ["part"] + (["rout"] if write_r else [])
    k = mx.fast.metal_kernel(name=f"mlxstats_resid_p{p}_s{int(has_s)}_r{int(write_r)}", input_names=ins,
                             output_names=outs, source=src, header=_HEADER)
    _kernels[key] = (k, r, nout)
    return _kernels[key]


def split2(v):
    """float64 -> (hi, lo) float32 pairs with hi + lo = v to ~2^-48 relative."""
    v = np.asarray(v, np.float64)
    hi = v.astype(np.float32)
    lo = (v - hi.astype(np.float64)).astype(np.float32)
    return hi, lo


def resid_launch(X, y, s, shift, beta, c0, write_r=False):
    """Queue the residual pass over one MLX block. beta: (p,) float64 in shifted coordinates, c0: float64 constant
    in shifted coordinates. Returns (part, rout or None, G, nout)."""
    n, p = X.shape
    k, r, nout = _kernel(p, s is not None, write_r)
    G = max(1, min(320, -(-n // (r * 4))))
    per = -(-n // G)
    per = -(-per // r) * r
    G = max(1, -(-n // per))
    bh, bl = split2(beta); ch, cl = split2(c0)
    coef = mx.array(np.concatenate([bh, bl, [ch], [cl]]).astype(np.float32))
    meta = mx.array(np.array([n & 0xffffffff, n >> 32, per & 0xffffffff, per >> 32], dtype=np.uint32))
    ins = [X, y] + ([s] if s is not None else []) + [shift, coef, meta]
    shapes = [(G * nout,)] + ([(n,)] if write_r else [])
    outs = k(inputs=ins, grid=(G * NT, 1, 1), threadgroup=(NT, 1, 1), output_shapes=shapes,
             output_dtypes=[mx.float32] * len(shapes))
    return outs[0], (outs[1] if write_r else None), G, nout


def resid_unpack(parts, p):
    """parts: list of (part MLX array, G, nout), one per block in row order -> float64 sums."""
    grad = np.zeros(p + 1); mom = np.zeros(8); dw = 0.0
    prev_last = None
    for part, G, nout in parts:
        a = np.array(part, dtype=np.float32).astype(np.float64).reshape(G, nout)
        grad += (a[:, 0:2 * (p + 1):2] + a[:, 1:2 * (p + 1):2]).sum(0)
        o = 2 * (p + 1)
        mom[:4] += (a[:, o:o + 8:2] + a[:, o + 1:o + 8:2]).sum(0)
        mom[5:8] += (a[:, o + 13:o + 19:2] + a[:, o + 14:o + 19:2]).sum(0)
        dw += (a[:, o + 8] + a[:, o + 9]).sum()
        for gi in range(G):
            if a[gi, o + 12] == 0:
                continue
            if prev_last is not None:
                dw += (a[gi, o + 10] - prev_last) ** 2
            prev_last = a[gi, o + 11]
    return grad, mom, dw
