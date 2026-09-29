"""The residual pass: r = y - x.beta for every row, with the coefficients in two-float precision, and the sums a
linear model needs from the residuals, in one read of X and y.

Per row, in shifted coordinates a = x - K, a_y = y - K_y: r = a_y - c0 - sum_j a_j beta_j. The shifted values are split
exactly (TwoSum), products formed exactly (TwoProd), and r is kept as an exact two-float (rh, rl). With row weights w
(WLS) every weighted sum uses w itself, not a rounded sqrt(w), so the sums are those of the weights as given.

Per threadgroup, as (hi, lo) two-float sums:
  - X^T W r in shifted coordinates, sum w [a, 1] r, exactly: the step of iterative refinement
  - sum w r^2 exactly: the sum of squared residuals
  - sum w a_y^2, sum w a_y, sum w exactly: the total sums of squares
  - sum rho', rho'^2, rho'^3, rho'^4 and sum (rho'_t - rho'_{t-1})^2 with rho' = sqrt(w) r * rscale, a power of two
    chosen on the host so that rho' is O(1) and the fourth powers cannot overflow float32: skew, kurtosis, omnibus,
    Jarque-Bera, Durbin-Watson
  - the first and last rho' of the range (Durbin-Watson across range boundaries)
Optionally r itself (rh) is written out.
"""
import numpy as np
import mlx.core as mx
from ._gram import _HEADER, pad_min

NT = 256
TG_BUDGET = 28 * 1024           # bytes of threadgroup memory a kernel may declare (the hardware limit is 32 KB)

_SRC = """
    uint tid = thread_position_in_threadgroup.x;
    uint g = threadgroup_position_in_grid.x;
    threadgroup float Za[{R} * {LDA}];
    threadgroup float RHO[{R}], VH[{R}], VL[{R}];
    threadgroup float CF[2 * {P} + 2];
    threadgroup float SH[{P} + 1];
    threadgroup float LAST[1];
    threadgroup float RED[18 * {R}];
    threadgroup float GR[2 * {NT}];
    for (uint j = tid; j < 2 * {P} + 2; j += {NT}) CF[j] = coef[j];
    for (uint j = tid; j < {P} + 1; j += {NT}) SH[j] = shift[j];
    float rs = rscale[0];
    ulong n = (ulong)meta[0] | ((ulong)meta[1] << 32);
    ulong per = (ulong)meta[2] | ((ulong)meta[3] << 32);
    ulong r0 = (ulong)g * per, r1 = min(r0 + per, n);
    // nine two-float sums: rho' powers 1-4, Durbin-Watson, sum w a_y^2, sum w a_y, sum w, sum w r^2
    float sh_[9], sl_[9];
    for (uint k = 0; k < 9; k++) {{ sh_[k] = 0.0f; sl_[k] = 0.0f; }}
    float gh = 0, gl = 0;
    // X^T W r is split over (column, share of the stage's rows): p + 1 threads alone would each walk every row
    uint gcol = tid % ({P} + 1), gsp = tid / ({P} + 1);
    bool gact = tid < {GSPLIT} * ({P} + 1);
    float first = 0.0f;
    if (tid == 0) LAST[0] = 0.0f;
    const device packed_float4 *X4 = (const device packed_float4 *)X;     // 4-byte alignment is enough
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (ulong rb = r0; rb < r1; rb += {R}) {{
        uint nr = (uint)min((ulong){R}, r1 - rb);
        if (nr == {R}) {{
            for (uint f = tid; f < {F4}; f += {NT}) {{
                float4 v = float4(X4[(rb >> 2) * {P} + f]);
                #pragma unroll
                for (uint c = 0; c < 4; c++) {{ uint e = 4 * f + c, i = e / {P}, j = e % {P}; Za[i * {LDA} + j] = v[c]; }}
            }}
        }} else {{
            for (uint e = tid; e < nr * {P}; e += {NT}) {{ uint i = e / {P}, j = e % {P}; Za[i * {LDA} + j] = X[(rb + i) * {P} + j]; }}
        }}
        float yv = 0.0f, sv = 1.0f, wv = 1.0f;
        if (tid < nr) {{ yv = Y[rb + tid]; {WLOAD} }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < nr) {{
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
            float rh = eh + el, rl = el - (rh - eh);                 // r as an exact two-float
            {ROUT}
            // v = w r (two-float), for X^T W r and sum w r^2
            float vh = wv * rh, vl = fma(wv, rh, -vh) + wv * rl;
            float q = rh * vh; float ql = fma(rh, vh, -q) + rh * vl + rl * vh;
            two_sum_acc(sh_[8], sl_[8], q); sl_[8] += ql;
            // total sums of squares: t = w a_y, sum t a_y, sum t, sum w
            float th = wv * yh, tl = fma(wv, yh, -th) + wv * yl;
            float u = yh * th; float ul = fma(yh, th, -u) + yh * tl + yl * th;
            two_sum_acc(sh_[5], sl_[5], u); sl_[5] += ul;
            two_sum_acc(sh_[6], sl_[6], th); sl_[6] += tl;
            two_sum_acc(sh_[7], sl_[7], wv);
            // moments of rho' = sqrt(w) r * rscale (O(1) by construction)
            float rho = (sv * rh) * rs;
            float rho2 = rho * rho;
            two_sum_acc(sh_[0], sl_[0], rho); two_sum_acc(sh_[1], sl_[1], rho2);
            two_sum_acc(sh_[2], sl_[2], rho2 * rho); two_sum_acc(sh_[3], sl_[3], rho2 * rho2);
            RHO[tid] = rho; VH[tid] = vh; VL[tid] = vl;
            if (rb == r0 && tid == 0) first = rho;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < nr && !(rb == r0 && tid == 0)) {{
            float d = RHO[tid] - (tid > 0 ? RHO[tid - 1] : LAST[0]);
            two_sum_acc(sh_[4], sl_[4], d * d);
        }}
        if (gact) {{
            if (gcol < {P}) {{
                float k = SH[gcol];
                for (uint i = gsp; i < nr; i += {GSPLIT}) {{
                    float ah, al; two_sum(Za[i * {LDA} + gcol], -k, ah, al);
                    float vh = VH[i], vl = VL[i];
                    float p = ah * vh; gl += fma(ah, vh, -p) + (ah * vl + al * vh); two_sum_acc(gh, gl, p);
                }}
            }} else {{ for (uint i = gsp; i < nr; i += {GSPLIT}) {{ two_sum_acc(gh, gl, VH[i]); gl += VL[i]; }} }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) LAST[0] = RHO[nr - 1];
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < {R}) {{
        for (uint k = 0; k < 9; k++) {{ RED[(2 * k) * {R} + tid] = sh_[k]; RED[(2 * k + 1) * {R} + tid] = sl_[k]; }}
    }}
    if (gact) {{ GR[2 * tid] = gh; GR[2 * tid + 1] = gl; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ulong o = (ulong)g * {NOUT};
    uint base = 2 * ({P} + 1);
    if (tid < 9) {{
        float h = 0.0f, l = 0.0f;
        for (uint i = 0; i < {R}; i++) {{ two_sum_acc(h, l, RED[(2 * tid) * {R} + i]); l += RED[(2 * tid + 1) * {R} + i]; }}
        uint k = tid < 5 ? 2 * tid : 2 * tid + 3;          // slots 0-9: rho' powers and DW; 13-20: exact sums
        part[o + base + k] = h; part[o + base + k + 1] = l;
    }}
    if (tid <= {P}) {{
        float h = 0.0f, l = 0.0f;
        for (uint sp = 0; sp < {GSPLIT}; sp++) {{ uint t = sp * ({P} + 1) + tid; two_sum_acc(h, l, GR[2 * t]); l += GR[2 * t + 1]; }}
        part[o + 2 * tid] = h; part[o + 2 * tid + 1] = l;
    }}
    if (tid == 0) {{ part[o + base + 10] = first; part[o + base + 11] = LAST[0]; part[o + base + 12] = (float)(r1 > r0); }}
"""

_kernels = {}


def _config(p):
    lda = p | 1                                       # odd stride: a thread per row reads conflict-free
    fixed = 4 * (3 * p + 4 + 2 * NT)
    for r in ([128, 64, 32, 16, 8] if p <= 32 else [64, 32, 16, 8] if p <= 64 else [32, 16, 8]):
        if 4 * r * (lda + 21) + fixed <= TG_BUDGET:
            return r, lda
    raise ValueError(f"the residual pass does not fit threadgroup memory at {p} columns")


def _kernel(p, weighted, write_r):
    key = (p, weighted, write_r)
    if key in _kernels:
        return _kernels[key]
    if p + 1 > NT:
        raise ValueError(f"the residual pass supports up to {NT - 1} columns; got {p}")
    r, lda = _config(p)
    nout = 2 * (p + 1) + 21
    src = _SRC.format(R=r, LDA=lda, P=p, NT=NT, F4=r * p // 4, NOUT=nout, GSPLIT=max(1, NT // (p + 1)),
                      WLOAD="sv = S[rb + tid]; wv = W[rb + tid];" if weighted else "",
                      ROUT="rout[rb + tid] = rh;" if write_r else "")
    ins = ["X", "Y"] + (["S", "W"] if weighted else []) + ["shift", "coef", "meta", "rscale"]
    outs = ["part"] + (["rout"] if write_r else [])
    k = mx.fast.metal_kernel(name=f"mlxstats_resid2_p{p}_w{int(weighted)}_r{int(write_r)}", input_names=ins,
                             output_names=outs, source=src, header=_HEADER)
    _kernels[key] = (k, r, nout)
    return _kernels[key]


def split2(v):
    """float64 -> (hi, lo) float32 pairs with hi + lo = v to ~2^-48 relative."""
    v = np.asarray(v, np.float64)
    hi = v.astype(np.float32)
    lo = (v - hi.astype(np.float64)).astype(np.float32)
    return hi, lo


def resid_launch(X, y, s, w, shift, beta, c0, rscale=1.0, write_r=False):
    """Queue the residual pass over one MLX block. beta: (p,) float64 in shifted coordinates, c0: float64 constant
    in shifted coordinates, s = sqrt(w) and w the row weights (both or neither). Returns (part, rout, G, nout)."""
    n, p = X.shape
    k, r, nout = _kernel(p, w is not None, write_r)
    G = max(1, min(320, -(-n // (r * 4))))
    per = -(-n // G)
    per = -(-per // r) * r
    G = max(1, -(-n // per))
    bh, bl = split2(beta); ch, cl = split2(c0)
    coef = mx.array(np.concatenate([bh, bl, [ch], [cl]]).astype(np.float32))
    meta = mx.array(np.array([n & 0xffffffff, n >> 32, per & 0xffffffff, per >> 32], dtype=np.uint32))
    ins = [pad_min(X), pad_min(y)] + ([pad_min(s), pad_min(w)] if w is not None else []) + \
          [shift, coef, meta, mx.array(np.array([rscale], np.float32))]
    shapes = [(G * nout,)] + ([(max(n, 8),)] if write_r else [])
    outs = k(inputs=ins, grid=(G * NT, 1, 1), threadgroup=(NT, 1, 1), output_shapes=shapes,
             output_dtypes=[mx.float32] * len(shapes))
    return outs[0], (outs[1][:n] if write_r else None), G, nout


def resid_unpack(parts, p):
    """parts: list of (part, G, nout), one per block in row order -> (gradient (p+1), sums (9), first/last pairs).
    sums: rho' powers 1-4, Durbin-Watson (rho' units, boundaries included), sum w a_y^2, sum w a_y, sum w, sum w r^2."""
    grad = np.zeros(p + 1); sums = np.zeros(9)
    prev_last = None
    for part, G, nout in parts:
        a = np.array(part, dtype=np.float32).astype(np.float64).reshape(G, nout)
        grad += (a[:, 0:2 * (p + 1):2] + a[:, 1:2 * (p + 1):2]).sum(0)
        o = 2 * (p + 1)
        sums[:5] += (a[:, o:o + 10:2] + a[:, o + 1:o + 10:2]).sum(0)
        sums[5:9] += (a[:, o + 13:o + 21:2] + a[:, o + 14:o + 21:2]).sum(0)
        for gi in range(G):
            if a[gi, o + 12] == 0:
                continue
            if prev_last is not None:
                sums[4] += (a[gi, o + 10] - prev_last) ** 2
            prev_last = a[gi, o + 11]
    return grad, sums
