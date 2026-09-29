"""The exact Gram: sum w [a, a_y, 1][a, a_y, 1]^T with a = x - K and a_y = y - K_y carried as exact two-float values
(TwoSum), every product formed exactly (TwoProd) and multiplied by the row weight w exactly, and the terms added into
two-float accumulators. The result is the Gram of the float32 data as given, to ~1e-14 relative (measured against an
exact rational sum), so an exact linear relation among columns stays exact.

The matrix-unit kernel in `_gram` is accurate to ~3e-8 per entry, which decides a covariance to ~3e-8 / lambda_min of
the scaled design: plenty for well-conditioned designs, not for collinear ones. The fit switches to this kernel when the
fast Gram shows a scaled eigenvalue below EXACT_TOL, or when asked. It runs on the scalar units, several times slower
than the fast kernel for wide designs; for narrow ones it keeps up with memory.

Layout: 4 x 4 blocks of the upper triangle; each thread owns one block (16 two-float accumulators) and a share of the
staged rows; per-thread partials go to device memory and a second kernel sums them in two-float arithmetic.
"""
import numpy as np
import mlx.core as mx
from ._gram import _HEADER, pad_min

NT = 256
MAX_TG = 160
TG_BUDGET = 28 * 1024

_SRC = """
    uint tid = thread_position_in_threadgroup.x;
    uint gidx = threadgroup_position_in_grid.x;
    uint g = gidx / {GT}, gt = gidx % {GT};
    threadgroup float ZH[{R} * {LD}];
    threadgroup float ZL[{R} * {LD}];
    threadgroup float WT[{R}];
    threadgroup float SH[{LD}];
    for (uint j = tid; j < {LD}; j += {NT}) SH[j] = j < {NSHIFT} ? shift[j] : 0.0f;
    ulong n = (ulong)meta[0] | ((ulong)meta[1] << 32);
    ulong per = (ulong)meta[2] | ((ulong)meta[3] << 32);
    ulong r0 = (ulong)g * per, r1 = min(r0 + per, n);
    uint b = gt * {BPT} + tid % {BPT};
    uint sp = tid / {BPT};
    bool active = (tid < {BPT} * {NSPLIT}) && (b < {B4});
    uint bi = active ? BI[b] : 0, bj = active ? BJ[b] : 0;
    float h[16], l[16];
    #pragma unroll
    for (uint k = 0; k < 16; k++) {{ h[k] = 0.0f; l[k] = 0.0f; }}
    for (uint e = tid; e < {R} * {LD}; e += {NT}) {{ ZH[e] = 0.0f; ZL[e] = 0.0f; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (ulong rb = r0; rb < r1; rb += {R}) {{
        uint nr = (uint)min((ulong){R}, r1 - rb);
        for (uint e = tid; e < {R} * {P}; e += {NT}) {{
            uint i = e / {P}, j = e % {P};
            float ah = 0.0f, al = 0.0f;
            if (i < nr) two_sum(X[(rb + i) * {P} + j], -SH[j], ah, al);
            ZH[i * {LD} + j] = ah; ZL[i * {LD} + j] = al;
        }}
        if (tid < {R}) {{
            bool live = tid < nr;
            {YSTORE}
            ZH[tid * {LD} + {ONE}] = live ? 1.0f : 0.0f;
            WT[tid] = live ? {WVAL} : 0.0f;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {{
            for (uint i = sp; i < nr; i += {NSPLIT}) {{
                float4 uh = *(const threadgroup float4 *)(ZH + i * {LD} + 4 * bi);
                float4 ul = *(const threadgroup float4 *)(ZL + i * {LD} + 4 * bi);
                float4 vh = *(const threadgroup float4 *)(ZH + i * {LD} + 4 * bj);
                float4 vl = *(const threadgroup float4 *)(ZL + i * {LD} + 4 * bj);
                float w = WT[i];
                #pragma unroll
                for (uint a = 0; a < 4; a++) {{
                    #pragma unroll
                    for (uint c = 0; c < 4; c++) {{
                        float p = uh[a] * vh[c];
                        float pe = fma(uh[a], vh[c], -p) + (uh[a] * vl[c] + ul[a] * vh[c]);
                        {WMUL}
                        uint k = 4 * a + c;
                        float s = h[k] + p; float bb = s - h[k];
                        l[k] += ((h[k] - (s - bb)) + (p - bb)) + pe; h[k] = s;
                    }}
                }}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    if (active) {{
        ulong o = (((ulong)g * {B4} + b) * {NSPLIT} + sp) * 32;
        #pragma unroll
        for (uint k = 0; k < 16; k++) {{ part[o + 2 * k] = h[k]; part[o + 2 * k + 1] = l[k]; }}
    }}
"""

_REDUCE = """
    uint e = thread_position_in_grid.x;
    uint b = e / 16, k = e % 16;
    float h = 0.0f, l = 0.0f;
    uint G = meta[0], B4 = meta[1], NS = meta[2];
    for (uint g = 0; g < G; g++)
        for (uint s = 0; s < NS; s++) {
            ulong o = (((ulong)g * B4 + b) * NS + s) * 32 + 2 * k;
            two_sum_acc(h, l, part[o]); l += part[o + 1];
        }
    out[2 * e] = h; out[2 * e + 1] = l;
"""

_kernels = {}


def _kernel(p, has_y, has_w):
    key = (p, has_y, has_w)
    if key in _kernels:
        return _kernels[key]
    q = p + (1 if has_y else 0) + 1
    q4 = (q + 3) // 4 * 4
    T = q4 // 4
    bi, bj = [], []
    for i in range(T):
        for j in range(i, T):
            bi.append(i); bj.append(j)
    B4 = len(bi)
    bpt = min(B4, NT)
    gt = -(-B4 // bpt)
    nsplit = max(1, NT // bpt)
    ld = q4
    R = next((r for r in (64, 32, 16, 8) if 4 * (2 * r * ld + r + ld) <= TG_BUDGET), None)
    if R is None:
        raise ValueError(f"the exact Gram does not fit threadgroup memory at {p} columns")
    header = _HEADER + (f"constant ushort BI[{B4}] = {{{', '.join(map(str, bi))}}};\n"
                        f"constant ushort BJ[{B4}] = {{{', '.join(map(str, bj))}}};\n")
    src = _SRC.format(R=R, LD=ld, NT=NT, P=p, B4=B4, BPT=bpt, GT=gt, NSPLIT=nsplit, ONE=q - 1,
                      NSHIFT=p + (1 if has_y else 0),
                      WVAL="W[rb + tid]" if has_w else "1.0f",
                      WMUL="float wp = w * p; pe = fma(w, p, -wp) + w * pe; p = wp;" if has_w else "",
                      YSTORE=(f"float yh = 0.0f, yl = 0.0f; if (live) two_sum(Y[rb + tid], -SH[{p}], yh, yl); "
                              f"ZH[tid * {ld} + {p}] = yh; ZL[tid * {ld} + {p}] = yl;") if has_y else "")
    ins = ["X"] + (["Y"] if has_y else []) + (["W"] if has_w else []) + ["shift", "meta"]
    k = mx.fast.metal_kernel(name=f"mlxstats_gramx2_p{p}_y{int(has_y)}_w{int(has_w)}", input_names=ins,
                             output_names=["part"], source=src, header=header)
    if "reduce" not in _kernels:
        _kernels["reduce"] = mx.fast.metal_kernel(name="mlxstats_gramx_reduce", input_names=["part", "meta"],
                                                  output_names=["out"], source=_REDUCE, header=_HEADER)
    _kernels[key] = (k, q, q4, bi, bj, B4, nsplit, gt, R)
    return _kernels[key]


def gram_exact_launch(X, y=None, w=None, shift=None):
    """Queue the exact Gram of one MLX block; w: row weights (not their square roots) or None."""
    n, p = X.shape
    k, q, q4, bi, bj, B4, nsplit, gt, R = _kernel(p, y is not None, w is not None)
    G = max(1, min(max(1, MAX_TG // gt), -(-n // (R * 4))))
    per = -(-n // G)
    per = -(-per // R) * R
    G = max(1, -(-n // per))
    meta = mx.array(np.array([n & 0xffffffff, n >> 32, per & 0xffffffff, per >> 32], dtype=np.uint32))
    ins = [pad_min(X)] + ([pad_min(y)] if y is not None else []) + ([pad_min(w)] if w is not None else []) + [shift, meta]
    part = k(inputs=ins, grid=(G * gt * NT, 1, 1), threadgroup=(NT, 1, 1), output_shapes=[(G * B4 * nsplit * 32,)],
             output_dtypes=[mx.float32], init_value=0.0)[0]
    red = _kernels["reduce"](inputs=[part, mx.array(np.array([G, B4, nsplit], np.uint32))], grid=(B4 * 16, 1, 1),
                             threadgroup=(min(256, B4 * 16), 1, 1), output_shapes=[(B4 * 32,)], output_dtypes=[mx.float32])[0]
    return red, (q, q4, bi, bj, B4)


def unpack_exact(red, layout):
    q, q4, bi, bj, B4 = layout
    r = np.array(red, dtype=np.float32).astype(np.float64).reshape(B4, 16, 2)
    blocks = (r[..., 0] + r[..., 1]).reshape(B4, 4, 4)
    G = np.zeros((q4, q4))
    for u in range(B4):
        i, j = bi[u], bj[u]
        G[4 * i:4 * i + 4, 4 * j:4 * j + 4] = blocks[u]
        G[4 * j:4 * j + 4, 4 * i:4 * i + 4] = blocks[u].T
    return G[:q, :q]
