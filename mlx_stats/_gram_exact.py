"""The exact Gram: the same sum s^2 [a, a_y, 1][a, a_y, 1]^T as `_gram`, with every product formed exactly (TwoProd)
and added into a two-float accumulator, so the result carries ~1e-14 relative error instead of float32 rounding.

The matrix-unit kernel in `_gram` is accurate to ~3e-8 per entry, which decides a covariance to ~3e-8 / lambda_min of
the scaled design: plenty for well-conditioned designs, not for collinear ones (a pair of columns correlated at
0.99995 put 5e-4 into the standard errors). The fit switches to this kernel when the fast Gram shows a scaled
eigenvalue below 1e-4, or when asked. It runs on the scalar units: about 9 operations per product, so it keeps up
with memory for narrow designs and is a few times slower than the fast kernel for 60+ columns.

Layout: 4 x 4 blocks of the upper triangle; each thread owns one block (16 two-float accumulators) and a share of the
staged rows; per-thread partials go to device memory and a second kernel sums them in two-float arithmetic.
"""
import numpy as np
import mlx.core as mx
from ._gram import _HEADER

NT = 256
MAX_TG = 160

_SRC = """
    uint tid = thread_position_in_threadgroup.x;
    uint gidx = threadgroup_position_in_grid.x;
    uint g = gidx / {GT}, gt = gidx % {GT};
    threadgroup float Z[{R} * {LD}];
    threadgroup float SH[{LD}];
    for (uint j = tid; j < {LD}; j += {NT}) SH[j] = j < {NSHIFT} ? shift[j] : 0.0f;
    ulong n = (ulong)meta[0] | ((ulong)meta[1] << 32);
    ulong per = (ulong)meta[2] | ((ulong)meta[3] << 32);
    ulong r0 = (ulong)g * per, r1 = min(r0 + per, n);
    uint b = gt * {BPT} + tid % {BPT};                     // block owned by this thread
    uint sp = tid / {BPT};                                  // which share of the staged rows
    bool active = (tid < {BPT} * {NSPLIT}) && (b < {B4});
    uint bi = active ? BI[b] : 0, bj = active ? BJ[b] : 0;
    float h[16], l[16];
    #pragma unroll
    for (uint k = 0; k < 16; k++) {{ h[k] = 0.0f; l[k] = 0.0f; }}
    for (uint e = tid; e < {R} * {LD}; e += {NT}) Z[e] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (ulong rb = r0; rb < r1; rb += {R}) {{
        uint nr = (uint)min((ulong){R}, r1 - rb);
        for (uint e = tid; e < {R} * {P}; e += {NT}) {{
            uint i = e / {P}, j = e % {P};
            Z[i * {LD} + j] = i < nr ? X[(rb + i) * {P} + j] - SH[j] : 0.0f;
        }}
        if (tid < {R}) {{
            bool live = tid < nr;
            float sc = live ? {SVAL} : 0.0f;
            {YSTORE}
            Z[tid * {LD} + {ONE}] = sc;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {SCALEPASS}
        if (active) {{
            for (uint i = sp; i < nr; i += {NSPLIT}) {{
                float4 u = *(const threadgroup float4 *)(Z + i * {LD} + 4 * bi);
                float4 v = *(const threadgroup float4 *)(Z + i * {LD} + 4 * bj);
                #pragma unroll
                for (uint a = 0; a < 4; a++) {{
                    #pragma unroll
                    for (uint c = 0; c < 4; c++) {{
                        float x = u[a], y = v[c];
                        float p = x * y; float pe = fma(x, y, -p);
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
    uint e = thread_position_in_grid.x;            // e < B4 * 16
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


def _kernel(p, has_y, has_s):
    key = (p, has_y, has_s)
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
    bpt = min(B4, NT)                                   # blocks per threadgroup
    gt = -(-B4 // bpt)                                  # threadgroups splitting the blocks over the same rows
    nsplit = max(1, NT // bpt)                          # threads sharing one block's rows
    R = 64 if q4 <= 64 else 32
    ld = q4
    header = _HEADER + (f"constant ushort BI[{B4}] = {{{', '.join(map(str, bi))}}};\n"
                        f"constant ushort BJ[{B4}] = {{{', '.join(map(str, bj))}}};\n")
    src = _SRC.format(R=R, LD=ld, NT=NT, P=p, B4=B4, BPT=bpt, GT=gt, NSPLIT=nsplit, ONE=q - 1,
                      NSHIFT=p + (1 if has_y else 0),
                      SVAL="S[rb + tid]" if has_s else "1.0f",
                      YSTORE=f"Z[tid * {ld} + {p}] = live ? Y[rb + tid] - SH[{p}] : 0.0f;" if has_y else "",
                      SCALEPASS=(f"if (tid < {R}) {{ float sc = Z[tid * {ld} + {q - 1}]; for (uint j = 0; j < {q - 1}; j++) Z[tid * {ld} + j] *= sc; }}\n"
                                 "        threadgroup_barrier(mem_flags::mem_threadgroup);") if has_s else "")
    ins = ["X"] + (["Y"] if has_y else []) + (["S"] if has_s else []) + ["shift", "meta"]
    k = mx.fast.metal_kernel(name=f"mlxstats_gramx_p{p}_y{int(has_y)}_s{int(has_s)}", input_names=ins,
                             output_names=["part"], source=src, header=header)
    red = _kernels.get("reduce")
    if red is None:
        red = _kernels["reduce"] = mx.fast.metal_kernel(name="mlxstats_gramx_reduce", input_names=["part", "meta"],
                                                        output_names=["out"], source=_REDUCE, header=_HEADER)
    _kernels[key] = (k, q, q4, bi, bj, B4, nsplit, gt, R)
    return _kernels[key]


def gram_exact_launch(X, y=None, s=None, shift=None):
    n, p = X.shape
    k, q, q4, bi, bj, B4, nsplit, gt, R = _kernel(p, y is not None, s is not None)
    G = max(1, min(max(1, MAX_TG // gt), -(-n // (R * 4))))
    per = -(-n // G)
    per = -(-per // R) * R
    G = max(1, -(-n // per))
    meta = mx.array(np.array([n & 0xffffffff, n >> 32, per & 0xffffffff, per >> 32], dtype=np.uint32))
    ins = [X] + ([y] if y is not None else []) + ([s] if s is not None else []) + [shift, meta]
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
