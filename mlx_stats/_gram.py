"""The one-pass Gram kernel: G = sum_r s_r^2 a_r a_r^T with a_r = [x_r - K_x, y_r - K_y, 1], on the GPU.

Everything a linear model needs is in that (k+2) x (k+2) matrix: X^T X, X^T y, y^T y, the column sums and the row
count (the last column, which is the constant 1), all taken around a shift K close to the column means so the
float32 arithmetic works on centred numbers. The host turns it into float64 statistics.

How it runs. Each threadgroup owns a contiguous range of rows and walks it in stages of R rows: the stage is copied
into threadgroup memory (shifted, scaled, the ones column appended, padded to a multiple of 8 columns), then each
simdgroup multiplies the 8 x 8 tiles it owns on the upper triangle with the matrix unit (`simdgroup_multiply_
accumulate`, one rounding per step). After every stage each lane folds its tile elements into a two-float sum
(hi, lo) with an exact TwoSum, so the float32 accumulation only ever spans R rows: the error per entry is about
float32 rounding times sqrt(R), not times sqrt(N). The per-threadgroup (hi, lo) partials are summed by a second
small kernel in two-float arithmetic and read back once.

Cost: one read of X (and y, and the scale vector if given). The multiply is k^2 / 2 per row on the matrix unit,
which stays under the read time up to roughly k = 64-100 on an M5 Max (15.7 TFLOP/s against ~500 GB/s).
"""
import numpy as np
import mlx.core as mx

MAX_TG = 320      # threadgroups for large inputs: 40 cores x 8

_HEADER = """
inline void two_sum_acc(thread float &h, thread float &l, float x) {
    float s = h + x; float bb = s - h; l += (h - (s - bb)) + (x - bb); h = s;
}
inline void two_sum(float a, float b, thread float &s, thread float &e) {
    s = a + b; float bb = s - a; e = (a - (s - bb)) + (b - bb);
}
inline void two_sum_acc2(thread float2 &h, thread float2 &l, float2 x) {
    float2 s = h + x; float2 bb = s - h; l += (h - (s - bb)) + (x - bb); h = s;
}
"""

_SRC = """
    uint tid = thread_position_in_threadgroup.x;
    uint gidx = threadgroup_position_in_grid.x;
    uint g = gidx / {GT}, gt = gidx % {GT};            // row range, tile group (consecutive threadgroups share rows)
    uint sg = simdgroup_index_in_threadgroup;
    threadgroup float Z[{R} * {LD}];
    threadgroup float SH[{Q16}];
    for (uint j = tid; j < {Q16}; j += {NT}) SH[j] = j < {NSHIFT} ? shift[j] : 0.0f;
    ulong n = (ulong)meta[0] | ((ulong)meta[1] << 32);
    ulong per = (ulong)meta[2] | ((ulong)meta[3] << 32);
    ulong r0 = (ulong)g * per, r1 = min(r0 + per, n);
    simdgroup_float8x8 acc[4 * {TPS}];
    float2 hi[4 * {TPS}], lo[4 * {TPS}];
    #pragma unroll
    for (uint t = 0; t < 4 * {TPS}; t++) {{ acc[t] = simdgroup_float8x8(0.0f); hi[t] = float2(0.0f); lo[t] = float2(0.0f); }}
    const device packed_float4 *X4 = (const device packed_float4 *)X;     // 4-byte alignment is enough
    for (uint e = tid; e < {R} * {LD}; e += {NT}) Z[e] = (e % {LD}) == {ONE} ? 1.0f : 0.0f;
    float4 pf[{NPF}]; float yv = 0.0f, sv = 1.0f;
    ulong rb = r0;
    if (rb + {R} <= r1) {{
        #pragma unroll
        for (uint k = 0; k < {NPF}; k++) {{ uint f = tid + k * {NT}; if (f < {F4}) pf[k] = float4(X4[(rb >> 2) * {P} + f]); }}
    }}
    if (tid < {R} && rb + tid < r1) {{ {YPRE} {SPRE} }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (; rb < r1; rb += {R}) {{
        uint nr = (uint)min((ulong){R}, r1 - rb);
        if (nr == {R}) {{
            #pragma unroll
            for (uint k = 0; k < {NPF}; k++) {{
                uint f = tid + k * {NT};
                if (f < {F4}) {{
                    #pragma unroll
                    for (uint c = 0; c < 4; c++) {{ uint e = 4 * f + c, i = e / {P}, j = e % {P}; Z[i * {LD} + j] = pf[k][c] - SH[j]; }}
                }}
            }}
        }} else {{
            for (uint e = tid; e < {R} * {P}; e += {NT}) {{
                uint i = e / {P}, j = e % {P};
                Z[i * {LD} + j] = i < nr ? X[(rb + i) * {P} + j] - SH[j] : 0.0f;
            }}
        }}
        if (tid < {R}) {{
            bool live = tid < nr;
            {YSTORE}
            {ONESTORE}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {SCALEPASS}
        ulong rn = rb + {R};
        if (rn + {R} <= r1) {{
            #pragma unroll
            for (uint k = 0; k < {NPF}; k++) {{ uint f = tid + k * {NT}; if (f < {F4}) pf[k] = float4(X4[(rn >> 2) * {P} + f]); }}
        }}
        if (tid < {R} && rn + tid < r1) {{ {YNEXT} {SNEXT} }}
        // 2 x 2 blocks of 8 x 8 tiles: four loads feed four multiplies
        #pragma unroll
        for (uint kb = 0; kb < {R} / 8; kb++) {{
            const threadgroup float *base = Z + kb * 8 * {LD};
            #pragma unroll
            for (uint t = 0; t < {TPS}; t++) {{
                uint u = gt * {NSG} * {TPS} + sg + t * {NSG};
                if (u < {U}) {{
                    simdgroup_float8x8 a0, a1, b0, b1;
                    simdgroup_load(a0, base + BI[u] * 16, {LD}, ulong2(0, 0), true);
                    simdgroup_load(a1, base + BI[u] * 16 + 8, {LD}, ulong2(0, 0), true);
                    simdgroup_load(b0, base + BJ[u] * 16, {LD});
                    simdgroup_load(b1, base + BJ[u] * 16 + 8, {LD});
                    simdgroup_multiply_accumulate(acc[4 * t + 0], a0, b0, acc[4 * t + 0]);
                    simdgroup_multiply_accumulate(acc[4 * t + 1], a0, b1, acc[4 * t + 1]);
                    simdgroup_multiply_accumulate(acc[4 * t + 2], a1, b0, acc[4 * t + 2]);
                    simdgroup_multiply_accumulate(acc[4 * t + 3], a1, b1, acc[4 * t + 3]);
                }}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint t = 0; t < {TPS}; t++) {{
            if (gt * {NSG} * {TPS} + sg + t * {NSG} < {U}) {{
                #pragma unroll
                for (uint c = 0; c < 4; c++) {{
                    auto el = acc[4 * t + c].thread_elements();
                    two_sum_acc2(hi[4 * t + c], lo[4 * t + c], float2(el[0], el[1]));
                    acc[4 * t + c] = simdgroup_float8x8(0.0f);
                }}
            }}
        }}
    }}
    #pragma unroll
    for (uint t = 0; t < {TPS}; t++) {{
        uint u = gt * {NSG} * {TPS} + sg + t * {NSG};
        if (u < {U}) {{
            #pragma unroll
            for (uint c = 0; c < 4; c++) {{
                simdgroup_float8x8 m;
                ulong o = (((ulong)g * {U} + u) * 4 + c) * 128;
                m.thread_elements()[0] = hi[4 * t + c].x; m.thread_elements()[1] = hi[4 * t + c].y;
                simdgroup_store(m, part + o, 8);
                m.thread_elements()[0] = lo[4 * t + c].x; m.thread_elements()[1] = lo[4 * t + c].y;
                simdgroup_store(m, part + o + 64, 8);
            }}
        }}
    }}
"""

# sum the per-row-range (hi, lo) partials in two-float arithmetic: one thread per output element
_REDUCE = """
    uint e = thread_position_in_grid.x;            // e < U * 4 * 64
    uint b = e / 64, w = e % 64;                   // b = 8 x 8 tile index (super-tile * 4 + c)
    float h = 0.0f, l = 0.0f;
    uint stride = nblocks[0] * 128;
    for (uint g = 0; g < ngroups[0]; g++) {
        ulong o = (ulong)g * stride + (ulong)b * 128 + w;
        two_sum_acc(h, l, part[o]);
        l += part[o + 64];
    }
    out[2 * e] = h; out[2 * e + 1] = l;
"""

_kernels = {}


def pad_min(a, k=8):
    """MLX binds an input with fewer than 8 elements in the `constant` address space, which the kernels' vector loads
    cannot read; pad such inputs with zero rows (the kernels never read past n, which is passed separately)."""
    if a is None or a.size >= k:
        return a
    if a.ndim == 1:
        return mx.concatenate([a, mx.zeros((k - a.size,), a.dtype)])
    cols = max(1, a.shape[1])
    extra = -(-(k - a.size) // cols)
    return mx.concatenate([a, mx.zeros((extra, a.shape[1]), a.dtype)], axis=0)


# Launch shape by padded width q16: (rows per stage, simdgroups per threadgroup, 16x16 blocks per simdgroup).
# From bench/tune_gram.py on an M5 Max, 2026-09-28 (3 GB inputs, best of 5):
#   p=8 410 GB/s, p=16 386, p=32 356, p=48 268, p=64 293, p=100 192, p=128 156, p=200 85 (read floor ~500).
def _tuned(q16):
    if q16 <= 32:
        return 128, 4, 1
    if q16 <= 48:
        return 128, 4, 2
    if q16 <= 80:
        return 64, 8, 2
    if q16 <= 112:
        return 32, 16, 2
    if q16 <= 144:
        return 32, 8, 2
    return 16, 16, 2


def _config(p, has_y, override=None):
    q = p + (1 if has_y else 0) + 1                  # x columns, y, the constant
    q16 = (q + 15) // 16 * 16
    T2 = q16 // 16
    bi, bj = [], []
    for i in range(T2):
        for j in range(i, T2):
            bi.append(i); bj.append(j)
    U = len(bi)
    r, nsg, tps = override if override is not None else _tuned(q16)
    ld = q16 + 4 if q16 % 32 == 0 else q16
    while r > 8 and 4 * (r * ld + q16) > 28 * 1024:    # stay inside threadgroup memory for very wide designs
        r //= 2
    tps = min(tps, -(-U // nsg))
    gt = -(-U // (nsg * tps))                         # threadgroups that split the tile set over the same rows
    return q, q16, T2, bi, bj, U, nsg, tps, gt, r


def _gram_kernel(p, has_y, has_s, override=None):
    key = (p, has_y, has_s, override)
    if key in _kernels:
        return _kernels[key]
    q, q16, T2, bi, bj, U, nsg, tps, gt, r = _config(p, has_y, override)
    nt = 32 * nsg
    header = _HEADER + (f"constant ushort BI[{U}] = {{{', '.join(map(str, bi))}}};\n"
                        f"constant ushort BJ[{U}] = {{{', '.join(map(str, bj))}}};\n")
    ld = q16 + 4 if q16 % 32 == 0 else q16            # a stride that is a multiple of 32 floats puts a tile's rows on one bank
    f4 = r * p // 4                                   # r is a multiple of 4, so a full stage is whole float4s
    npf = max(1, -(-f4 // nt))
    src = _SRC.format(R=r, Q16=q16, LD=ld, TPS=tps, NT=nt, P=p, F4=f4, NPF=npf, U=U, NSG=nsg, GT=gt,
                      ONE=q - 1, NSHIFT=p + (1 if has_y else 0),
                      YPRE="yv = Y[rb + tid];" if has_y else "",
                      SPRE="sv = S[rb + tid];" if has_s else "",
                      YNEXT="yv = Y[rn + tid];" if has_y else "",
                      SNEXT="sv = S[rn + tid];" if has_s else "",
                      YSTORE=f"Z[tid * {ld} + {p}] = live ? yv - SH[{p}] : 0.0f;" if has_y else "",
                      ONESTORE=(f"Z[tid * {ld} + {q - 1}] = live ? 1.0f : 0.0f;" if has_s else
                                f"if (!live) Z[tid * {ld} + {q - 1}] = 0.0f;"),
                      SCALEPASS=(f"if (tid < {r}) {{ float sc = tid < nr ? sv : 0.0f; for (uint j = 0; j < {q}; j++) Z[tid * {ld} + j] *= sc; }}\n"
                                 "        threadgroup_barrier(mem_flags::mem_threadgroup);") if has_s else "")
    ins = ["X"] + (["Y"] if has_y else []) + (["S"] if has_s else []) + ["shift", "meta"]
    tag = "" if override is None else "_r%d_g%d_t%d" % override
    k = mx.fast.metal_kernel(name=f"mlxstats_gram_p{p}_y{int(has_y)}_s{int(has_s)}{tag}", input_names=ins,
                             output_names=["part"], source=src, header=header)
    _kernels[key] = (k, q, q16, T2, bi, bj, U, nt, gt, r)
    return _kernels[key]


def _reduce_kernel():
    if "reduce" not in _kernels:
        _kernels["reduce"] = mx.fast.metal_kernel(name="mlxstats_gram_reduce", input_names=["part", "ngroups", "nblocks"],
                                                  output_names=["out"], source=_REDUCE, header=_HEADER)
    return _kernels["reduce"]


def gram_launch(X, y=None, s=None, shift=None, override=None):
    """Queue the Gram of one MLX block. X: (n, p) float32 MLX, row-contiguous and 16-byte aligned; y, s: (n,)
    float32 MLX or None; shift: (p [+1],) float32. Returns an MLX array of (hi, lo) pairs, not yet evaluated, and
    the layout needed to unpack it."""
    n, p = X.shape
    k, q, q16, T2, bi, bj, U, nt, gt, r = _gram_kernel(p, y is not None, s is not None, override)
    G = max(1, min(MAX_TG // gt if gt < MAX_TG else 1, -(-n // (r * 4))))    # at least 4 stages per threadgroup
    per = -(-n // G)
    per = -(-per // r) * r
    G = max(1, -(-n // per))
    meta = mx.array(np.array([n & 0xffffffff, n >> 32, per & 0xffffffff, per >> 32], dtype=np.uint32))
    ins = [pad_min(X)] + ([pad_min(y)] if y is not None else []) + ([pad_min(s)] if s is not None else []) + [shift, meta]
    part = k(inputs=ins, grid=(G * gt * nt, 1, 1), threadgroup=(nt, 1, 1), output_shapes=[(G * U * 4 * 128,)],
             output_dtypes=[mx.float32], init_value=0.0)[0]
    red = _reduce_kernel()(inputs=[part, mx.array(np.array([G], np.uint32)), mx.array(np.array([U * 4], np.uint32))],
                           grid=(U * 4 * 64, 1, 1), threadgroup=(64, 1, 1), output_shapes=[(U * 4 * 128,)],
                           output_dtypes=[mx.float32])[0]
    return red, (q, q16, T2, bi, bj, U)


def unpack(red, layout):
    """(hi, lo) tiles -> the full symmetric q x q Gram in float64."""
    q, q16, T2, bi, bj, U = layout
    r = np.array(red, dtype=np.float32).astype(np.float64).reshape(U, 4, 64, 2)
    tiles = (r[..., 0] + r[..., 1]).reshape(U, 4, 8, 8)
    G = np.zeros((q16, q16))
    for u in range(U):
        for c in range(4):
            i, j = 2 * bi[u] + c // 2, 2 * bj[u] + c % 2
            if i <= j:
                G[i * 8:i * 8 + 8, j * 8:j * 8 + 8] = tiles[u, c]
                G[j * 8:j * 8 + 8, i * 8:i * 8 + 8] = tiles[u, c].T
    return G[:q, :q]
