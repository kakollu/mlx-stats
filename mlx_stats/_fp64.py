"""float64 -> float32 on the GPU, bit-exact with NumPy's astype (round to nearest even, subnormals, overflow to inf).

NumPy's cast runs at ~21 GB/s on one core; copying the raw 8-byte patterns into MLX memory runs at ~69 GB/s, and the
GPU then rounds them at memory speed. Metal has no double type, so the rounding is done on the bit pattern with
integer arithmetic, which also makes it deterministic.
"""
import numpy as np
import mlx.core as mx

_SRC = """
    ulong e = thread_position_in_grid.x;
    if (e >= n[0]) return;
    uint lo = B[2 * e], hi = B[2 * e + 1];
    uint sign = hi & 0x80000000u;
    int ex = (int)((hi >> 20) & 0x7ffu);
    ulong mant = ((ulong)(hi & 0xfffffu) << 32) | (ulong)lo;
    uint bits;
    if (ex == 0x7ff) {
        bits = mant ? 0x7fc00000u : (sign | 0x7f800000u);                 // NaN, +-inf
    } else if (ex == 0) {
        bits = sign;                                                        // zero and float64 subnormals -> +-0
    } else {
        int e2 = ex - 1023;
        ulong m = mant | (1ul << 52);                                       // 53-bit significand
        if (e2 > 127) {
            bits = sign | 0x7f800000u;
        } else if (e2 >= -126) {                                            // normal float32: keep 24 bits
            ulong q = m >> 29, rem = m & ((1ul << 29) - 1), hlf = 1ul << 28;
            if (rem > hlf || (rem == hlf && (q & 1ul))) q++;
            if (q == (1ul << 24)) { q >>= 1; e2++; }
            bits = e2 > 127 ? (sign | 0x7f800000u) : (sign | ((uint)(e2 + 127) << 23) | (uint)(q & 0x7fffffu));
        } else {                                                            // float32 subnormal: units of 2^-149
            int sh = -e2 - 97;
            if (sh >= 64) { bits = sign; }
            else {
                ulong q = m >> sh, rem = m & ((1ul << sh) - 1), hlf = 1ul << (sh - 1);
                if (rem > hlf || (rem == hlf && (q & 1ul))) q++;
                bits = sign | (uint)q;                                      // a carry into bit 23 is the smallest normal
            }
        }
    }
    out[e] = as_type<float>(bits);
"""
_k = None


def bits_to_f32(bits, shape):
    """bits: MLX uint32 array holding float64 patterns (little-endian pairs) -> float32 MLX array of `shape`."""
    global _k
    if _k is None:
        _k = mx.fast.metal_kernel(name="mlxstats_f64_to_f32", input_names=["B", "n"], output_names=["out"], source=_SRC)
    n = int(np.prod(shape))
    out = _k(inputs=[bits, mx.array(np.array([n], np.uint64))], grid=(max(n, 1), 1, 1), threadgroup=(256, 1, 1),
             output_shapes=[(max(n, 1),)], output_dtypes=[mx.float32])[0]
    return out[:n].reshape(shape) if n else mx.zeros(shape, mx.float32)


def to_f32(a64):
    """a64: C-contiguous float64 numpy array -> float32 MLX array of the same shape (rounded on the GPU)."""
    global _k
    if _k is None:
        _k = mx.fast.metal_kernel(name="mlxstats_f64_to_f32", input_names=["B", "n"], output_names=["out"], source=_SRC)
    a64 = np.ascontiguousarray(a64, dtype=np.float64)
    n = a64.size
    if n == 0:
        return mx.zeros(a64.shape, mx.float32)
    bits = mx.array(a64.reshape(-1).view(np.uint32))
    out = _k(inputs=[bits, mx.array(np.array([n], np.uint64))], grid=(n, 1, 1), threadgroup=(256, 1, 1),
             output_shapes=[(n,)], output_dtypes=[mx.float32])[0]
    return out.reshape(a64.shape)
