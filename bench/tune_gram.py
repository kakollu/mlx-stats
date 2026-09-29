#!/usr/bin/env python3
"""Sweep the Gram kernel's launch shape per width: rows per stage R, simdgroups per threadgroup, 16x16 blocks per
simdgroup. Prints the best of each and the read floor. Usage: tune_gram.py [p,p,...]"""
import sys, time, itertools, numpy as np, mlx.core as mx
from mlx_stats import _gram
def best(f, r=5):
    f(); ts = []
    for _ in range(r):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return min(ts)
PS = [int(v) for v in sys.argv[1].split(",")] if len(sys.argv) > 1 else [4, 8, 16, 32, 48, 64, 100, 128, 200]
for p in PS:
    n = int(3e9 / (4 * (p + 1))) // 64 * 64
    X = mx.random.normal((n, p)); y = mx.random.normal((n,)); K = mx.zeros((p + 1,)); mx.eval(X, y, K)
    gb = n * (p + 1) * 4 / 1e9
    q = p + 2; q16 = (q + 15) // 16 * 16; ld = q16 + 4 if q16 % 32 == 0 else q16
    res = []
    for r, nsg, tps in itertools.product([16, 32, 64, 128], [4, 8, 16], [1, 2, 4]):
        if r * ld * 4 > 24576 or (tps > 1 and nsg * (tps - 1) >= (q16 // 16) * (q16 // 16 + 1) // 2):
            continue
        try:
            t = best(lambda: mx.eval(_gram.gram_launch(X, y, None, K, override=(r, nsg, tps))[0]))
        except Exception as e:
            print(f"  p={p} r={r} nsg={nsg} tps={tps}: {str(e)[:60]}"); continue
        res.append((t, r, nsg, tps))
    res.sort()
    t, r, nsg, tps = res[0]
    top = ", ".join(f"R{r2}/SG{n2}/B{t2} {gb/t2_:.0f}" for t2_, r2, n2, t2 in res[:4])
    print(f"p={p:>3} q16={q16:>3}: best R={r} SG={nsg} blocks={tps}: {t*1e3:6.2f} ms = {gb/t:4.0f} GB/s (floor {gb/0.5:5.2f} ms at 500) | top 4 GB/s: {top}", flush=True)
    del X, y
