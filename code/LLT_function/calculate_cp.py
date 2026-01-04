import numpy as np


def calculation_cp(Wing, state, cp, y, z):

    n = int(Wing["span_div"])

    # 軽い形状チェック（必要なら外してOK）
    if cp.shape != (3, 2 * n):
        raise ValueError(f"cp shape must be (3, {2*n}), got {cp.shape}")
    if y.shape[0] != 2 * n + 1 or z.shape[0] != 2 * n + 1:
        raise ValueError(f"y,z length must be {2*n+1}")

    for i in range(n):
        # 右翼: index n+i と n+i+1 の中点
        ri = n + i
        cp[0, ri] = 0.0
        cp[1, ri] = 0.5 * (y[ri] + y[ri + 1])
        cp[2, ri] = 0.5 * (z[ri] + z[ri + 1])

        # 左翼: index (n-1-i) と (n-i) の中点
        li = n - 1 - i
        cp[0, li] = 0.0
        cp[1, li] = 0.5 * (y[li] + y[li + 1])
        cp[2, li] = 0.5 * (z[li] + z[li + 1])
