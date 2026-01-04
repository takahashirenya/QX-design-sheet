# culculate_phi_theta.py
from math import pi, isfinite
from typing import Sequence, Any
import numpy as np


def calculation_phi_theta(
    wing: Any,  # dict / dataclass いずれも可（span_div, dy を持つ）
    state: Any,  # 未使用（互換のため）
    deflection: Sequence[float],  # 長さ 2n+1
    theta: Sequence[float],  # 長さ 2n+1 （関数内でrad計算→degで上書き）
    phi: Sequence[float],  # 長さ 2n+1 （関数内でrad計算→degで上書き）
    bending_moment: Sequence[float],  # 長さ 2n でも 2n+1 でもOK
    torque: Sequence[float],  # 長さ 2n でも 2n+1 でもOK
    eix: Sequence[float],  # 長さ 2n
    gj: Sequence[float],  # 長さ 2n
) -> None:
    # --- 取り出し（dict/dataclass対応） ---
    if isinstance(wing, dict):
        n = int(wing["span_div"])
        dy = wing["dy"]
    else:
        n = int(getattr(wing, "span_div"))
        dy = getattr(wing, "dy")

    nodes = 2 * n + 1
    pans = 2 * n

    # --- dy を関数化（スカラー/配列両対応） ---
    if isinstance(dy, (list, tuple, np.ndarray)):
        if len(dy) != pans:
            raise ValueError(
                f"wing['dy'] length mismatch: expected {pans}, got {len(dy)}"
            )

        def dL(i):
            return float(dy[i])

        def dR(i):
            return float(dy[i])

    else:
        dy_s = float(dy)

        def dL(i):
            return dy_s

        def dR(i):
            return dy_s

    # --- 入力サイズざっくりチェック ---
    for name, arr, exp in [
        ("deflection", deflection, nodes),
        ("theta", theta, nodes),
        ("phi", phi, nodes),
        ("eix", eix, pans),
        ("gj", gj, pans),
    ]:
        if len(arr) != exp:
            raise ValueError(f"{name} length mismatch: expected {exp}, got {len(arr)}")

    # --- BM/TQ を右端アクセス(num2+1)に耐えるよう拡張ビューを作る ---
    #     入力が 2n   : [0..2n-1] -> 末尾を複写して [0..2n] を作る
    #     入力が 2n+1 : そのまま使う
    def _mk_plus1(a, name):
        a = np.asarray(a, dtype=float)
        if a.size == pans:
            out = np.empty(pans + 1, dtype=float)
            out[:pans] = a
            out[pans] = a[-1]  # 末尾複写（右端の外挿）
            return out
        elif a.size == pans + 1:
            return a
        else:
            raise ValueError(
                f"{name} length mismatch: expected {pans} or {pans+1}, got {a.size}"
            )

    bm_ext = _mk_plus1(bending_moment, "bending_moment")
    tq_ext = _mk_plus1(torque, "torque")

    # --- 安全div: 0/NaN/Inf を検知して寄与を0にする ---
    def safe_div(num, den):
        if not (isfinite(num) and isfinite(den)) or den == 0.0:
            return 0.0
        return num / den

    # --- 積分（計算中はrad、最後にdegへ上書き） ---
    for i in range(n):
        num1 = n - 1 - i  # 左パネル index
        num2 = n + i  # 右パネル index

        d_left = dL(num1)
        d_right = dR(num2)

        # θ [rad]
        theta[num1] = theta[num1 + 1] + safe_div(bm_ext[num1], eix[num1]) * d_left
        theta[num2 + 1] = (
            theta[num2] + safe_div(bm_ext[num2 + 1], eix[num2]) * d_right
        )  # 右は num2+1

        # w [m]
        deflection[num1] = deflection[num1 + 1] + theta[num1 + 1] * d_left
        deflection[num2 + 1] = deflection[num2] + theta[num2] * d_right

        # Φ [rad]
        phi[num1] = phi[num1 + 1] + safe_div(tq_ext[num1] * d_left, gj[num1])
        phi[num2 + 1] = phi[num2] + safe_div(tq_ext[num2 + 1] * d_right, gj[num2])

    # --- [rad]→[deg] にインプレース変換 ---
    deg = 180.0 / pi
    for k in range(nodes):
        theta[k] *= deg
        phi[k] *= deg
