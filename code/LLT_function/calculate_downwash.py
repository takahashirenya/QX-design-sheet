import numpy as np

_RAD = np.pi / 180.0
_DEG = 180.0 / np.pi


def calculation_downwash(Wing, state, circulation, Qij, cp, alpha_induced, wi):

    n2 = 2 * Wing["span_div"]

    # 吹きおろし速度 wi = (1/(4π)) * sum_j Qij(i,j) * circulation(j)
    # （行列積でベクトル化）
    wi[:] = (1.0 / (4.0 * np.pi)) * (Qij @ circulation)

    # 誘導迎角 αi[deg] = DEG * atan( wi / (Vair - RAD * r * cp_y) )
    denom = state["Vair"] - _RAD * state["r"] * cp[1, :]  # cp(1,i) ← 2行目
    alpha_induced[:] = _DEG * np.arctan(wi / denom)
    Wing["ai"] = alpha_induced.copy()

    # 平均吹きおろし角 epsilon
    # VBAは (2*n - 1) で割っていたので忠実に踏襲
    Wing["epsilon"] = float(alpha_induced.sum() / (n2 - 1))
