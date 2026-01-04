import numpy as np
from dataclasses import dataclass


@dataclass
class Specifications:
    span_div: int  # = n


@dataclass
class Variables:
    # 必要ならここに状態量を追加（VBAでも使っていないのでダミー）
    pass


def calculation_moment(
    Wing: Specifications,
    state: Variables,
    dN: np.ndarray,  # shape: (2*n,)
    dT: np.ndarray,  # shape: (2*n,)
    dM_cg: np.ndarray,  # shape: (2*n,)
    dW: np.ndarray,  # shape: (2*n,)
    dihedral_angle: np.ndarray,  # shape: (2*n,) [deg または rad*angle を下の rad で処理]
    cp: np.ndarray,  # shape: (2, 2*n)  cp[0,:]=y座標, cp[1,:]=z座標
    y: np.ndarray,  # shape: (2*n,)
    z: np.ndarray,  # shape: (2*n,)
    rad: float = np.pi
    / 180.0,  # 角度が度のときは既定値のまま、既にラジアンなら 1.0 を渡す
):
    """
    VBAの Sub calculation_Moment(...) をNumPyで忠実移植。
    返り値: Bending_Moment, Bending_Moment_T, Shear_Force, Torque （各 shape: (2*n,)）
    """
    n = Wing["span_div"]
    size = 2 * n  # インデックスは 0..2n-1 を想定（VBAのReDim(2*n)と同じ使用範囲）

    # 出力配列を0初期化
    Bending_Moment = np.zeros(size, dtype=float)
    Bending_Moment_T = np.zeros(size, dtype=float)
    Shear_Force = np.zeros(size, dtype=float)
    Torque = np.zeros(size, dtype=float)

    # 主要ループ（VBAは i=1..n, j=1..i）
    for i in range(1, n + 1):  # 1..n
        num1 = 2 * n - i  # 右翼側の対称インデックス（0..2n-1 の範囲に入る）

        for j in range(1, i + 1):  # 1..i
            num2 = 2 * n - j  # 右翼側の“荷重位置”インデックス

            # 左翼側の寄与（j-1 を参照）
            idxL = j - 1
            # 右翼側の寄与（num2 を参照）
            idxR = num2

            # 上反角（度→ラジアン換算を rad でコントロール）
            cosL = np.cos(rad * dihedral_angle[idxL])
            sinAbsL = np.sin(np.abs(rad * dihedral_angle[idxL]))
            cosR = np.cos(rad * dihedral_angle[idxR])
            sinAbsR = np.sin(np.abs(rad * dihedral_angle[idxR]))

            # y, z 距離（絶対値）
            dy_L = np.abs(cp[0, idxL] - y[i])
            dz_L = np.abs(cp[1, idxL] - z[i])
            dy_R = np.abs(cp[0, idxR] - y[num1])
            dz_R = np.abs(cp[1, idxR] - z[num1])

            # 曲げモーメント（Bending_Moment）
            Bending_Moment[i] += (dN[idxL] * cosL - dW[idxL]) * dy_L + dN[
                idxL
            ] * sinAbsL * dz_L  # 左翼
            Bending_Moment[num1] += (dN[idxR] * cosR - dW[idxR]) * dy_R + dN[
                idxR
            ] * sinAbsR * dz_R  # 右翼

            # T曲げモーメント（Bending_Moment_T）: ヨー方向の曲げ
            Bending_Moment_T[i] += dT[idxL] * dy_L
            Bending_Moment_T[num1] += dT[idxR] * dy_R

            # トルク（Torque）: 回りモーメント
            Torque[i] += dM_cg[idxL] + dT[idxL] * (cp[1, idxL] - z[i])
            Torque[num1] += dM_cg[idxR] + dT[idxR] * (cp[1, idxR] - z[num1])

            # せん断力（Shear_Force）
            Shear_Force[i] += dN[idxL] * cosL - dW[idxL]
            Shear_Force[num1] += dN[idxR] * cosR - dW[idxR]

    # 中央（翼根）位置は左右で二重加算になるので 1/2（VBAと同じ処理）
    Bending_Moment[n] *= 0.5
    Bending_Moment_T[n] *= 0.5
    Torque[n] *= 0.5
    Shear_Force[n] *= 0.5

    return Bending_Moment, Bending_Moment_T, Shear_Force, Torque
