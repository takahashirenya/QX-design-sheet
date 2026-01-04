from dataclasses import dataclass, field
import numpy as np
from LLT_function.module import FlightState, Specifications


def calculate_specifications(
    wing: dict,
    state: FlightState,
    iteration: int,  # 1始まりで渡される想定
    dihedral_angle: np.ndarray,  # (2n,)
    chord_cp: np.ndarray,  # (2n,)
    y: np.ndarray,  # (2n+1,)
    cp: np.ndarray,  # (2, 2n) -> [y_cp, z_cp]
    a1: np.ndarray,  # (2n,)
    dDp: np.ndarray,  # (2n,)
    dN: np.ndarray,  # (2n,)
    dT: np.ndarray,  # (2n,)
    dM_ac: np.ndarray,  # (2n,)
    CL_arr: np.ndarray,  # (2n,)
    Cdp_arr: np.ndarray,  # (2n,)
    Lift: np.ndarray,  # (iters,)
    Induced_Drag: np.ndarray,  # (iters,)
):
    pi = np.pi
    rad = pi / 180.0
    deg = 180.0 / pi

    n = int(wing["span_div"])
    n2 = 2 * n

    # --- ここが重要：0始まりの履歴インデックスに変換（安全ガード付き） ---
    it_idx = max(0, min(iteration - 1, len(Lift) - 1))

    # dy はスカラー/配列の両対応
    dy = wing["dy"]
    dy_is_scalar = np.ndim(dy) == 0

    def dy_i(i: int) -> float:
        return float(dy) if dy_is_scalar else float(dy[i])

    # --- S ---
    wing["S"] = 0.0
    for i in range(n2):
        wing["S"] += chord_cp[i] * dy_i(i) * np.cos(rad * dihedral_angle[i])

    # --- chord_mac（左翼のみ積分→2/S で正規化）---
    wing["chord_mac"] = 0.0
    for i in range(0, n):
        wing["chord_mac"] += (
            (chord_cp[i] ** 2) * dy_i(i) * np.cos(rad * dihedral_angle[i])
        )
    wing["chord_mac"] = (2.0 / wing["S"]) * wing["chord_mac"]

    # --- 片翼面積中心 y_ ---
    wing["y_"] = 0.0
    for i in range(n, n2):
        wing["y_"] += chord_cp[i] * cp[0, i] * dy_i(i) * np.cos(rad * dihedral_angle[i])
    wing["y_"] = (2.0 / wing["S"]) * wing["y_"]

    # --- b, AR ---
    wing["b"] = 2.0 * y[n2]
    wing["AR"] = (wing["b"] * wing["b"]) / wing["S"]

    # --- Cla（翼素面積重み）---
    wing["Cla"] = 0.0
    for i in range(n2):
        cdy = chord_cp[i] * dy_i(i) * np.cos(rad * dihedral_angle[i])
        wing["Cla"] += (a1[i] * cdy) / wing["S"]

    # --- 力・モーメント合計 ---
    wing["Drag_parasite"] = 0.0
    wing["L_roll"] = 0.0
    wing["M_pitch"] = 0.0
    wing["N_yaw"] = 0.0
    for i in range(n2):
        wing["Drag_parasite"] += dDp[i]
        wing["L_roll"] -= dN[i] * np.cos(rad * dihedral_angle[i]) * cp[0, i]
        wing["M_pitch"] += dM_ac[i] + dT[i] * cp[1, i]
        wing["N_yaw"] += dT[i] * cp[0, i]

    # --- 履歴から総揚力・誘導抗力（オフバイワン修正済み）---
    wing["Lift"] = float(Lift[it_idx])
    wing["Drag_induced"] = float(Induced_Drag[it_idx])
    wing["Drag"] = wing["Drag_parasite"] + wing["Drag_induced"]

    # --- 係数 ---
    qS = wing["dynamic_pressure"] * wing["S"]
    wing["CL"] = wing["Lift"] / qS
    wing["Cdp"] = wing["Drag_parasite"] / qS
    wing["CDi"] = wing["Drag_induced"] / qS
    wing["CD"] = wing["Drag"] / qS
    wing["Cm_ac"] = wing["M_pitch"] / (qS * wing["chord_mac"])

    # e と aw
    wing["e"] = (
        ((wing["CL"] ** 2) / (pi * wing["AR"])) / wing["CDi"]
        if wing["CDi"] != 0
        else np.nan
    )
    wing["aw"] = (
        rad * (deg * wing["Cla"]) / (1.0 + ((deg * wing["Cla"]) / (pi * wing["AR"])))
    )

    # 重心周り
    wing["M_pitch"] = wing["M_pitch"] + wing["Lift"] * wing["chord_mac"] * (
        wing["hspar"] - wing["hac"] - state["dh"]
    )
    wing["Cm_cg"] = wing["M_pitch"] / (qS * wing["chord_mac"])

    # --- 横・方向の安定微係数 ---
    wing["Cyb"] = wing["Cyp"] = wing["Cyr"] = 0.0
    wing["Clb"] = wing["Clp"] = wing["Clr"] = 0.0
    wing["Cnb"] = wing["Cnp"] = wing["Cnr"] = 0.0

    for i in range(n, n2):  # 右翼のみ（式内で2/S等で対称考慮済み）
        # ここは元コード通り xy投影を掛けない版
        cdy = chord_cp[i] * dy_i(i)

        sinG = np.sin(rad * dihedral_angle[i])
        cosG = np.cos(rad * dihedral_angle[i])

        ycp = cp[0, i]
        zcp = cp[1, i]

        alpha_rad = rad * state["alpha"]
        Cx = CL_arr[i] * np.sin(alpha_rad) - Cdp_arr[i] * np.cos(alpha_rad)
        Cxa = (
            a1[i] * (1.0 / rad) * np.sin(alpha_rad)
            + CL_arr[i] * np.cos(alpha_rad)
            + Cdp_arr[i] * np.sin(alpha_rad)
        )  # [1/rad]

        wing["Cyb"] += -(2.0 / wing["S"]) * a1[i] * (sinG**2) * cdy
        wing["Cyp"] += (
            -(4.0 / (wing["S"] * wing["b"]))
            * a1[i]
            * (1.0 / rad)
            * sinG
            * cosG
            * ycp
            * cdy
        )
        wing["Cyr"] += (8.0 / (wing["S"] * wing["b"])) * CL_arr[i] * sinG * ycp * cdy

        wing["Clb"] += (
            -(2.0 / (wing["S"] * wing["b"]))
            * a1[i]
            * sinG
            * (cosG * ycp + sinG * zcp)
            * cdy
        )
        wing["Clp"] += (
            -(4.0 / (wing["S"] * wing["b"] * wing["b"]))
            * a1[i]
            * (1.0 / rad)
            * ycp
            * cosG
            * (cosG * ycp + sinG * zcp)
            * cdy
        )
        wing["Clr"] += (
            (8.0 / (wing["S"] * wing["b"] * wing["b"]))
            * CL_arr[i]
            * ycp
            * (cosG * ycp + sinG * zcp)
            * cdy
        )

        wing["Cnb"] += -(2.0 / (wing["S"] * wing["b"])) * Cxa * sinG * ycp * cdy * rad
        wing["Cnp"] += (
            -(4.0 / (wing["S"] * wing["b"] * wing["b"])) * Cxa * (ycp**2) * cosG * cdy
        )
        wing["Cnr"] += (8.0 / (wing["S"] * wing["b"] * wing["b"])) * Cx * (ycp**2) * cdy
