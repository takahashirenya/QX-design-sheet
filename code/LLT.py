# -*- coding: utf-8 -*-
import math
from pathlib import Path
from dataclasses import dataclass
import traceback

import numpy as np
import xlwings as xw

from LLT_function.calculaton_yz import calculation_yz
from LLT_function.calculate_cp import calculation_cp
from LLT_function.caluculate_ypzp import calculation_ypzp
from LLT_function.calculate_Rij import calculation_Rij
from LLT_function.caluculate_Qij import calculation_Qij
from LLT_function.update_circulation import update_circulation
from LLT_function.calculate_downwash import calculation_downwash
from LLT_function.calculate_alpha_effective import calculation_alpha_effective
from LLT_function.calculate_Re import calculate_Re
from LLT_function.calculate_aero import calculate_aero
from LLT_function.calculate_force import calculation_force
from LLT_function.calcurate_moment import calculation_moment
from LLT_function.culculate_phi_theta import calculation_phi_theta
from LLT_function.calculate_specifications import calculate_specifications
from LLT_function.module import WingType, FlightState, Specifications

from excel_col_to_num import excel_col_to_num


def read_from_excel(
    Vair,
    alpha,
    hE,
    wing_sheet_name="主翼(平面形)",
    col_chord="F",
    col_weight="E",
    col_setting_angle="G",  # 取り付け角
    col_foil_mixture_start="H",
    col_setting_dihedral="M",
):
    wb = xw.Book.caller()
    wing = wb.sheets[wing_sheet_name]
    spar = wb.sheets["主翼(桁)"]
    environment = wb.sheets["パラメーター"]

    n = int(wing.range("C204").value)  # 片側分割数 (n)

    col_chord_num = excel_col_to_num(col_chord)
    col_weight_num = excel_col_to_num(col_weight)
    col_setting_angle_num = excel_col_to_num(col_setting_angle)
    col_foil_mixture_start_num = excel_col_to_num(col_foil_mixture_start)
    col_setting_dihedral_num = excel_col_to_num(col_setting_dihedral)

    # 節点（中心=208行, 右へnつ: 計 n+1 本）
    chord_nodes_right = (
        wing.range((208, col_chord_num), (208 + n, col_chord_num))
        .options(np.array)
        .value.reshape(-1)
    )  # コード長
    w_nodes_right = (
        wing.range((208, col_weight_num), (208 + n, col_weight_num))
        .options(np.array)
        .value.reshape(-1)
    )  # 重さ
    setting_nodes_right = (
        wing.range((208, col_setting_angle_num), (208 + n, col_setting_angle_num))
        .options(np.array)
        .value.reshape(-1)
    )  # 取り付け角

    # 翼型配合率 1..5（H..L）
    fm_cols = [
        col_foil_mixture_start_num,
        col_foil_mixture_start_num + 1,
        col_foil_mixture_start_num + 2,
        col_foil_mixture_start_num + 3,
        col_foil_mixture_start_num + 4,
    ]
    fms = [
        wing.range((208, c), (208 + n, c)).options(np.array).value.reshape(-1)
        for c in fm_cols
    ]

    dihedral_nodes_right = (
        wing.range((208, col_setting_dihedral_num), (208 + n, col_setting_dihedral_num))
        .options(np.array)
        .value.reshape(-1)
    )  # 上反角

    # 桁シートの節点 EI, GJ（R,S）
    Eix_nodes_right = (
        spar.range((45, 18), (45 + n, 18)).options(np.array).value.reshape(-1)
    )
    GJ_nodes_right = (
        spar.range((45, 19), (45 + n, 19)).options(np.array).value.reshape(-1)
    )

    # パネル幅 dy（右側 n 個）D列
    dy_right = (
        wing.range((208, 4), (208 + n - 1, 4))
        .options(np.array)
        .value.reshape(-1)
        .astype(float)
    )
    assert dy_right.size == n

    # --- 節点（2n+1）左右展開 ---
    chord = np.empty(2 * n + 1, dtype=float)
    chord[n:] = chord_nodes_right
    chord[:n] = chord_nodes_right[:0:-1]

    # --- 右側パネル量（n個）→ 左右展開（2n）---
    chord_cp_right = 0.5 * (chord_nodes_right[:-1] + chord_nodes_right[1:])
    dw_right = 0.5 * (w_nodes_right[:-1] + w_nodes_right[1:])
    setting_angle0_right = 0.5 * (setting_nodes_right[:-1] + setting_nodes_right[1:])
    dihedral_angle0_right = 0.5 * (dihedral_nodes_right[:-1] + dihedral_nodes_right[1:])
    Eix_right = 0.5 * (Eix_nodes_right[:-1] + Eix_nodes_right[1:])
    GJ_right = 0.5 * (GJ_nodes_right[:-1] + GJ_nodes_right[1:])
    fm_right_list = [0.5 * (fm[:-1] + fm[1:]) for fm in fms]  # 各(n,)

    chord_cp = np.empty(2 * n)
    chord_cp[n:] = chord_cp_right
    chord_cp[:n] = chord_cp_right[::-1]
    dw = np.empty(2 * n)
    dw[n:] = dw_right
    dw[:n] = dw_right[::-1]
    setting_angle0 = np.empty(2 * n)
    setting_angle0[n:] = setting_angle0_right
    setting_angle0[:n] = setting_angle0_right[::-1]
    dihedral_angle0 = np.empty(2 * n)
    dihedral_angle0[n:] = dihedral_angle0_right
    dihedral_angle0[:n] = -dihedral_angle0_right[::-1]  # 左は負
    Eix = np.empty(2 * n)
    Eix[n:] = Eix_right
    Eix[:n] = Eix_right[::-1]
    GJ = np.empty(2 * n)
    GJ[n:] = GJ_right
    GJ[:n] = GJ_right[::-1]

    # dy（2n に左右展開）
    dy_full = np.empty(2 * n)
    dy_full[n:] = dy_right
    dy_full[:n] = dy_right[::-1]

    # 配合率を (2n,5) 行列へ
    right_stack = np.vstack(fm_right_list).T  # (n,5)
    foil_mixture = np.empty((2 * n, 5))
    foil_mixture[n:, :] = right_stack
    foil_mixture[:n, :] = right_stack[::-1, :]

    block = np.array(wing.range((426, 4), (440, 19)).value)

    # Cl, Cdp, Cm 係数配列を初期化
    Cl_coef = np.zeros((16, 5))
    Cdp_coef = np.zeros((16, 5))
    Cm_coef = np.zeros((16, 5))

    # block の中から、3 行ごと・16 列分を切り出して Cl, Cdp, Cm に割り当てる
    for j in range(5):
        r0 = 3 * j  # Cl の先頭行
        Cl_coef[:, j] = block[r0 + 0, 0:16]
        Cdp_coef[:, j] = block[r0 + 1, 0:16]
        Cm_coef[:, j] = block[r0 + 2, 0:16]

    hspar_value = wing.range("AI2").value
    if hspar_value is None:
        hspar_value = 0.35

    Wing = {
        "span_div": n,
        "dy": dy_full,
        "hspar": float(hspar_value),  # 桁位置(無次元)
        "hac": 0.25,  # AC位置(必要ならExcel化)
        "dynamic_pressure": 0.0,
    }

    State = {
        "Vair": float(Vair),
        "rho": float(environment.range("D4").value),
        "mu": float(environment.range("D6").value),
        "alpha": float(alpha),
        "beta": 0.0,
        "p": 0.0,
        "r": 0.0,
        "dh": 0.0,
        "hE": float(hE),
    }

    data = {
        "n": n,
        "chord": chord,  # 2n+1
        "chord_cp": chord_cp,  # 2n
        "dy_full": dy_full,  # 2n（左右展開済み）
        "dy_right": dy_right,  # 右 n（幾何・たわみ用）
        "dw": dw,  # 重さ
        "setting_angle0": setting_angle0,  # 2n
        "dihedral_angle0": dihedral_angle0,  # 2n（左負）
        "Eix": Eix,  # 2n
        "GJ": GJ,  # 2n
        "foil_mixture": foil_mixture,  # 2n×5
        "Cl_coef": Cl_coef,
        "Cdp_coef": Cdp_coef,
        "Cm_coef": Cm_coef,
    }

    return {
        "State": State,
        "Wing": Wing,
        "data": data,
    }


# エクセルを探す関数。
def FindWorkbookUpOne(Pattern="*.xlsm"):
    parent_dir = Path(__file__).resolve().parent.parent
    matches = list(parent_dir.glob(Pattern))
    if not matches:
        raise FileNotFoundError(f"No files matching {Pattern} in {parent_dir}")
    # 最新更新ファイル
    return max(matches, key=lambda p: p.stat().st_mtime)


def smooth_circulation(circulation, dy, n_iter=10, lam=0.5, preserve_lift=True):
    """
    circulation : shape (2n,) の Γ 分布
    dy          : shape (2n,) のパネル幅
    n_iter      : スムージング反復回数（大きいほど滑らかになる）
    lam         : スムージング強さ (0<lam<1)
    preserve_lift : True のとき、スムージング前後で総揚力が変わらないようスケール

    ここでは、右半分だけを平滑化してから左右対称にコピーする前提
    （幾何が左右対称のケース）。
    """
    gamma = circulation.copy()
    n2 = gamma.size
    n = n2 // 2

    # 右半分のΓと幅
    g = gamma[n:].copy()  # shape (n,)
    w = dy[n:].copy()  # shape (n,)

    # スムージング前の総揚力に相当する量（Γ×dy の和）
    if preserve_lift:
        lift_before = float(np.sum(g * w))

    # 1次元ラプラシアン型スムージング
    for _ in range(n_iter):
        g_new = g.copy()
        # 端はそのまま、中だけ隣との平均に寄せる
        g_new[1:-1] = (1.0 - lam) * g[1:-1] + 0.5 * lam * (g[2:] + g[:-2])
        g = g_new

    if preserve_lift:
        lift_after = float(np.sum(g * w))
        if lift_after != 0.0:
            scale = lift_before / lift_after
            g *= scale

    # 右半分を書き戻し、左半分は左右対称と仮定してミラーコピー
    gamma[n:] = g
    gamma[:n] = g[::-1]

    return gamma


def make_smooth_circulation_poly(circulation, dy, cp, degree=5, preserve_lift=True):
    """
    circulation : shape (2n,)   収束後の Γ 分布
    dy          : shape (2n,)   パネル幅
    cp          : shape (3,2n)  コントロールポイント（cp[1,:] が y 座標と仮定）
    degree      : 多項式の次数（3〜6くらいがおすすめ）
    preserve_lift : True なら ∑Γdy を保つようスケール
    """

    gamma = circulation.copy()
    n2 = gamma.size
    n = n2 // 2

    # 右半分の y 座標と Γ
    y_right = cp[1, n:]  # (n,)
    g_right = gamma[n:]  # (n,)
    w_right = dy[n:]  # (n,)

    # 無次元スパン位置 η = y / ymax にして条件を安定化
    eta = y_right / np.max(np.abs(y_right))

    # η–Γ を degree 次の多項式でフィット
    coeff = np.polyfit(eta, g_right, deg=degree)
    g_fit = np.polyval(coeff, eta)

    if preserve_lift:
        lift_before = float(np.sum(g_right * w_right))
        lift_after = float(np.sum(g_fit * w_right))
        if abs(lift_after) > 0.0:
            g_fit *= lift_before / lift_after

    gamma_smooth = gamma.copy()
    gamma_smooth[n:] = g_fit  # 右側
    gamma_smooth[:n] = g_fit[::-1]  # 左側はミラー（左右対称を仮定）

    return gamma_smooth


def vlm_wing(Wing, state, excel_data, output_circulation_position, smooth_circ=False):

    # =========================================================
    # 値を読み込み
    # =========================================================

    n = Wing["span_div"]  # スパン分割数
    # vlm_wing の冒頭あたり（ループ前）
    Wing["dynamic_pressure"] = 0.5 * float(state["rho"]) * float(state["Vair"]) ** 2

    n2 = 2 * n
    setting_angle0 = excel_data["setting_angle0"]  # 取り付け角
    setting_angle = np.zeros(n2)
    phi = np.zeros(2 * n + 1)  # ねじれ角

    dihedral_angle0 = excel_data["dihedral_angle0"]
    dihedral_angle = np.zeros(n2)
    theta = np.zeros(2 * n + 1)  # たわみ角

    y = np.zeros(2 * n + 1)
    z = np.zeros(2 * n + 1)

    cp = np.zeros((3, 2 * n))

    yp = np.zeros((n2, n2))
    zp = np.zeros((n2, n2))
    ymp = np.zeros((n2, n2))
    zmp = np.zeros((n2, n2))

    ds = Wing["dy"] / 2.0  # パネル半幅

    Rpij = np.zeros((n2, n2))
    Rmij = np.zeros((n2, n2))
    Rpijm = np.zeros((n2, n2))
    Rmijm = np.zeros((n2, n2))

    Qij = np.zeros((n2, n2))

    circulation = np.zeros(n2, dtype=float)
    circulation_old = np.zeros_like(circulation)

    wi = np.zeros(n2, dtype=float)  # 吹きおろし角
    alpha_induced = np.zeros(n2, dtype=float)  # 誘導迎角

    alpha_effective = np.zeros(n2, dtype=float)  # 有効迎角
    Re = np.zeros(n2, dtype=float)  # レイノルズ数

    chord_cp = excel_data["chord_cp"]
    foil_mixture = excel_data["foil_mixture"]
    Cl_coef = excel_data["Cl_coef"]
    Cdp_coef = excel_data["Cdp_coef"]
    Cm_coef = excel_data["Cm_coef"]

    max_iterations = 200
    tol = 1e-4  # 収束判定: 相対変化率 < 1e-4

    Lift_hist = np.zeros(max_iterations)
    Dind_hist = np.zeros(max_iterations)

    dW = excel_data["dw"]  # (2n,)
    deflection = np.zeros(2 * n + 1)  # たわみ量

    Specifications = {
        "Re": Re,
        "wi": wi,
        "alpha_induced": alpha_induced,
        "span_div": Wing["span_div"],
        "dy": Wing["dy"],
        "hspar": Wing["hspar"],
        "hac": Wing["hac"],
        "dynamic_pressure": 0.5 * float(state["rho"]) * float(state["Vair"]) ** 2,
        "S": 0.0,
        "chord_mac": 0.0,
        "y_": 0.0,
        "b": 0.0,
        "AR": 0.0,
        "Cla": 0.0,
        "Drag_parasite": 0.0,
        "L_roll": 0.0,
        "M_pitch": 0.0,
        "N_yaw": 0.0,
        "Lift": 0.0,
        "Drag_induced": 0.0,
        "Drag": 0.0,
        "CL": 0.0,
        "Cdp": 0.0,
        "CDi": 0.0,
        "CD": 0.0,
        "Cm_ac": 0.0,
        "Cm_cg": 0.0,
        "e": 0.0,
        "aw": 0.0,
        "Cyb": 0.0,
        "Cyp": 0.0,
        "Cyr": 0.0,
        "Clb": 0.0,
        "Clp": 0.0,
        "Clr": 0.0,
        "Cnb": 0.0,
        "Cnp": 0.0,
        "Cnr": 0.0,
    }

    for iteration in range(1, max_iterations + 1):

        calculation_yz(
            Wing,
            state,
            setting_angle0,
            setting_angle,
            phi,
            dihedral_angle0,
            dihedral_angle,
            theta,
            y,
            z,
            y_center=0.0,
            z_center=0.0,
        )

        calculation_cp(Wing, state, cp, y, z)

        calculation_ypzp(Wing, state, yp, zp, ymp, zmp, cp, dihedral_angle)

        calculation_Rij(Wing, state, ds, yp, zp, ymp, zmp, Rpij, Rmij, Rpijm, Rmijm)

        calculation_Qij(
            Wing,
            state,
            ds,
            yp,
            zp,
            ymp,
            zmp,
            Rpij,
            Rmij,
            Rpijm,
            Rmijm,
            dihedral_angle,
            Qij,
        )

        if smooth_circ:
            circulation = smooth_circulation(
                circulation,
                Wing["dy"],
                n_iter=10,
                lam=0.5,
                preserve_lift=True,
            )

        update_circulation(
            Wing,
            state,
            iteration=iteration,
            circulation=circulation,
            circulation_old=circulation_old,
        )

        calculation_downwash(Wing, state, circulation, Qij, cp, alpha_induced, wi)

        calculation_alpha_effective(
            Wing,
            state,
            alpha_effective,
            alpha_induced=alpha_induced,
            cp=cp,
            dihedral_angle=dihedral_angle,
            setting_angle=setting_angle,
            alpha_max=20.0,
            alpha_min=-15.0,
        )

        calculate_Re(
            Wing,
            state,
            Re,
            cp=cp,
            chord_cp=excel_data["chord_cp"],
            Re_max=1e7,
            Re_min=5e3,
        )

        (
            CL,
            Cdp,
            Cm_ac,
            a0,
            a1,
            Cda,
            Cm_cg,
            dN,
            dT,
            dL,
            dDp,
            dM_cg,
            dM_ac,
        ) = calculate_aero(
            Wing,
            state,
            cp[:2, :],
            chord_cp,
            setting_angle,
            alpha_effective,
            Re,
            dihedral_angle,
            foil_mixture,
            Cl_coef,
            Cdp_coef,
            Cm_coef,
            alpha_unit="deg",
            log_base="10",
        )

        Lift_hist, Dind_hist, circulation = calculation_force(
            Wing,
            state,
            iteration=iteration,
            Lift=Lift_hist,
            Induced_Drag=Dind_hist,
            cp=cp,
            circulation=circulation,
            dihedral_angle=dihedral_angle,
            wi=wi,
            CL=CL,
            chord_cp=chord_cp,
        )

        # 断面力に基づく変形
        cp_yz = np.vstack([cp[1, :], cp[2, :]])
        BM, BMT, SF, TQ = calculation_moment(
            Wing,
            state,
            dN=dN,
            dT=dT,
            dM_cg=dM_cg,
            dW=dW,
            dihedral_angle=dihedral_angle,
            cp=cp_yz,
            y=y,
            z=z,
        )
        calculation_phi_theta(
            Wing,
            state,
            deflection=deflection,
            theta=theta,
            phi=phi,
            bending_moment=BM,
            torque=TQ,
            eix=excel_data["Eix"],
            gj=excel_data["GJ"],
        )

        if smooth_circ:
            circulation = make_smooth_circulation_poly(
                circulation,
                Wing["dy"],
                cp,
                degree=5,
                preserve_lift=True,
            )

        # ====== 収束判定（Lift の相対変化）======
        if iteration >= 2:
            L_now = Lift_hist[iteration - 1]
            L_prev = Lift_hist[iteration - 2]
            rel_change = abs(L_now - L_prev) / (abs(L_now) + 1e-12)

            if rel_change < tol:
                print(
                    f"Converged at iter {iteration}: Lift={L_now:.6e}, dL/L={rel_change:.3e}"
                )
                break
    else:
        print(
            f"WARNING: Not converged within {max_iterations} iterations. Last dL/L={rel_change:.3e}"
        )

    # wi, alpha_induced の再計算
    calculation_downwash(Wing, state, circulation, Qij, cp, alpha_induced, wi)

    calculate_specifications(
        Specifications,
        state,
        iteration,
        dihedral_angle,
        chord_cp,
        y,
        cp_yz,
        a1,
        dDp,
        dN,
        dT,
        dM_ac,
        CL,
        Cdp,
        Lift_hist[:iteration],
        Dind_hist[:iteration],
    )
    S = float(np.sum(chord_cp * Wing["dy"]))
    q = Wing["dynamic_pressure"]  # 0.5 * rho * V^2

    Lift = Lift_hist[iteration - 1]  # total lift [N]
    Drag_induced = Dind_hist[iteration - 1]  # induced drag [N]
    Drag_parasite = float(np.sum(dDp))  # parasite drag [N] (from calculate_aero)
    Cm_cg_scalar = Specifications["Cm_cg"]

    if q > 0.0 and S > 0.0:
        CL = Lift / (q * S)
        Cdp = Drag_parasite / (q * S)
        CDi = Drag_induced / (q * S)
        CD = Cdp + CDi
    else:
        CL = Cdp = CDi = CD = 0.0

    Drag_total = Drag_parasite + Drag_induced
    L_over_D = Lift / Drag_total if Drag_total > 0.0 else 0.0

    output_data = {
        "Specifications": Specifications,
        "n": n,
        "circulation": circulation,
        "wi": wi,
        "w": deflection,
        "y": y,
        "z": z,
        "cp": cp,
        "dihedral_angle": dihedral_angle,
        "theta": theta,
        "phi": phi,
        "deflection": deflection,
        "Re": Re,
        "alpha_effective": alpha_effective,
        "dL": dL,
        "dD": Wing["dDp"] + Wing["dDi"],
        "dM_ac": dM_ac,
        "dM_cg": dM_cg,
        "Cl": CL,
        "dCdp": Wing["dCdp"],
        "dCi": Wing["dCdi"],
        "dCm_ac": Cm_ac,
        "dCm_cg": Cm_cg,
        "alpha_induced": alpha_induced,
        "setting_angle": setting_angle,
        "dihedral_angle0": dihedral_angle0,
        "BM": BM,
        "Lift": Lift_hist[iteration - 1],
        "Drag_induced": Dind_hist[iteration - 1],
        "Drag_parasite": Drag_parasite,
        "Drag_total": Drag_total,
        "efficiency": Specifications["e"],
        "S": S,
        "CL": CL,
        "Cdp": Cdp,
        "CDi": CDi,
        "CD": CD,
        "L_over_D": L_over_D,
        "Cm_cg": Cm_cg_scalar,
    }
    return output_data


def output_to_excel(
    output_data_smooth,
    output_data,
    col_circulation="O",
    col_dihedral="N",
    col_y="P",
    col_z="Q",
    output_data_cell="N28",
):
    wb = xw.Book.caller()
    sheet = wb.sheets["主翼(平面形)"]
    n = output_data_smooth["n"]
    sheet.range(f"{col_circulation}208").options(transpose=True).value = (
        output_data_smooth["circulation"][n:]
    )
    sheet.range(f"{col_dihedral}208").options(transpose=True).value = (
        output_data_smooth["dihedral_angle"][n:]
    )
    sheet.range(f"{col_y}208").options(transpose=True).value = output_data["y"][n:]
    sheet.range(f"{col_z}208").options(transpose=True).value = output_data["z"][n:]

    out = np.array(
        [
            output_data_smooth["Lift"],  # N28: 最終反復の揚力
            output_data_smooth["Drag_induced"],  # N29: 最終反復の誘導抗力
            output_data_smooth["efficiency"],  # N30: 翼効率 e
            output_data_smooth["Drag_total"],  # N31: 最終反復の総抗力
        ],
        dtype=float,
    ).reshape(4, 1)

    sheet.range(output_data_cell).value = out

    pass


def solve_V_for_L(
    Wing,
    State,
    data,
    alpha_deg,
    L_target,
    V_min=7.0,
    V_max=15.0,
    tol_rel=1e-2,
    max_iter=10,
):
    """
    与えられた迎角 alpha_deg[deg] で、
    L = L_target となる速度 V[m/s] を二分法で求める。

    Wing, State, data : read_from_excel() から得たもの
    alpha_deg         : [deg] 固定迎角
    L_target          : [N] 目標揚力
    V_min, V_max      : 探索する速度範囲 [m/s]
    tol_rel           : L の相対誤差許容値
    max_iter          : 二分法反復回数
    """

    State["alpha"] = float(alpha_deg)

    def lift_at_V(V):
        State["Vair"] = float(V)
        out = vlm_wing(
            Wing,
            State,
            data,
            output_circulation_position=True,
            smooth_circ=False,
        )
        return float(out["Lift"])

    # まず端点で L を評価して、符号が反転しているか確認
    L_low = lift_at_V(V_min)
    L_high = lift_at_V(V_max)

    if (L_low - L_target) * (L_high - L_target) > 0.0:
        raise ValueError(
            f"solve_V_for_L: L_target が探索速度範囲内にありません。"
            f" V_min={V_min}, L={L_low:.3f}, "
            f" V_max={V_max}, L={L_high:.3f}, "
            f" L_target={L_target:.3f}"
        )

    V_lo = V_min
    V_hi = V_max
    V_mid = 0.5 * (V_lo + V_hi)
    L_mid = lift_at_V(V_mid)

    for _ in range(max_iter):
        V_mid = 0.5 * (V_lo + V_hi)
        L_mid = lift_at_V(V_mid)

        # 相対誤差で収束判定
        if abs(L_mid - L_target) / (abs(L_target) + 1e-12) < tol_rel:
            return V_mid, L_mid

        # 二分法のブランチ更新
        if (L_low - L_target) * (L_mid - L_target) <= 0.0:
            V_hi = V_mid
            L_high = L_mid
        else:
            V_lo = V_mid
            L_low = L_mid

    # 収束しなかった場合でも最新値を返す
    return V_mid, L_mid


def LLT_1_alpha_fixed_find_V():
    """
    主翼(平面形) シートで α を固定して、L_target になる V を探索し、
    その条件で LLT を実行して結果を Excel に出力する。まあ使わないんじゃね。
    """

    wb = xw.Book.caller()
    wing_sheet = wb.sheets["主翼(平面形)"]

    # 入力
    alpha = float(wing_sheet.range("E6").value)  # 固定する迎角[deg]
    hE = float(wing_sheet.range("C6").value)  # 高度など
    L_target = float(wing_sheet.range("F6").value)  # 目標揚力[N]

    # 初期 V は適当な値（探索用なので何でもよい）
    V_init = 9.5

    # Excel から geometry 等を読み込み（1回だけ）
    excel_data = read_from_excel(
        Vair=V_init,
        alpha=alpha,
        hE=hE,
        col_chord="F",
        col_weight="E",
        col_setting_angle="G",
        col_foil_mixture_start="H",
        col_setting_dihedral="M",
    )
    Wing = excel_data["Wing"]
    State = excel_data["State"]
    data = excel_data["data"]

    # V を二分法で求める
    V_sol, L_sol = solve_V_for_L(
        Wing,
        State,
        data,
        alpha_deg=alpha,
        L_target=L_target,
        V_min=7.0,
        V_max=15,
        tol_rel=1e-2,
        max_iter=10,
    )
    print(f"alpha={alpha:.3f} deg -> V={V_sol:.3f} m/s, Lift={L_sol:.1f} N")

    # 見つかった V をセルに書く（任意）
    wing_sheet.range("J6").value = V_sol


def LLT_main(
    col_chord="F",
    col_weight="E",
    col_setting_angle="G",
    col_foil_mixture_start="H",
    col_setting_dihedral="M",
    col_circulation="O",
    col_dihedral="N",
    col_y="P",
    col_z="Q",
    output_data_cell="N28",
    Vair=0,
    alpha=0,
    hE=0,
):
    wing_sheet_name = "主翼(平面形)"
    excel_data = read_from_excel(
        Vair,
        alpha,
        hE,
        wing_sheet_name,
        col_chord,
        col_weight,
        col_setting_angle,
        col_foil_mixture_start,
        col_setting_dihedral,
    )
    state = excel_data["State"]
    wing = excel_data["Wing"]
    data = excel_data["data"]
    out_raw = vlm_wing(
        wing, state, data, output_circulation_position=True, smooth_circ=True
    )
    gamma_smooth = make_smooth_circulation_poly(
        out_raw["circulation"],
        wing["dy"],
        out_raw["cp"],
        degree=5,
        preserve_lift=True,
    )

    out_smooth = out_raw.copy()
    out_smooth["circulation"] = gamma_smooth
    output_to_excel(
        out_smooth,
        out_raw,
        col_circulation,
        col_dihedral,
        col_y,
        col_z,
        output_data_cell,
    )


def LLT_1():
    wb = xw.Book.caller()
    wing = wb.sheets["主翼(平面形)"]
    Vair = float(wing.range("D6").value)
    alpha = float(wing.range("E6").value)
    hE = float(wing.range("C6").value)
    LLT_main(Vair=Vair, alpha=alpha, hE=hE)


def LLT_2():
    wb = xw.Book.caller()
    wing = wb.sheets["主翼(平面形)"]
    Vair = float(wing.range("D7").value)
    alpha = float(wing.range("E7").value)
    hE = float(wing.range("C7").value)
    LLT_main(
        Vair=Vair,
        alpha=alpha,
        hE=hE,
        col_chord="R",
        col_setting_angle="S",
        col_foil_mixture_start="T",
        col_setting_dihedral="Y",
        col_circulation="AA",
        col_dihedral="Z",
        col_y="AB",
        col_z="AC",
        output_data_cell="N47",
    )


def LLT_3():
    wb = xw.Book.caller()
    wing = wb.sheets["主翼(平面形)"]
    Vair = float(wing.range("D8").value)
    alpha = float(wing.range("E8").value)
    hE = float(wing.range("C8").value)
    LLT_main(
        Vair=Vair,
        alpha=alpha,
        hE=hE,
        col_chord="AD",
        col_setting_angle="AE",
        col_foil_mixture_start="AF",
        col_setting_dihedral="AK",
        col_circulation="AM",
        col_dihedral="AL",
        col_y="AN",
        col_z="AO",
        output_data_cell="N67",
    )


def LLT_result():
    wb = xw.Book.caller()
    wing_sheet = wb.sheets["主翼(平面形結果)"]

    analysis_case = wing_sheet.range("K3").value

    condition_base_col = excel_col_to_num("J")
    condition_col = condition_base_col + (analysis_case - 1)

    Vair = float(wing_sheet.range((6, condition_col)).value)
    hE = float(wing_sheet.range((7, condition_col)).value)

    alpha_range = wing_sheet.range("B46:B61").value
    print(f"αリスト: {alpha_range}")
    alpha_list = [a for a in alpha_range if a not in (None, "")]
    if not alpha_list:
        raise ValueError(
            "主翼(平面形結果) シート B46:B61 に α[deg] を入力してください。"
        )

    excel_data = read_from_excel(
        Vair=Vair,
        alpha=0.0,
        hE=hE,
        wing_sheet_name="主翼(平面形結果)",
        col_chord="F",
        col_weight="E",
        col_setting_angle="G",
        col_foil_mixture_start="H",
        col_setting_dihedral="M",
    )
    Wing = excel_data["Wing"]
    Wing["hspar"] = float(wing_sheet.range("J10").value)
    State = excel_data["State"]
    data = excel_data["data"]
    State["Vair"] = float(Vair)

    print()

    CL_list = []
    CD_list = []
    LD_list = []
    Cm_list = []

    for alpha in alpha_list:
        State["alpha"] = float(alpha)

        out = vlm_wing(
            Wing,
            State,
            data,
            output_circulation_position=True,
            smooth_circ=False,
        )

        CL = float(out["CL"])
        CD = float(out["CD"])
        LD = float(out["L_over_D"])
        Cm = float(out["Cm_cg"])

        CL_list.append(CL)
        CD_list.append(CD)
        LD_list.append(LD)
        Cm_list.append(Cm)
        # 遅くなるけど、一個ずつ書き出したほうが動いてる感ある。
        # output_base_col = excel_col_to_num("C") + 4 * (analysis_case - 1)

        # wing_sheet.range((46, output_base_col)).options(transpose=True).value = CL_list
        # wing_sheet.range((46, output_base_col + 1)).options(transpose=True).value = CD_list
        # wing_sheet.range((46, output_base_col + 2)).options(transpose=True).value = LD_list
        # wing_sheet.range((46, output_base_col + 3)).options(transpose=True).value = Cm_list
    output_base_col = excel_col_to_num("C") + 4 * (analysis_case - 1)

    wing_sheet.range((46, output_base_col)).options(transpose=True).value = CL_list
    wing_sheet.range((46, output_base_col + 1)).options(transpose=True).value = CD_list
    wing_sheet.range((46, output_base_col + 2)).options(transpose=True).value = LD_list
    wing_sheet.range((46, output_base_col + 3)).options(transpose=True).value = Cm_list


def LLT_steady():
    wb = xw.Book.caller()
    output_sheet = wb.sheets["主翼(型紙出力)"]
    spar_sheet = wb.sheets["主翼(桁)"]

    Vair = float(output_sheet.range("Q6").value)
    hE = float(output_sheet.range("Q5").value)
    alpha = float(output_sheet.range("Q7").value)

    excel_data = read_from_excel(
        Vair=Vair,
        alpha=alpha,
        hE=hE,
        wing_sheet_name="主翼(平面形結果)",
        col_chord="F",
        col_weight="E",
        col_setting_angle="G",
        col_foil_mixture_start="H",
        col_setting_dihedral="M",
    )
    Wing = excel_data["Wing"]
    State = excel_data["State"]
    data = excel_data["data"]

    out = vlm_wing(
        Wing,
        State,
        data,
        output_circulation_position=True,
        smooth_circ=False,
    )
    n = Wing["span_div"]

    alpha_effective = out["alpha_effective"][n:]
    Re = out["Re"][n:]
    dL = out["dL"][n:]
    dD = out["dD"][n:]
    dM_ac = out["dM_ac"][n:]
    dM_cg = out["dM_cg"][n:]
    dCl = Wing["dCl"][n:]
    dCdp = Wing["dCdp"][n:]
    dCdi = Wing["dCdi"][n:]
    dCm_ac = out["dCm_ac"][n:]
    dCm_cg = out["dCm_cg"][n:]
    Cla = Wing["Cla"][n:]
    Cda = Wing["Cda"][n:]
    circuration = out["circulation"][n:]
    wi = out["wi"][n:]
    ai = out["alpha_induced"][n:]

    setting_angle = out["setting_angle"][n:]
    dihedral_angle = out["dihedral_angle"][n:]
    y = out["y"][n:]
    z = out["z"][n:]
    w = out["w"][n:]
    phi = out["phi"][n:]

    Bending_moment = out["BM"][n:] * 1000

    # 5つの配列をまとめて「行ごとのリスト」に変換
    # 1行 = [alpha_effective[i], Re[i], dL[i], dD[i], dM_cg[i]]
    output_1 = list(
        zip(
            alpha_effective,
            Re,
            dL,
            dD,
            dM_ac,
            dM_cg,
            dCl,
            dCdp,
            dCdi,
            dCm_ac,
            dCm_cg,
            Cla,
            Cda,
            circuration,
            wi,
            ai,
        )
    )

    output_2 = list(
        zip(
            setting_angle,
            dihedral_angle,
            y,
            z,
            w,
            phi,
        )
    )

    # N51 から一括で書き込み（縦方向に伸びて、5列分書かれる）
    output_sheet.range("N51").value = output_1
    output_sheet.range("AN51").value = output_2
    spar_sheet.range("AQ45").options(transpose=True).value = Bending_moment


import time

if __name__ == "__main__":

    wb_path = FindWorkbookUpOne("*.xlsm")
    wb = xw.Book(str(wb_path))
    wb.set_mock_caller()

    LLT_result()
    # LLT_1_alpha_fixed_find_V()

    # wb_path = FindWorkbookUpOne("*.xlsm")
    # wb = xw.Book(str(wb_path))
    # wb.set_mock_caller()

    # t0 = time.perf_counter()
    # excel_data = read_from_excel(Vair=10, alpha=0, hE=2)
    # t1 = time.perf_counter()
    # output_data = vlm_wing(
    #     excel_data["Wing"],
    #     excel_data["State"],
    #     excel_data["data"],
    #     True,
    #     smooth_circ=False,
    # )
    # output_data_smooth = vlm_wing(
    #     excel_data["Wing"],
    #     excel_data["State"],
    #     excel_data["data"],
    #     True,
    #     smooth_circ=True,
    # )
    # t2 = time.perf_counter()
    # output_to_excel(output_data_smooth, output_data)
    # t3 = time.perf_counter()

    # print(f"read_from_excel: {t1 - t0:.3f} s")
    # print(f"vlm_wing:        {t2 - t1:.3f} s")
    # print(f"output_to_excel: {t3 - t2:.3f} s")
