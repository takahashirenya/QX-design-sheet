import math
import numpy as np

_RAD = math.pi / 180.0
_DEG = 180.0 / math.pi


def calculate_aero(
    Wing,
    state,
    cp,
    chord_cp,
    setting_angle,
    alpha_effective,
    Re,
    dihedral_angle,
    foil_mixture,
    Cl_coef,
    Cdp_coef,
    Cm_coef,
    alpha_unit="deg",  # 'deg'（係数表が度ベース前提） or 'rad'（ラジアン入力なら度へ変換）
    log_base="e",  # 'e'（ln） or '10'（log10）
):
    m = chord_cp.shape[0]
    assert cp.shape == (2, m), "cp must be shape (2, 2n)"
    assert setting_angle.shape == (m,)
    assert alpha_effective.shape == (m,)
    assert Re.shape == (m,)
    assert dihedral_angle.shape == (m,)
    assert foil_mixture.shape == (m, 5)
    assert Cl_coef.shape == (16, 5)
    assert Cdp_coef.shape == (16, 5)
    assert Cm_coef.shape == (16, 5)

    # --- dy の取得（_dy_full_cache -> dy -> Wing['dy']） ---
    if hasattr(Wing, "_dy_full_cache"):
        dy_full = np.asarray(Wing._dy_full_cache, dtype=float)
    else:
        dy_in = None
        # 属性 or dict の両対応
        if hasattr(Wing, "dy"):
            dy_in = getattr(Wing, "dy")
        elif isinstance(Wing, dict) and "dy" in Wing:
            dy_in = Wing["dy"]
        if dy_in is None:
            raise RuntimeError(
                "可変 dy が見つかりません（Wing._dy_full_cache / Wing.dy / Wing['dy'] のいずれかを設定してください）。"
            )
        dy_full = (
            np.full(m, float(dy_in), dtype=float)
            if np.isscalar(dy_in)
            else np.asarray(dy_in, dtype=float)
        )
    if dy_full.shape != (m,):
        raise ValueError("dy はスカラーまたは shape=(2n,) の配列である必要があります。")

    # --- 角度：係数は度ベースなので α を[deg]に統一 ---
    if alpha_unit == "deg":
        a_deg = np.asarray(alpha_effective, dtype=float)
    elif alpha_unit == "rad":
        a_deg = np.asarray(alpha_effective, dtype=float) * _DEG
    else:
        raise ValueError("alpha_unit must be 'deg' or 'rad'")

    # --- log(Re) ---
    Re = np.asarray(Re, dtype=float)
    if np.any(Re <= 0.0):
        raise ValueError("Re は正である必要があります。")
    if log_base == "e":
        logf = np.log
    elif log_base == "10":
        logf = np.log10
    else:
        raise ValueError("log_base must be 'e' or '10'")
    L = logf(Re)  # (m,)

    # --- 出力配列 ---
    CL = np.zeros(m)
    Cdp = np.zeros(m)
    Cm_ac = np.zeros(m)
    a0 = (foil_mixture * Cl_coef[0, :]).sum(axis=1)  # α^0 の合成
    a1 = (foil_mixture * Cl_coef[1, :]).sum(axis=1)  # α^1 の合成
    Cda = np.zeros(m)
    Cla = np.zeros(m)
    Cm_cg = np.zeros(m)
    dN = np.zeros(m)
    dT = np.zeros(m)
    dL = np.zeros(m)
    dDp = np.zeros(m)
    dM_cg = np.zeros(m)
    dM_ac = np.zeros(m)

    y_cp = cp[0, :]  # cp のスパン位置

    # 係数は列=翼型なので、@ terms 用に転置
    ClT = Cl_coef.T  # (5,16)
    CdT = Cdp_coef.T  # (5,16)
    CmT = Cm_coef.T  # (5,16)

    # --- 16基底で評価（各 i を逐次。必要ならベクトル化可） ---
    # terms = [1, a..a^10, L, L^2, L^3, aL, a^2 L]
    for i in range(m):
        a = float(a_deg[i])
        logR = float(L[i])

        terms = np.empty(16, dtype=float)
        terms[0] = 1.0
        apow = 1.0
        for k in range(1, 11):  # α^1..α^10
            apow *= a
            terms[k] = apow
        terms[11] = logR
        terms[12] = logR * logR
        terms[13] = terms[12] * logR
        terms[14] = a * logR
        terms[15] = (a * a) * logR

        mix = foil_mixture[i, :]  # (5,)

        # 係数評価（配合率で合成）
        CL[i] = float(mix @ (ClT @ terms))
        Cdp[i] = float(mix @ (CdT @ terms))
        Cm_ac[i] = float(mix @ (CmT @ terms))

        # Cda = d(Cdp)/dα（αは度ベース）
        dterms = np.zeros(16, dtype=float)
        for p in range(1, 11):
            dterms[p] = p * (a ** (p - 1))  # d(α^p)/dα = p·α^(p-1)
        dterms[14] = logR  # d(α·logR)/dα = logR
        dterms[15] = 2.0 * a * logR  # d(α^2·logR)/dα = 2α·logR
        Cda[i] = float(mix @ (CdT @ dterms))
        Cla[i] = float(mix @ (ClT @ dterms))

    # --- 桁位置回りへ ---
    # Wing が属性/辞書両対応
    hspar = float(
        getattr(Wing, "hspar", Wing["hspar"] if isinstance(Wing, dict) else None)
    )
    hac = float(getattr(Wing, "hac", Wing["hac"] if isinstance(Wing, dict) else None))
    Cm_cg = Cm_ac + CL * (hspar - hac)

    # --- 面積・動圧 ---
    area = chord_cp * dy_full * np.cos(_RAD * dihedral_angle)
    V_local = (
        float(getattr(state, "Vair", state["Vair"]))
        - _RAD * float(getattr(state, "r", state["r"])) * y_cp
    )
    rho = float(getattr(state, "rho", state["rho"]))
    q = 0.5 * rho * (V_local**2)

    # 代表値を Wing に格納（平均）※両対応
    q_mean = float(np.mean(q))
    try:
        setattr(Wing, "dynamic_pressure", q_mean)
    except Exception:
        if isinstance(Wing, dict):
            Wing["dynamic_pressure"] = q_mean

    # --- 翼素力・モーメント ---
    dL = q * area * CL
    dDp = q * area * Cdp
    dM_cg = q * area * chord_cp * Cm_cg
    dM_ac = q * area * chord_cp * Cm_ac

    Wing["dDp"] = dDp

    Wing["dCl"] = CL
    Wing["dCdp"] = Cdp
    if "dDi" in Wing:
        Wing["dCdi"] = Wing["dDi"] / (q * area)  # 全体で割る
    Wing["Cda"] = Cda
    Wing["Cla"] = Cla

    # --- 機体軸へ分解： s = rad * (α_deg - setting_angle) ---
    s = _RAD * (a_deg - setting_angle)
    cs = np.cos(s)
    sn = np.sin(s)
    dN = q * area * (CL * cs - Cdp * sn)
    dT = q * area * (-CL * sn + Cdp * cs)

    return CL, Cdp, Cm_ac, a0, a1, Cda, Cm_cg, dN, dT, dL, dDp, dM_cg, dM_ac
