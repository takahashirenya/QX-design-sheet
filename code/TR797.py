# -*- coding: utf-8 -*-
# ------------------------------------------------------------
# TR-797 準拠・VBAロジックのみ：与えられた揚力と翼根曲げモーメント制約下で
# 誘導抗力（Di）を最小化する循環分布の最適化（片翼 N 分割, 均等パネル）
#
# ・VBAの数式・係数・両翼係数・dsの定義に完全準拠
# ・核 Qij は VBA 式を 1/(2π) 係数まで同一で実装
# ・Aij = π * Qij * ds,  bi, ci も VBA と同一式
# ・beta=0 の特別ケースも VBA と同様に扱う
# ・戻り値は VBA の TR797 と同じ： (span_div+1) x 2 配列（circulation, wi）
#   末尾行は [0, 0]
# ・あわせて AW15:AW19 相当（beta, e, Lift, Di, BendingMoment）も出力
# ------------------------------------------------------------

import numpy as np
from numpy import pi
import xlwings as xw
from pathlib import Path
from excel_col_to_num import excel_col_to_num


def _build_interp_matrix(y: np.ndarray, knots_y: np.ndarray) -> np.ndarray:
    """
    線形補間行列 N を作る（Γ = N @ Γ_ctrl）。
    y:   (N,) パネル中心 [m]
    knots_y: (K,) 節点位置 [m], 昇順, 範囲は [0, le] を推奨
    端での外挿は「最近節点にクランプ」。
    """
    Np = y.size
    K = knots_y.size
    Nmat = np.zeros((Np, K))
    # 走査用に節点間の区間番号を y ごとに見つける
    j = 0
    for i in range(Np):
        yi = y[i]
        # 左端クランプ
        if yi <= knots_y[0]:
            Nmat[i, 0] = 1.0
            continue
        # 右端クランプ
        if yi >= knots_y[-1]:
            Nmat[i, -1] = 1.0
            continue
        # 区間探索（単調増加なので前回位置から進める）
        while j < K - 1 and yi > knots_y[j + 1]:
            j += 1
        # yi ∈ [knots_y[j], knots_y[j+1]]
        x0, x1 = knots_y[j], knots_y[j + 1]
        t = (yi - x0) / (x1 - x0)
        Nmat[i, j] = 1.0 - t
        Nmat[i, j + 1] = t
    return Nmat


def TR797_poly_py(
    span_div: int,
    beta: float | None,  # ← None を許可
    hE: float,
    Lift: float,
    le: float,
    U: float,
    rho: float,
    knots_y: np.ndarray,
    ds_vec: np.ndarray | None = None,
    ds: float | None = None,
    phi: np.ndarray | None = None,
    y_panel: np.ndarray | None = None,
    z: np.ndarray | None = None,
):
    N = span_div

    # --- ds_vec 優先。無ければ ds から一様生成 ---
    if ds_vec is not None:
        ds_vec = np.asarray(ds_vec, dtype=float).reshape(-1)
        if ds_vec.size != N:
            raise ValueError(f"ds_vec size {ds_vec.size} != span_div {N}")
    else:
        if ds is None:
            ds = le / (2 * N)  # 半幅（従来互換）
        ds_vec = np.full(N, float(ds), dtype=float)

    ds_i = ds_vec.reshape(N, 1)
    ds_j = ds_vec.reshape(1, N)

    # --- y ---
    if y_panel is not None:
        y = np.asarray(y_panel, dtype=float).reshape(-1)
        if y.size != N:
            raise ValueError(f"y_panel size {y.size} != span_div {N}")
    else:
        dy_uni = le / N
        y = (np.arange(N) + 0.5) * dy_uni

    # --- z ---
    if z is None:
        z = np.zeros(N)
    else:
        z = np.asarray(z, dtype=float).reshape(-1)
        if z.size != N:
            raise ValueError(f"z size {z.size} != span_div {N}")

    # --- phi ---
    if phi is None:
        phi = np.zeros(N)
    else:
        phi = np.asarray(phi, dtype=float).reshape(-1)
        if phi.size != N:
            raise ValueError(f"phi size {phi.size} != span_div {N}")

    if np.any(ds_vec <= 0):
        print(
            "[DIAG-1][WARN] ds_vec に 0 以下があります -> A,b,c が壊れて特異化しやすい"
        )

    yi = y.reshape(N, 1)
    yj = y.reshape(1, N)
    zi = z.reshape(N, 1)
    zj = z.reshape(1, N)
    phij = phi.reshape(1, N)
    phii = phi.reshape(N, 1)

    yp = (yi - yj) * np.cos(phij) + (zi - zj) * np.sin(phij)
    zp = -(yi - yj) * np.sin(phij) + (zi - zj) * np.cos(phij)
    ypp = (yi + yj) * np.cos(phij) - (zi - zj) * np.sin(phij)
    zpp = (yi + yj) * np.sin(phij) + (zi - zj) * np.cos(phij)

    ymp = (yi - yj) * np.cos(phij) - (zi - (-zj - 2 * hE)) * np.sin(phij)
    zmp = (yi - yj) * np.sin(phij) + (zi - (-zj - 2 * hE)) * np.cos(phij)
    ympp = (yi + yj) * np.cos(phij) + (zi - (-zj - 2 * hE)) * np.sin(phij)
    zmpp = -(yi + yj) * np.sin(phij) + (zi - (-zj - 2 * hE)) * np.cos(phij)

    # ★ここから ds_j を使う（ds は一切参照しない）
    Rpij = np.sqrt((yp - ds_j) ** 2 + (zp**2))
    Rmij = np.sqrt((yp + ds_j) ** 2 + (zp**2))
    Rpijp = np.sqrt((ypp + ds_j) ** 2 + (zpp**2))
    Rmijp = np.sqrt((ypp - ds_j) ** 2 + (zpp**2))
    Rpijm = np.sqrt((ymp - ds_j) ** 2 + (zmp**2))
    Rmijm = np.sqrt((ymp + ds_j) ** 2 + (zmp**2))
    Rpijmp = np.sqrt((ympp + ds_j) ** 2 + (zmpp**2))
    Rmijmp = np.sqrt((ympp - ds_j) ** 2 + (zmpp**2))

    c_imj = np.cos(phii - phij)
    s_imj = np.sin(phii - phij)
    c_ipj = np.cos(phii + phij)
    s_ipj = np.sin(phii + phij)

    Qij = (
        (-(yp - ds_j) / (Rpij * Rpij) + (yp + ds_j) / (Rmij * Rmij)) * c_imj
        + (-(zp) / (Rpij * Rpij) + (zp) / (Rmij * Rmij)) * s_imj
        + (-(ypp - ds_j) / (Rmijp * Rmijp) + (ypp + ds_j) / (Rpijp * Rpijp)) * c_ipj
        + (-(zpp) / (Rmijp * Rmijp) + (zpp) / (Rpijp * Rpijp)) * s_ipj
        + ((ymp - ds_j) / (Rpijm * Rpijm) - (ymp + ds_j) / (Rmijm * Rmijm)) * c_ipj
        + ((zmp) / (Rpijm * Rpijm) - (zmp) / (Rmijm * Rmijm)) * s_ipj
        + ((ympp - ds_j) / (Rmijmp * Rmijmp) - (ympp + ds_j) / (Rpijmp * Rpijmp))
        * c_imj
        + ((zmpp) / (Rmijmp * Rmijmp) - (zmpp) / (Rpijmp * Rpijmp)) * s_imj
    ) * (1.0 / (2.0 * np.pi))
    np.set_printoptions(threshold=np.inf)

    # ★Aij, b, c も ds_vec で
    Aij = np.pi * Qij * ds_i
    eta = y / le
    zeta = z / le
    bi = (3 * np.pi / 2.0) * (eta * np.cos(phi) + zeta * np.sin(phi)) * (ds_vec / le)
    ci = 2.0 * np.cos(phi) * (ds_vec / le)

    # --- 多角化の補間行列 N とスケーリング行列 S の構築 ---
    knots_y = np.asarray(knots_y, dtype=float)
    if knots_y.ndim != 1 or knots_y.size < 2:
        raise ValueError("knots_y は長さ2以上の一次元昇順配列にしてください。")
    if not np.all(np.diff(knots_y) > 0):
        raise ValueError("knots_y は昇順である必要があります。")
    # 推奨：端点 0, le を含める（含まれない場合は端でクランプ）
    Nmat = _build_interp_matrix(y, knots_y)  # (N, K)
    alpha = (2.0 * le * rho * U) / Lift  # g = alpha * Γ
    S = alpha * Nmat  # (N, K)

    # === DIAG-2: knots と補間行列の健全性 ===
    K = knots_y.size
    if K > N:
        print(
            "[DIAG-2][ERROR] K > N なので H=S^TMS は必ず特異になり得ます（要：節点数削減）"
        )

    rankS = np.linalg.matrix_rank(S)

    # --- 射影後の KKT を組む：H = Sᵀ M S,  C = -Sᵀ c,  B = -Sᵀ b ---
    M = Aij + Aij.T
    H = S.T @ M @ S
    C = -(S.T @ ci)
    B = -(S.T @ bi)

    K = H.shape[0]

    if beta is None:
        # --- β制約なし：KKT (K+1) ---
        LHS = np.zeros((K + 1, K + 1))
        RHS = np.zeros(K + 1)

        LHS[:K, :K] = H
        LHS[:K, K] = C
        LHS[K, :K] = C
        RHS[K] = -1.0

        sol = np.linalg.solve(LHS, RHS)
        Gamma_ctrl = sol[:K]

    else:
        # --- 現状：制約2本：KKT (K+2) ---
        LHS = np.zeros((K + 2, K + 2))
        RHS = np.zeros(K + 2)

        LHS[:K, :K] = H
        LHS[:K, K] = C
        LHS[:K, K + 1] = B
        LHS[K, :K] = C
        LHS[K + 1, :K] = B

        RHS[K] = -1.0
        RHS[K + 1] = -beta

        sol = np.linalg.solve(LHS, RHS)
        Gamma_ctrl = sol[:K]

    Gamma = Nmat @ Gamma_ctrl

    wi = 0.5 * (Qij @ Gamma)
    Di = 2.0 * rho * float(np.sum(Gamma * (Qij @ Gamma) * ds_vec))
    geom_proj = y * np.cos(phi) + z * np.sin(phi)
    BendingMoment = float(np.sum(2.0 * rho * U * Gamma * geom_proj * ds_vec))
    e = ((Lift * Lift) / (2.0 * np.pi * rho * U * U * le * le)) / Di
    beta_out = BendingMoment / ((2.0 / (3.0 * np.pi)) * le * Lift)

    print("Qij:", Qij)
    print(Gamma)

    # 出力テーブル（VBA 互換）
    out_tbl = np.zeros((N + 1, 2))
    out_tbl[:N, 0] = Gamma
    out_tbl[:N, 1] = wi

    summary = {
        "beta": beta_out,
        "e": e,
        "Lift": Lift,
        "Di": Di,
        "BendingMoment": BendingMoment,
    }
    return out_tbl, summary, knots_y, Gamma_ctrl


def _read_vector_from_range(sheet, col_letter: str, span_div: int) -> np.ndarray:
    """
    指定列から span_div 個の値を読み出して 1D numpy 配列にするユーティリティ。
    ブランクや非数値は 0.0 とみなす。
    例: col_letter="R" の場合、R4:R(3+span_div) を読む。
    """
    addr = f"{col_letter}4:{col_letter}{3 + span_div}"
    raw = sheet.range(addr).options(np.array, dtype=object).value
    flat = np.ravel(raw)
    if flat.size < span_div:
        raise ValueError(
            f"{addr} の要素数 {flat.size} が span_div={span_div} より少ないです。"
        )

    vec = np.zeros(span_div, dtype=float)
    for i in range(span_div):
        v = flat[i]
        if v is None:
            vec[i] = 0.0
        else:
            try:
                val = float(v)
                if not np.isfinite(val):
                    val = 0.0
                vec[i] = val
            except Exception:
                vec[i] = 0.0
    return vec


def _read_col_ffill(
    wing, row_start: int, row_end: int, col: int, *, default=0.0
) -> np.ndarray:
    """
    Excel の 1 列を object で読み、None/空白/非数を直前値で forward fill して float 配列で返す。
    先頭が欠損の場合は default を使う。
    """
    raw = (
        wing.range((row_start, col), (row_end, col))
        .options(np.array, dtype=object)
        .value
    )
    v = np.ravel(raw)

    out = np.empty(v.size, dtype=float)
    last = None

    for i, x in enumerate(v):
        # None/空文字は欠損扱い
        if x is None or (isinstance(x, str) and x.strip() == ""):
            if last is None:
                out[i] = float(default)
                last = out[i]
            else:
                out[i] = float(last)
            continue

        # 数値化を試みる
        try:
            fx = float(x)
            if not np.isfinite(fx):
                raise ValueError("non-finite")
            out[i] = fx
            last = fx
        except Exception:
            # 非数（文字等）は欠損扱いで forward fill
            if last is None:
                out[i] = float(default)
                last = out[i]
            else:
                out[i] = float(last)

    return out


def read_span_arrays_from_sheet(
    wing,
    n: int,
    row0: int = 208,
    col_y: int = 16,
    col_z: int = 17,
    col_phi: int = 13,
    col_dy: int = 4,  # D列 = 4
):
    # 端点は n+1 点必要
    y_edge = _read_col_ffill(wing, row0, row0 + n, col_y, default=0.0)
    z_edge = _read_col_ffill(wing, row0, row0 + n, col_z, default=0.0)
    phi_edge_deg = _read_col_ffill(wing, row0, row0 + n, col_phi, default=0.0)

    # パネル中心（n点）
    y = 0.5 * (y_edge[:-1] + y_edge[1:])
    z = 0.5 * (z_edge[:-1] + z_edge[1:])
    phi = (pi / 180.0) * 0.5 * (phi_edge_deg[:-1] + phi_edge_deg[1:])

    # dy は n 点必要
    dy_right = _read_col_ffill(wing, row0, row0 + n - 1, col_dy, default=0.0)
    if dy_right.size != n:
        raise ValueError(f"dy_right size {dy_right.size} != n {n}")

    ds_vec = 0.5 * dy_right
    return y, z, phi, ds_vec


def TR797_1():
    try:
        wb = xw.Book.caller()
    except Exception:
        wb = xw.books.active  # 既に開いているブックを使う
    sheet = wb.sheets.active

    span_div = int(sheet.range("N5").value)
    beta_val = sheet.range("G6").value
    beta = float(beta_val) if beta_val is not None else None
    hE = float(sheet.range("C6").value)
    Lift = float(sheet.range("F6").value)
    U = float(sheet.range("D6").value)
    rho = float(sheet.range("N8").value)
    b = float(sheet.range("N6").value)
    le = b / 2.0

    y_panel, z_panel, phi_panel, ds_vec = read_span_arrays_from_sheet(
        sheet,
        n=span_div,
        row0=208,
        col_y=16,
        col_z=17,
        col_phi=14,
        col_dy=4,
    )

    phi_zero = np.zeros(span_div)

    # 節点入力（Q4:Q13）。0..le にクランプし、端点を必ず含める
    raw = sheet.range("Q4:Q13").options(np.array, dtype=object).value
    arr = np.array([v for v in np.ravel(raw) if v is not None], dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = np.clip(arr, 0.0, le)
    arr = np.unique(np.round(arr, 10))
    arr.sort()
    if arr.size == 0:
        knots_y = np.array([0.0, le])
    else:
        if arr[0] > 0.0:
            arr = np.insert(arr, 0, 0.0)
        if arr[-1] < le:
            arr = np.append(arr, le)
        if arr.size == 1:
            arr = np.array([0.0, le])
        knots_y = arr

    table, summary, ky, Gamma_ctrl = TR797_poly_py(
        span_div=span_div,
        beta=beta,
        hE=hE,
        Lift=Lift,
        le=le,
        U=U,
        rho=rho,
        knots_y=knots_y,
        ds_vec=ds_vec,
        y_panel=y_panel,
        z=z_panel,
        phi=phi_panel,
    )

    table_eclipse, summary_eclipse, ky_eclipse, Gamma_ctrl_eclipse = TR797_poly_py(
        span_div=span_div,
        beta=1.0,
        hE=10000.0,  # 十分大きい値で地面効果を無視
        Lift=Lift,
        le=le,
        U=U,
        rho=rho,
        knots_y=knots_y,
        ds=None,
        phi=phi_zero,
        z=None,
    )

    # 出力（TR797_1 と同じ場所）
    sheet.range("N22").value = summary["e"]
    sheet.range("N23").value = summary["Lift"]
    sheet.range("N24").value = summary["Di"]
    sheet.range("N25").value = summary["BendingMoment"]
    sheet.range("N26").value = summary["beta"]
    sheet.range("I22").options(transpose=True).value = Gamma_ctrl
    sheet.range("H22").options(transpose=True).value = Gamma_ctrl_eclipse


def TR797_2():
    try:
        wb = xw.Book.caller()
    except Exception:
        wb = xw.books.active  # 既に開いているブックを使う
    sheet = wb.sheets.active

    span_div = int(sheet.range("N5").value)
    beta_val = sheet.range("G7").value
    beta = float(beta_val) if beta_val is not None else None
    hE = float(sheet.range("C7").value)
    Lift = float(sheet.range("F7").value)
    U = float(sheet.range("D7").value)
    rho = float(sheet.range("N8").value)
    b = float(sheet.range("N6").value)  # スパン
    le = b / 2.0

    col_y_num = excel_col_to_num("AB")
    col_z_num = excel_col_to_num("AC")
    col_phi_num = excel_col_to_num("Z")
    col_dy_num = excel_col_to_num("D")

    y_panel, z_panel, phi_panel, ds_vec = read_span_arrays_from_sheet(
        sheet,
        n=span_div,
        row0=208,
        col_y=col_y_num,
        col_z=col_z_num,
        col_phi=col_phi_num,
        col_dy=col_dy_num,
    )

    phi_zero = np.zeros(span_div)

    # 節点入力（Q4:Q13）。0..le にクランプし、端点を必ず含める
    raw = sheet.range("Q4:Q13").options(np.array, dtype=object).value
    arr = np.array([v for v in np.ravel(raw) if v is not None], dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = np.clip(arr, 0.0, le)
    arr = np.unique(np.round(arr, 10))
    arr.sort()
    if arr.size == 0:
        knots_y = np.array([0.0, le])
    else:
        if arr[0] > 0.0:
            arr = np.insert(arr, 0, 0.0)
        if arr[-1] < le:
            arr = np.append(arr, le)
        if arr.size == 1:
            arr = np.array([0.0, le])
        knots_y = arr

    table, summary, ky, Gamma_ctrl = TR797_poly_py(
        span_div=span_div,
        beta=beta,
        hE=hE,
        Lift=Lift,
        le=le,
        U=U,
        rho=rho,
        knots_y=knots_y,
        ds_vec=ds_vec,
        y_panel=y_panel,
        z=z_panel,
        phi=phi_panel,
    )

    table_eclipse, summary_eclipse, ky_eclipse, Gamma_ctrl_eclipse = TR797_poly_py(
        span_div=span_div,
        beta=1.0,
        hE=10000.0,  # 十分大きい値で地面効果を無視
        Lift=Lift,
        le=le,
        U=U,
        rho=rho,
        knots_y=knots_y,
        ds=None,
        phi=phi_zero,
        z=None,
    )

    # 出力（TR797_1 と同じ場所）
    sheet.range("N41").value = summary["e"]
    sheet.range("N42").value = summary["Lift"]
    sheet.range("N43").value = summary["Di"]
    sheet.range("N44").value = summary["BendingMoment"]
    sheet.range("N45").value = summary["beta"]
    sheet.range("I41").options(transpose=True).value = Gamma_ctrl
    sheet.range("H41").options(transpose=True).value = Gamma_ctrl_eclipse


def TR797_3():
    wb = xw.Book.caller()
    sheet = wb.sheets.active

    span_div = int(sheet.range("N5").value)
    beta_val = sheet.range("G8").value
    beta = float(beta_val) if beta_val is not None else None
    hE = float(sheet.range("C8").value)
    Lift = float(sheet.range("F8").value)
    U = float(sheet.range("D8").value)
    rho = float(sheet.range("N8").value)
    b = float(sheet.range("N6").value)
    le = b / 2.0

    phi_zero = np.zeros(span_div)

    # 節点入力（Q4:Q13）。0..le にクランプし、端点を必ず含める
    raw = sheet.range("Q4:Q13").options(np.array, dtype=object).value
    arr = np.array([v for v in np.ravel(raw) if v is not None], dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = np.clip(arr, 0.0, le)
    arr = np.unique(np.round(arr, 10))
    arr.sort()
    if arr.size == 0:
        knots_y = np.array([0.0, le])
    else:
        if arr[0] > 0.0:
            arr = np.insert(arr, 0, 0.0)
        if arr[-1] < le:
            arr = np.append(arr, le)
        if arr.size == 1:
            arr = np.array([0.0, le])
        knots_y = arr

    table, summary, ky, Gamma_ctrl = TR797_poly_py(
        span_div=span_div,
        beta=beta,
        hE=hE,
        Lift=Lift,
        le=le,
        U=U,
        rho=rho,
        knots_y=knots_y,
        ds=None,
        phi=phi_zero,
        z=None,
    )

    table_eclipse, summary_eclipse, ky_eclipse, Gamma_ctrl_eclipse = TR797_poly_py(
        span_div=span_div,
        beta=1.0,
        hE=10000.0,  # 十分大きい値で地面効果を無視
        Lift=Lift,
        le=le,
        U=U,
        rho=rho,
        knots_y=knots_y,
        ds=None,
        phi=phi_zero,
        z=None,
    )

    sheet.range("N61").value = summary["e"]
    sheet.range("N62").value = summary["Lift"]
    sheet.range("N63").value = summary["Di"]
    sheet.range("N64").value = summary["BendingMoment"]
    sheet.range("N65").value = summary["beta"]
    sheet.range("I61").options(transpose=True).value = Gamma_ctrl
    sheet.range("H61").options(transpose=True).value = Gamma_ctrl_eclipse


# エクセルを探す関数。
def FindWorkbookUpOne(Pattern="*.xlsm"):
    parent_dir = Path(__file__).resolve().parent.parent
    matches = list(parent_dir.glob(Pattern))
    if not matches:
        raise FileNotFoundError(f"No files matching {Pattern} in {parent_dir}")
    # 最新更新ファイル
    return max(matches, key=lambda p: p.stat().st_mtime)


if __name__ == "__main__":
    wb_path = FindWorkbookUpOne("*.xlsm")

    app = xw.App(visible=True)  # Excel を確実に起動
    wb = app.books.open(str(wb_path))  # open を使う（Book(path)より堅い）

    wb.set_mock_caller()
    TR797_1()
