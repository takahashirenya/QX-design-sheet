import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor
import xlwings as xw
import os
import tempfile


# ========= データ前処理 =========
def preprocess(data, smooth_window=1):
    df = pd.DataFrame(data, columns=["alpha", "Cl", "Cd", "Cm", "Re"])

    # Cl/Cd/Cm を移動平均で少しならす（端は NaN になるので後で dropna）
    for col in ["Cl", "Cd", "Cm"]:
        df[col] = df[col].rolling(window=smooth_window, center=True).mean()

    df = df.dropna().reset_index(drop=True)
    return df


# ========= 設計行列 =========
def build_design_matrix(
    alpha, Re, mode="Cl", deg_alpha=10, deg_re=3, include_cross=True
):
    """
    alpha [deg]、Re(そのまま)を入れると、回帰用の特徴行列を返す。
    Cl: α^1..α^deg_alpha
    Cd/Cm: α^1..α^deg_alpha, (log10Re)^1..^deg_re, α·log10Re, α^2·log10Re
    """
    alpha = np.array(alpha, dtype=float)
    Re_log = np.log10(np.array(Re, dtype=float))

    # α^1 .. α^deg_alpha
    A_alpha = [alpha**j for j in range(1, deg_alpha + 1)]

    if mode == "Cl":
        A = np.vstack(A_alpha).T  # shape (N, deg_alpha)
    else:
        # log10(Re)^1 .. log10(Re)^deg_re
        A_re = [Re_log**j for j in range(1, deg_re + 1)]
        features = A_alpha + A_re
        if include_cross:
            features += [alpha * Re_log, (alpha**2) * Re_log]
        A = np.vstack(features).T

    return A


# ========= 設計行列の列名（build_design_matrix と同じ順番） =========
def get_feature_names(mode="Cl", deg_alpha=10, deg_re=3, include_cross=True):
    """
    build_design_matrix()が返す列順と一致するfeature名リストを返す。
    ※切片は含まない。
    """
    names = []

    # α^1 .. α^deg_alpha
    for i in range(1, deg_alpha + 1):
        names.append(f"alpha^{i}")

    if mode != "Cl":
        # logRe^1 .. logRe^deg_re
        for j in range(1, deg_re + 1):
            names.append(f"logRe^{j}")

        if include_cross:
            names.append("alpha*logRe")
            names.append("alpha^2*logRe")

    return names


# ========= 回帰（ロバスト） =========
def robust_poly_fit(alpha, Re, y, mode="Cl"):
    """
    HuberRegressor + StandardScaler でロバストフィットする。
    戻り値: (model, scaler)
    """
    A = build_design_matrix(alpha, Re, mode)
    scaler = StandardScaler()
    A_scaled = scaler.fit_transform(A)

    model = HuberRegressor(alpha=1e-4, max_iter=5000)
    model.fit(A_scaled, y)

    return model, scaler


# ========= StandardScaler込み係数 → 生の式係数に戻す =========
def unscale_model(model, scaler, feature_names):
    """
    scikit-learnの
        y = intercept_scaled + coef_scaled · ((X_raw - mean)/scale)
    を

        y = intercept_unscaled + sum_j coef_unscaled[j] * X_raw[j]

    の形に変換する。

    return:
        intercept_unscaled (float)
        coefs_unscaled (dict: {feature_name: coef})
    """
    coef_scaled = model.coef_  # shape (n_features,)
    intercept_scaled = model.intercept_
    mean = scaler.mean_  # shape (n_features,)
    scale = scaler.scale_  # shape (n_features,)

    # 各特徴の係数を元スケールに戻す
    coef_unscaled_arr = coef_scaled / scale  # elementwise

    # 切片を補正
    intercept_unscaled = intercept_scaled - np.sum(coef_scaled * mean / scale)

    # feature_name と対応づける
    coefs_unscaled = {
        feature_names[i]: coef_unscaled_arr[i] for i in range(len(feature_names))
    }

    return intercept_unscaled, coefs_unscaled


# ========= データ読み込み (XFLR5 出力) =========
def load_xflr5_data(
    foil_name,
    Re_min,
    Re_max,
    Re_delta,
    alpha_min,
    alpha_max,
    path=".",
    Ncrit=5.0,
):
    """
    XFLR5のPolarフォルダ内テキストをまとめて読む。
    期待フォーマット:
      ..._Re0.400_M0.00_N5.0.txt みたいなやつ
    中身は10行目以降が数値データで [alpha, Cl, Cd, ..., Cm, ...] みたいな列順想定。
    """
    data = []
    Re_values = np.arange(Re_min, Re_max + Re_delta, Re_delta)

    for Re_val in Re_values:
        file_name = f"{foil_name}_T1_Re{Re_val/1e6:.3f}_M0.00_N{Ncrit:.1f}.txt"
        file_path = os.path.join(path, f"{foil_name}", file_name)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # 10行目以降にデータがある想定
            for line in lines[10:]:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    alpha = float(parts[0])
                    Cl = float(parts[1])
                    Cd = float(parts[2])
                    Cm = float(parts[4])
                except ValueError:
                    # 数値行じゃない場合スキップ
                    continue

                if alpha_min <= alpha <= alpha_max:
                    data.append([alpha, Cl, Cd, Cm, Re_val])

        except FileNotFoundError:
            # 指定Reのファイルが無い場合はスキップ
            continue

    return np.array(data)


# ========= 係数を Excel に書き出す（Excelでそのまま使える式） =========
def export_coefficients(
    sheet,
    model_Cl,
    scaler_Cl,
    model_Cd,
    scaler_Cd,
    model_Cm,
    scaler_Cm,
    deg_alpha=10,
    deg_re=3,
    include_cross=True,
):
    """
    Excelに『そのまま使える形』の係数を出力する。
    出力するのは y = intercept + Σ coef_i * feature_i の intercept と coef_i。

    featureの並び:
       alpha^1, alpha^2, ..., logRe^1, ..., alpha*logRe, alpha^2*logRe
    logRe は log10(Re) を意味する、とセルに注釈も書く。
    """

    # --- まず各モデルごとに feature 名リストを作る
    fn_Cl = get_feature_names("Cl", deg_alpha, deg_re, include_cross)
    fn_Cd = get_feature_names("Cd", deg_alpha, deg_re, include_cross)
    fn_Cm = get_feature_names("Cm", deg_alpha, deg_re, include_cross)

    # --- スケール解除して "生の式" の係数を得る
    b_Cl, w_Cl = unscale_model(model_Cl, scaler_Cl, fn_Cl)
    b_Cd, w_Cd = unscale_model(model_Cd, scaler_Cd, fn_Cd)
    b_Cm, w_Cm = unscale_model(model_Cm, scaler_Cm, fn_Cm)

    # --- すべての特徴名のユニオンを列として並べたい
    all_features = []
    for lst in [fn_Cl, fn_Cd, fn_Cm]:
        for name in lst:
            if name not in all_features:
                all_features.append(name)

    # 先頭に "intercept" 列
    header = ["model", "intercept"] + all_features

    def make_row(model_name, b, w_dict):
        row = [model_name, b]
        for feat in all_features:
            row.append(w_dict.get(feat, ""))  # そのモデルに存在しない項は空欄
        return row

    row_Cl = make_row("Cl", b_Cl, w_Cl)
    row_Cd = make_row("Cd", b_Cd, w_Cd)
    row_Cm = make_row("Cm", b_Cm, w_Cm)

    table = [header, row_Cl, row_Cd, row_Cm]

    # Excel書き込み
    sheet.range("S4").value = table


# ========= Reごとのフィットと元データの比較（Excelに画像） =========
def compare_fit_at_Re_to_excel(
    sheet, df, model, scaler, mode="Cl", Re_target=400000, cell="Q15"
):
    mask = np.isclose(df["Re"], Re_target)
    alpha_sel = df.loc[mask, "alpha"].values
    y_sel = df.loc[mask, mode].values
    Re_sel = df.loc[mask, "Re"].values

    if len(alpha_sel) == 0:
        sheet.range(cell).value = f"⚠️ Re={Re_target:.1e} のデータなし"
        return

    # 予測
    A = build_design_matrix(alpha_sel, Re_sel, mode)
    A_scaled = scaler.transform(A)
    y_pred = model.predict(A_scaled)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(alpha_sel, y_sel, label="Raw data", s=20)
    ax.plot(alpha_sel, y_pred, label="Fitted curve")
    ax.set_xlabel("Alpha [deg]")
    ax.set_ylabel(mode)
    ax.set_title(f"{mode} vs Alpha @ Re={Re_target:.1e}")
    ax.legend()
    ax.grid(True)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        fig.savefig(tmpfile.name, format="png", bbox_inches="tight")
        sheet.pictures.add(
            tmpfile.name,
            name=f"{mode}_plot",
            update=True,
            left=sheet.range(cell).left,
            top=sheet.range(cell).top,
        )
    plt.close(fig)


# ========= L/D vs Cl プロット（Excelに画像） =========
def compare_LoverD_to_excel(
    sheet,
    df,
    model_Cl,
    scaler_Cl,
    model_Cd,
    scaler_Cd,
    Re_target=400000,
    cell="Q40",
):
    mask = np.isclose(df["Re"], Re_target)
    alpha_sel = df.loc[mask, "alpha"].values
    Cl_raw = df.loc[mask, "Cl"].values
    Cd_raw = df.loc[mask, "Cd"].values
    Re_sel = df.loc[mask, "Re"].values

    if len(alpha_sel) == 0:
        sheet.range(cell).value = f"⚠️ Re={Re_target:.1e} のデータなし"
        return

    # フィット値
    A_Cl = build_design_matrix(alpha_sel, Re_sel, "Cl")
    Cl_fit = model_Cl.predict(scaler_Cl.transform(A_Cl))

    A_Cd = build_design_matrix(alpha_sel, Re_sel, "Cd")
    Cd_fit = model_Cd.predict(scaler_Cd.transform(A_Cd))

    LoverD_raw = Cl_raw / Cd_raw
    LoverD_fit = Cl_fit / Cd_fit

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(Cl_raw, LoverD_raw, label="Raw data", s=20)
    ax.plot(Cl_fit, LoverD_fit, label="Fitted curve")
    ax.set_xlabel("Cl")
    ax.set_ylabel("L/D")
    ax.set_title(f"L/D vs Cl @ Re={Re_target:.1e}")
    ax.legend()
    ax.grid(True)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        fig.savefig(tmpfile.name, format="png", bbox_inches="tight")
        sheet.pictures.add(
            tmpfile.name,
            name="LoverD_plot",
            update=True,
            left=sheet.range(cell).left,
            top=sheet.range(cell).top,
        )
    plt.close(fig)


# ========= 翼型 dat 読み込み =========
def load_airfoil_dat(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    coords = []
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                x = float(parts[0])
                z = float(parts[1])
                coords.append([x, z])
            except ValueError:
                continue

    coords = np.array(coords, dtype=float)
    return coords[:, 0], coords[:, 1]


# ========= キャンバーライン（平均ライン） =========
def compute_camber_line(x, z):
    """
    x,z はTE→LE→TE順の一周座標を想定。
    上側と下側に分けて、同じxでの(zu+zl)/2を取って中心線を作る。
    """
    num = len(x)
    num_up = num // 2  # 雑に前半=上面, 後半=下面 として扱ってる
    x_up, z_up = x[:num_up], z[:num_up]
    x_lo, z_lo = x[num_up:], z[num_up:]

    zc_pts = []
    for xi, zi in zip(x_up, z_up):
        # 下面側で xi に近い2点を探して線形補間
        idx = np.searchsorted(x_lo, xi)
        if idx == 0 or idx >= len(x_lo):
            continue
        x1, x2 = x_lo[idx - 1], x_lo[idx]
        z1, z2 = z_lo[idx - 1], z_lo[idx]
        z_interp = z1 + (z2 - z1) * (xi - x1) / (x2 - x1)

        zc_pts.append([xi, 0.5 * (zi + z_interp)])

    zc_pts = np.array(zc_pts, dtype=float)
    return zc_pts[:, 0], zc_pts[:, 1]


# ========= 翼型+キャンバーラインをExcelに貼る =========
def plot_airfoil_with_camber(sheet, file_path, cell="I3"):
    x, z = load_airfoil_dat(file_path)
    xc, zc = compute_camber_line(x, z)

    chord = np.max(x) - np.min(x)
    thickness = np.max(z) - np.min(z)
    aspect_ratio = thickness / chord if chord != 0 else 0.2

    fig_width = 6
    fig_height = fig_width * aspect_ratio

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.plot(x, z, linewidth=1.2)
    ax.plot(xc, zc, linestyle="--", linewidth=1.0)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(z) - 0.05 * thickness, np.max(z) + 0.05 * thickness)

    ax.axis("off")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        fig.savefig(tmpfile.name, format="png", bbox_inches="tight", pad_inches=0.02)
        sheet.pictures.add(
            tmpfile.name,
            name="airfoil_plot",
            update=True,
            left=sheet.range(cell).left,
            top=sheet.range(cell).top,
        )

    plt.close(fig)


def compute_thickness_at_spar(x, z, spar):
    """
    弦方向位置 spar(0〜1) での翼厚(上面-下面)を返す
    """
    num = len(x)
    num_up = num // 2
    x_up, z_up = x[:num_up], z[:num_up]  # たぶん TE→LE
    x_lo, z_lo = x[num_up:], z[num_up:]  # たぶん LE→TE

    # 上面側を昇順に並べたいので反転してinterp
    zu = np.interp(spar, x_up[::-1], z_up[::-1])
    zl = np.interp(spar, x_lo, z_lo)
    return zu - zl


def airfoil_geometry(sheet, file_path):
    # datファイルから座標を読み込み
    x, z = load_airfoil_dat(file_path)

    # Excelからパラメータ読み込み
    start = sheet.range("R19").value  # 上面プランク位置(弦方向 0-1)
    Fin = sheet.range("R20").value  # 下面プランク位置(弦方向 0-1)
    spar = sheet.range("R21").value  # 桁位置(弦方向 0-1)

    # 翼型周長（折れ線で積分）
    perimeter = np.sum(np.sqrt(np.diff(x) ** 2 + np.diff(z) ** 2))

    # リブ面積（左右対称っぽく前提して台形和）
    area = 0.0
    # 前半: 上面 TE→LE, 後半: 下面 LE→TE という素朴な前提
    half = (len(x) - 2) // 2
    for i in range(half):
        # x[i]~x[i+1] を上面, 反対側を下面で対応づけて台形近似
        area += (
            0.5
            * ((z[i + 1] - z[-(i + 2)]) + (z[i] - z[-(i + 1)]))
            * abs(x[i + 1] - x[i])
        )

    thickness = compute_thickness_at_spar(x, z, spar)

    # 必要ならここで sheet.range(...).value = perimeter 等も書ける
    # 今は呼び出し側で使ってないので return のみ
    return x, z, perimeter, area, thickness


# ========= Excel呼び出しメイン関数 =========
def importAirfoil():
    wb = xw.Book.caller()
    sheet = wb.sheets.active

    foil_name = sheet.range("C3").value
    if not foil_name:
        sheet.range("F3").value = "C3に翼型名を入力してください"
        return

    # C3に "NACA4412.dat" みたいに .dat 付きで書いてくるかもしれないので除去
    if foil_name.lower().endswith(".dat"):
        foil_name = foil_name[:-4]

    Re_min = sheet.range("C7").value
    Re_max = sheet.range("D7").value
    Re_delta = sheet.range("E7").value
    alpha_min = sheet.range("C6").value
    alpha_max = sheet.range("D6").value
    Ncrit = sheet.range("C9").value
    target_Re = sheet.range("D11").value

    base_dir = os.path.dirname(wb.fullname)
    airfoil_dir = os.path.join(base_dir, "Polar")

    raw_data = load_xflr5_data(
        foil_name,
        Re_min,
        Re_max,
        Re_delta,
        alpha_min,
        alpha_max,
        path=airfoil_dir,
        Ncrit=Ncrit,
    )

    if raw_data.size == 0:
        sheet.range("G4").value = "データが読み込めませんでした"
        return

    df = preprocess(raw_data, smooth_window=1)

    alpha = df["alpha"].values
    Cl = df["Cl"].values
    Cd = df["Cd"].values
    Cm = df["Cm"].values
    Re = df["Re"].values

    # それぞれロバスト回帰
    model_Cl, scaler_Cl = robust_poly_fit(alpha, Re, Cl, mode="Cl")
    model_Cd, scaler_Cd = robust_poly_fit(alpha, Re, Cd, mode="Cd")
    model_Cm, scaler_Cm = robust_poly_fit(alpha, Re, Cm, mode="Cm")

    # Excelでそのまま使える係数をS4から書き出し
    export_coefficients(
        sheet,
        model_Cl,
        scaler_Cl,
        model_Cd,
        scaler_Cd,
        model_Cm,
        scaler_Cm,
        deg_alpha=10,
        deg_re=3,
        include_cross=True,
    )

    # 可視化: 指定Reでのフィット vs 元データ
    compare_fit_at_Re_to_excel(
        sheet, df, model_Cl, scaler_Cl, "Cl", Re_target=target_Re, cell="C15"
    )
    compare_fit_at_Re_to_excel(
        sheet, df, model_Cd, scaler_Cd, "Cd", Re_target=target_Re, cell="I15"
    )
    compare_fit_at_Re_to_excel(
        sheet, df, model_Cm, scaler_Cm, "Cm", Re_target=target_Re, cell="C31"
    )
    compare_LoverD_to_excel(
        sheet,
        df,
        model_Cl,
        scaler_Cl,
        model_Cd,
        scaler_Cd,
        Re_target=target_Re,
        cell="I31",
    )

    # 翼型の形状とキャンバーラインをExcelに貼る
    dat_path = os.path.join(base_dir, "Airfoil", foil_name + ".dat")
    if os.path.exists(dat_path):
        x, z, perimeter, area, thickness = airfoil_geometry(sheet, dat_path)
        plot_airfoil_with_camber(sheet, dat_path, cell="I3")

    sheet.range("G4").value = "回帰完了"
