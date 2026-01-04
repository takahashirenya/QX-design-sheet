#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
楕円(ゆるいテーパー対応)パイプを 6° で切るための巻き付けテンプレート生成スクリプト
出力: A4 SVG (210 x 297 mm) を 1:1 で作図

■ 近似モデル
- 断面は楕円: 長軸 a, 短軸 b [mm]
- テーパーは「大径側(a0,b0) と 小径側(a1,b1) の中間」を用いた近似
- 切断面はパイプ軸に対し θ(=6°) 傾いた平面
- パイプへ巻いたとき、周方向 φ に対する高さは z(φ) ≈ tan(θ) * r_eff(φ)
- 楕円の方向半径 r_eff(φ) を支持関数近似で r_eff(φ) ≈ (a*b)/sqrt((b*cosφ)^2 + (a*sinφ)^2)
- 展開幅 W は中間楕円の周長(ラマヌジャン近似) + seam_gap(=13mm)
- 幅方向 x を φ に線形対応: x = (φ / 2π) * (W - seam_gap)

■ パラメータを最上部で設定してください
"""

import math
from pathlib import Path

# ===================== ユーザー設定 =====================
PARAMS = {
    # 楕円寸法（大径側：パイプの「根元側」）
    "a0_mm": 70.0,   # 長軸(大径側) [mm]
    "b0_mm": 40.0,   # 短軸(大径側) [mm]

    # 楕円寸法（小径側：パイプの「先端側」）
    "a1_mm": 64.0,   # 長軸(小径側) [mm]
    "b1_mm": 36.0,   # 短軸(小径側) [mm]

    # パイプ全長（テンプレートは長さ方向の“高さ目安”のみ必要。値は注記表示用）
    "pipe_length_mm": 300.0,

    # 切断角度（度）
    "cut_angle_deg": 6.0,

    # 巻き合わせの余白・重ねシールなど
    "seam_gap_mm": 13.0,        # ユーザー要望の「切って減る分の考慮」＝開ける隙間
    "glue_tab_mm": 10.0,        # 片側に付ける貼り合わせタブ幅（任意）
    "margin_mm": 10.0,          # 用紙の外周マージン
    "tick_every_mm": 10.0,      # スケール線のピッチ
}

# 出力ファイル名
OUTPUT_SVG = "template_ellipse_pipe_6deg.svg"

# 用紙サイズ（A4）
A4_W_MM = 210.0
A4_H_MM = 297.0

# サンプリング密度（滑らかさ）
N_SAMPLES = 1200  # φ の分割数（多いほど滑らか）


def ramanujan_perimeter(a, b):
    """楕円周長のラマヌジャン近似 (第1近似)"""
    h = ((a - b) ** 2) / ((a + b) ** 2)
    return math.pi * (a + b) * (1 + (3 * h) / (10 + math.sqrt(4 - 3 * h)))


def r_ellipse_directional(a, b, phi):
    """
    楕円の方向半径（近似）
    中心→角度φ方向の境界までの距離。
    支持関数の近似式: r(φ) ≈ (a*b) / sqrt( (b*cosφ)^2 + (a*sinφ)^2 )
    """
    c = math.cos(phi)
    s = math.sin(phi)
    denom = math.sqrt((b * c) ** 2 + (a * s) ** 2)
    return (a * b) / denom if denom > 1e-12 else min(a, b)


def mm(v):  # SVG mm単位のフォーマット
    return f"{v:.3f}mm"


def main():
    p = PARAMS

    # 中間断面（テーパーの近似）
    a_mid = 0.5 * (p["a0_mm"] + p["a1_mm"])
    b_mid = 0.5 * (p["b0_mm"] + p["b1_mm"])

    # 中間断面の周長 + シーム隙間
    perim_mid = ramanujan_perimeter(a_mid, b_mid)  # [mm]
    W = perim_mid + p["seam_gap_mm"]               # 展開の横幅（用紙上での幅）

    # 切断角
    theta = math.radians(p["cut_angle_deg"])
    tan_theta = math.tan(theta)

    # φ→x の対応: φ∈[0, 2π] を x∈[0, W - seam_gap] に線形対応
    # 右端に貼り合わせタブを付けるため W に glue_tab も考慮し配置
    glue = p["glue_tab_mm"]
    margin = p["margin_mm"]

    # 作図領域の計算（横に W + タブ、縦は余裕を見て）
    draw_w = W + glue + margin * 2
    # 振幅の見込み: tanθ * max(r) ~ tanθ * a_mid （長軸方向に傾ける前提）
    amp_est = tan_theta * a_mid
    draw_h = max(A4_H_MM - margin * 2, 2 * amp_est + 40.0) + margin * 2

    # ページチェック
    if draw_w + 1e-6 > A4_W_MM or draw_h + 1e-6 > A4_H_MM:
        print("⚠ 注意: A4 に収まりません。パラメータを見直すか、印刷時に用紙設定を変更してください。")

    # SVG ヘッダ
    lines = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{mm(A4_W_MM)}" height="{mm(A4_H_MM)}" viewBox="0 0 {A4_W_MM:.3f} {A4_H_MM:.3f}">')

    # 白背景
    lines.append(f'<rect x="0" y="0" width="{A4_W_MM:.3f}" height="{A4_H_MM:.3f}" fill="white"/>')

    # 外枠
    lines.append(f'<rect x="{mm(margin/2)}" y="{mm(margin/2)}" width="{mm(A4_W_MM-margin)}" height="{mm(A4_H_MM-margin)}" fill="none" stroke="black" stroke-width="0.2"/>')

    origin_x = margin
    origin_y = margin

    # 情報ラベル
    label = (f'Elliptical pipe cut template (approx)\n'
             f'a0={p["a0_mm"]}mm, b0={p["b0_mm"]}mm / a1={p["a1_mm"]}mm, b1={p["b1_mm"]}mm  '
             f'| mid: a={a_mid:.2f}, b={b_mid:.2f}  | angle={p["cut_angle_deg"]}°  '
             f'| seam gap={p["seam_gap_mm"]}mm  | glue tab={p["glue_tab_mm"]}mm  '
             f'| pipe L={p["pipe_length_mm"]}mm')
    lines.append(f'<text x="{mm(origin_x)}" y="{mm(origin_y - 2)}" font-size="3.2" font-family="monospace">{label}</text>')

    # 作図領域の枠
    plot_x = origin_x
    plot_y = origin_y + 8
    plot_w = W + glue
    plot_h = draw_h - (plot_y + margin)

    lines.append(f'<rect x="{plot_x:.3f}" y="{plot_y:.3f}" width="{plot_w:.3f}" height="{plot_h:.3f}" fill="none" stroke="black" stroke-width="0.2"/>')

    # スケール線（10mmごと）
    tick = p["tick_every_mm"]
    # 垂直グリッド
    gx = plot_x
    while gx <= plot_x + plot_w + 0.001:
        lines.append(f'<line x1="{gx:.3f}" y1="{plot_y:.3f}" x2="{gx:.3f}" y2="{plot_y+plot_h:.3f}" stroke="#ddd" stroke-width="0.2"/>')
        gx += tick
    # 水平グリッド
    gy = plot_y
    while gy <= plot_y + plot_h + 0.001:
        lines.append(f'<line x1="{plot_x:.3f}" y1="{gy:.3f}" x2="{plot_x+plot_w:.3f}" y2="{gy:.3f}" stroke="#ddd" stroke-width="0.2"/>')
        gy += tick

    # 貼り合わせタブのガイド
    tab_x0 = plot_x + (W)  # 右端 glue 区画
    lines.append(f'<rect x="{tab_x0:.3f}" y="{plot_y:.3f}" width="{glue:.3f}" height="{plot_h:.3f}" fill="none" stroke="#999" stroke-dasharray="2,2" stroke-width="0.2"/>')
    lines.append(f'<text x="{tab_x0 + glue/2:.3f}" y="{plot_y + 5:.3f}" font-size="3" text-anchor="middle" fill="#555">GLUE TAB ({glue:.0f} mm)</text>')

    # シーム位置（左端）を点線で
    seam_x = plot_x
    lines.append(f'<line x1="{seam_x:.3f}" y1="{plot_y:.3f}" x2="{seam_x:.3f}" y2="{plot_y+plot_h:.3f}" stroke="#555" stroke-dasharray="2,1" stroke-width="0.3"/>')
    lines.append(f'<text x="{seam_x + 2:.3f}" y="{plot_y + 5:.3f}" font-size="3" fill="#555">SEAM GAP {p["seam_gap_mm"]:.0f} mm→</text>')

    # 基準高さ（中央）
    mid_y = plot_y + plot_h/2
    lines.append(f'<line x1="{plot_x:.3f}" y1="{mid_y:.3f}" x2="{plot_x+plot_w:.3f}" y2="{mid_y:.3f}" stroke="#aaa" stroke-dasharray="3,2" stroke-width="0.2"/>')
    lines.append(f'<text x="{plot_x + 2:.3f}" y="{mid_y - 2:.3f}" font-size="3" fill="#777">centerline</text>')

    # 切断ラインを生成（長軸方向へ傾ける想定）
    pts = []
    usable_w = W - p["seam_gap_mm"]  # 実際に周方向に相当する幅
    for i in range(N_SAMPLES + 1):
        t = i / N_SAMPLES
        phi = 2 * math.pi * t  # 周方向
        x = plot_x + t * usable_w
        r = r_ellipse_directional(a_mid, b_mid, phi)
        z = tan_theta * r  # 高さ偏差（+/-）
        y = mid_y - z  # 上がマイナス方向のため
        pts.append((x, y))

    # パス描画
    d = "M " + " L ".join(f"{x:.3f},{y:.3f}" for x, y in pts)
    lines.append(f'<path d="{d}" fill="none" stroke="black" stroke-width="0.6"/>')

    # 端部の注意書き
    note = ("Wrap template: align SEAM at pipe seam.\n"
            "Cut along the bold curve. Keep saw kerf in mind.\n"
            "For tapered oval tubes, this is a small-angle practical approximation.")
    note_y0 = plot_y + plot_h - 18
    for j, ln in enumerate(note.split("\n")):
        lines.append(f'<text x="{mm(plot_x)}" y="{mm(note_y0 + j*4)}" font-size="3.2" font-family="monospace">{ln}</text>')

    # SVG 終了
    lines.append("</svg>")

    Path(OUTPUT_SVG).write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ 出力しました: {OUTPUT_SVG}")
    print("※ 100%スケールで印刷してください（ページに合わせる等の拡縮は無効化）。")


if __name__ == "__main__":
    main()
