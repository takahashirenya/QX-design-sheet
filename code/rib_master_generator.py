"""
翼型のデータを読み込み、型紙を出力するプログラム
mode = "print" : 印刷用(DXF)
       "lasercut" : レーザーカット用(DXF)
       "jig" : ジグレーザーカット用(SVG)
       "plot": matplotlibでプロット
"""

import os
import xlwings as xw
from datetime import datetime
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import ezdxf
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
from ezdxf.enums import TextEntityAlignment
import itertools
from ezdxf.addons.importer import Importer

from LLT import FindWorkbookUpOne
import classes.Config as cf
from classes.Geometry import GeometricalAirfoil

# ========= 設定 =========
mode = "jig"  # "print", "lasercut", "jig", "stilo_lazer"
preview = False  # matplotlibでプレビューを表示するか
all_at_once = False  # 一つの図面/ファイルにまとめるか

protrude_length = 1  # 線引き線の飛び出し長さ

# "lasercut"の場合のパラメータ
peephole_length = 10  # 線引き用に開ける四角穴の一辺長さ（peephole: のぞき穴）
lazercutter_workspace_width = 600  # レーザーカッターのワーク幅
TRI_DEPTH = 8.0  # ツメの張り出し量 [mm]（左右の食い違い）
TRI_APEX_POS = 0.5  # 三角の頂点のY位置（0=下端, 1=上端の間の比率）
CARBON_TIP_INSET = 0.45  # [mm] 内側オフセット量


# "jig"の場合のパラメータ
channel_width = 60  # チャンネル材の幅
channel_height = 30  # チャンネル材の高さ
torelance = 0.1  # チャンネル材とのはめあい交差
jig_width = 100  # ジグ全体の幅
jig_height = 45  # ジグの四角部分の高さ
spar_height = 140  # リブ付時の桁中心とチャンネル材下部との高さ差

# ==== matplotlib（ツールバー画像の問題回避）====
# ファイル保存は常に非インタラクティブでOK
mpl.rcParams["toolbar"] = "none"
if not preview:
    mpl.use("Agg")

# 出力ファイル名 例：rib_master_print_20210901_123456.dxf
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
if all_at_once:
    file_name = os.path.join(
        cf.Settings.OUTPUTS_PATH, "master", f"rib_master_{mode}_{current_time}.dxf"
    )
else:
    output_dir = os.path.join(
        cf.Settings.OUTPUTS_PATH, "master", f"rib_master_{mode}_{current_time}"
    )
    os.makedirs(output_dir, exist_ok=True)


# ========= ユーティリティ =========
def create_doc(layer_color=1, lineweight=50):
    """DXFドキュメントを作成し、必要レイヤを用意して modelspace を返す"""
    doc = ezdxf.new("R2007", setup=True)
    msp = doc.modelspace()
    # 作図用レイヤ
    try:
        doc.layers.new(
            name="Layer", dxfattribs={"color": layer_color, "lineweight": lineweight}
        )
    except ezdxf.DXFError:
        pass
    # 番号レイヤ（無いとエラーになる）
    try:
        doc.layers.new(name="NumLabel", dxfattribs={"color": 1, "lineweight": 15})
    except ezdxf.DXFError:
        pass
    return doc, msp


def main():
    global file_name
    # エクセルのシートを取得
    wb = xw.Book.caller()
    sht = wb.sheets[cf.Wing.name]

    # シートから値を読み取る
    foil1name_arr = sht.range(cf.Wing.foil1name).expand("down").value
    foil1rate_arr = sht.range(cf.Wing.foil1rate).expand("down").value
    foil2name_arr = sht.range(cf.Wing.foil2name).expand("down").value
    foil2rate_arr = sht.range(cf.Wing.foil2rate).expand("down").value
    chordlen_arr = sht.range(cf.Wing.chordlen).expand("down").value
    taper_arr = sht.range(cf.Wing.taper).expand("down").value
    spar_arr = sht.range(cf.Wing.spar).expand("down").value
    ishalf_arr = [
        item == "half" for item in sht.range(cf.Wing.ishalf).expand("down").value
    ]
    diam_z_arr = sht.range(cf.Wing.diam_z).expand("down").value
    diam_x_arr = sht.range(cf.Wing.diam_x).expand("down").value
    spar_position_arr = sht.range(cf.Wing.spar_position).expand("down").value
    alpha_rib_arr = sht.range(cf.Wing.alpha_rib).expand("down").value
    stringer_arr = sht.range(cf.Wing.stringer).expand("down").value

    ribzai_thickness = sht.range(cf.Wing.ribzai_thickness).value
    plank_thickness = sht.range(cf.Wing.plank_thickness).value
    plank_start = sht.range(cf.Wing.plank_start).value
    plank_end = sht.range(cf.Wing.plank_end).value
    halfline_start = sht.range(cf.Wing.halfline_start).value
    halfline_end = sht.range(cf.Wing.halfline_end).value
    balsatip_length = sht.range(cf.Wing.balsatip_length).value
    carbontip_length = sht.range(cf.Wing.carbontip_length).value
    koenzai_length = sht.range(cf.Wing.koenzai_length).value
    ribset_line = np.array(sht.range(cf.Wing.ribset_line).value)
    channel_distance = np.array(sht.range(cf.Wing.channel_distance).value)
    hole_margin = sht.range(cf.Wing.hole_margin).value
    refline_offset = sht.range(cf.Wing.refline_offset).value

    total_rib_num = len(foil1name_arr)

    # 必要な翼型のdatファイルを呼び出して辞書型配列に格納
    dat_dict = {}
    foilnames = np.unique(np.concatenate((foil1name_arr, foil2name_arr)))

    for foilname in foilnames:
        name = str(foilname).strip()
        base_path = os.path.dirname(wb.fullname)
        fname = os.path.join(base_path, "Airfoil", f"{name}")
        if not os.path.isfile(fname):
            raise FileNotFoundError(f"Airfoil .dat が見つかりません: {fname}")
        dat = np.loadtxt(fname=fname, dtype="float", skiprows=1)
        dat_dict[name] = dat

    if all_at_once:
        doc, msp = create_doc(layer_color=0, lineweight=50)

    for i, id in enumerate(range(total_rib_num)):  # リブ番号の範囲を指定
        chord = chordlen_arr[id]
        taper = taper_arr[id]
        spar = spar_arr[id]
        is_half = ishalf_arr[id]
        diam_z = diam_z_arr[id]
        diam_x = diam_x_arr[id]
        spar_position = spar_position_arr[id]
        foil1name = foil1name_arr[id]
        foil1rate = foil1rate_arr[id]
        foil2name = foil2name_arr[id]
        foil2rate = foil2rate_arr[id]
        alpha_rib = alpha_rib_arr[id]

        do_ribset = False
        if spar in ("0番", "1番", "2番", "3番"):
            do_ribset = True
            ribset_line_num = {"0番": 0, "1番": 1, "2番": 2, "3番": 3}[spar]
            ribset_line_offsets = np.ravel(ribset_line[:, ribset_line_num])
            channel_distances = np.ravel(channel_distance[:, ribset_line_num])

        # 描画の基準点
        point_ref = np.array([spar_position * chord, -(100 + i * 200)])

        if not all_at_once:
            doc, msp = create_doc(layer_color=1, lineweight=50)
            if mode in ("lasercut", "stilo_lazer"):
                point_ref = np.array([spar_position * chord + 10, 200])
            elif mode == "jig":
                point_ref = np.array([jig_width / 2, 0])

        # 翼型のdatデータを取得
        dat_raw = dat_dict[foil1name] * foil1rate + dat_dict[foil2name] * foil2rate
        # 幾何値参照用オブジェクトを作成
        geo = GeometricalAirfoil(dat_raw, chord_ref=chord)
        # プランク情報を取得（未使用変数の警告抑止のため定義は維持）
        offset_arr = np.array([[plank_start, plank_end, plank_thickness]])
        # 桁中心を計算
        spar_x = spar_position * chord
        spar_center = np.array([spar_x, geo.camber(spar_x)])
        # リブの段差オフセットを定義
        offset_base = 0 if is_half else ribzai_thickness
        offset_arr = np.array([[plank_start, plank_end, plank_thickness]])
        # カーボンチップオフセット

        carbon_radius = carbontip_length
        carbontip_thickness = ribzai_thickness

        if is_half:
            carbon_radius = None
            carbontip_thickness = 0

        # リブオフセットプランクとリブ材両方オフセットしている。
        dat_offset = geo.offset_foil(
            offset_base,
            offset_arr,
            carbon_length=carbon_radius,
            carbon_depth=carbontip_thickness,
        ).copy()

        # ハーフリブ処理
        if is_half:
            half_x_start = halfline_start * geo.chord_ref
            half_x_end = halfline_end * geo.chord_ref
            half_start = (
                np.array([abs(half_x_start), geo.y(half_x_start)])
                + geo.nvec(half_x_start) * plank_thickness
            )
            half_end = np.array([abs(half_x_end), geo.y(half_x_end)])
            start_index = nearest_next_index(geo.dat_extended, half_x_start)
            end_index = nearest_next_index(geo.dat_extended, half_x_end)
            dat_out = np.vstack(
                [
                    [half_start],
                    dat_offset[start_index + 1 : end_index + 3],
                    [half_end],
                    [half_start],
                ]
            )
        else:
            dat_out = dat_offset.copy()
        # 桁中心を原点に移動＆迎角だけ回転
        dat_out = rotate_points(dat_out - spar_center, (0, 0), alpha_rib)

        # ===== 作図 =====
        if mode in ("print", "lasercut", "stilo_lazer"):
            # リブ外形
            msp.add_lwpolyline(
                dat_out + point_ref,
                format="xy",
                close=True,
                dxfattribs={"layer": "Layer"},
            )

            # ストリンガー
            for stringer in stringer_arr:
                add_tangedsquare(
                    msp,
                    geo,
                    point_ref,
                    plank_thickness,
                    stringer[0],
                    stringer[1],
                    stringer[2],
                    spar_center,
                    alpha_rib,
                )

            # 後縁円弧
            if not is_half:
                add_TEarc(msp, geo, point_ref, koenzai_length, spar_center, alpha_rib)
                add_TEarc(msp, geo, point_ref, carbontip_length, spar_center, alpha_rib)
                add_TEarc(msp, geo, point_ref, balsatip_length, spar_center, alpha_rib)

            # 桁穴
            theta = np.linspace(0, 2 * np.pi, 300)
            if diam_x != 0:
                x = (diam_x + hole_margin) / 2 * np.cos(theta)
                y = (diam_z + hole_margin) / 2 * np.sin(theta)
                spar_hole = np.vstack([x, y]).T
                attrs = (
                    {"layer": "Layer"}
                    if mode != "stilo_lazer"
                    else {"layer": "Layer", "color": 7}
                )
                msp.add_lwpolyline(
                    spar_hole + point_ref, format="xy", close=True, dxfattribs=attrs
                )

        if mode == "print":
            # ダミーライン
            add_line_inside_foil(msp, dat_out, (0, 0), 90, point_ref)

            if not is_half:
                # 外形（基準形状）
                rotated_outline = rotate_points(
                    geo.dat_ref - spar_center, (0, 0), alpha_rib
                )
                msp.add_lwpolyline(
                    rotated_outline + point_ref,
                    format="xy",
                    close=True,
                    dxfattribs={"layer": "Layer"},
                )
                # コードライン
                rotated_chordline = rotate_points(
                    np.array([[0, 0], [chord, 0]]) - spar_center, (0, 0), alpha_rib
                )
                msp.add_line(
                    rotated_chordline[0] + point_ref,
                    rotated_chordline[1] + point_ref,
                    dxfattribs={"layer": "Layer"},
                )

            # 桁線
            intersections_center = find_line_intersection(dat_out, (0, 0), 0)
            msp.add_line(
                intersections_center[0] + point_ref,
                intersections_center[1] + point_ref,
                dxfattribs={"layer": "Layer", "linetype": "CENTER"},
            )

            # オフセット線
            if taper == "基準":
                intersections = find_line_intersection(dat_out, (refline_offset, 0), 0)
                msp.add_line(
                    intersections[0] + point_ref,
                    intersections[1] + point_ref,
                    dxfattribs={"layer": "Layer", "linetype": "CENTER"},
                )
            if do_ribset:
                for offset in ribset_line_offsets:
                    add_line_inside_foil(msp, dat_out, (offset, 0), 0, point_ref)

            # テキスト
            label_location = np.array([0.1 * chord - spar_x, 0])
            label_text = str(id)
            if taper == "基準":
                label_text += " ref"
            if spar == "端リブ":
                label_text += " end"
            label_height = 15
            info_text = str(np.round(chord * 1e3) / 1e3) + "mm"
            info_height = 10
            msp.add_text(
                label_text, height=label_height, dxfattribs={"layer": "Layer"}
            ).set_placement(
                label_location + point_ref, align=TextEntityAlignment.BOTTOM_LEFT
            )
            msp.add_text(
                info_text, height=info_height, dxfattribs={"layer": "Layer"}
            ).set_placement(
                (label_location[0], label_location[1] - 5) + point_ref,
                align=TextEntityAlignment.TOP_LEFT,
            )

        if mode == "lasercut":
            # ダミーライン
            add_line_inside_foil(
                msp, dat_out, (0, 0), 90, point_ref, 2, 2, peephole=True
            )
            # コードライン
            add_line_inside_foil(
                msp,
                dat_out,
                rotate_points(np.array([0, -spar_center[1]]), (0, 0), alpha_rib),
                90 - alpha_rib,
                point_ref,
                2,
                2,
                peephole=True,
                peephole_max=2.5,
            )
            # 桁線（端だけ出す）
            msp.add_line(
                np.array([0, (diam_z + hole_margin) / 2]) + point_ref,
                np.array([0, (diam_z + hole_margin) / 2 + protrude_length]) + point_ref,
                dxfattribs={"layer": "Layer"},
            )
            msp.add_line(
                np.array([0, -(diam_z + hole_margin) / 2]) + point_ref,
                np.array([0, -(diam_z + hole_margin) / 2 - protrude_length])
                + point_ref,
                dxfattribs={"layer": "Layer"},
            )
            msp.add_line(
                np.array([(diam_x + hole_margin) / 2, 0]) + point_ref,
                np.array([(diam_x + hole_margin) / 2 + protrude_length, 0]) + point_ref,
                dxfattribs={"layer": "Layer"},
            )
            msp.add_line(
                np.array([-(diam_x + hole_margin) / 2, 0]) + point_ref,
                np.array([-(diam_x + hole_margin) / 2 - protrude_length, 0])
                + point_ref,
                dxfattribs={"layer": "Layer"},
            )

            # オフセット線
            if do_ribset:
                for offset in ribset_line_offsets:
                    add_line_inside_foil(
                        msp,
                        dat_out,
                        (offset, 0),
                        0,
                        point_ref,
                        2,
                        2,
                        peephole=True,
                        peephole_max=2,
                    )

            # 文字入れ：色を整数（例：1）で指定、layerは"NumLabel"
            text_location = np.array([0.28 * chord - spar_x, -0.05 * chord])
            text_interval = 7.5
            text_height = 10
            for digit in str(id):
                write_num_with_lines(
                    msp,
                    digit,
                    text_location + point_ref,
                    text_height,
                    color=1,
                    layer="NumLabel",
                )
                text_location += np.array([text_interval, 0])

        if mode == "stilo_lazer":
            add_line_inside_foil(
                msp, dat_out, (0, 0), 90, point_ref, 2, 2, peephole=True
            )
            add_line_inside_foil(
                msp,
                dat_out,
                rotate_points(np.array([0, -spar_center[1]]), (0, 0), alpha_rib),
                90 - alpha_rib,
                point_ref,
                2,
                2,
            )
            # 桁線（白=7）
            for a, b in (
                (
                    np.array([0, (diam_z + hole_margin) / 2]),
                    np.array([0, (diam_z + hole_margin) / 2 + protrude_length]),
                ),
                (
                    np.array([0, -(diam_z + hole_margin) / 2]),
                    np.array([0, -(diam_z + hole_margin) / 2 - protrude_length]),
                ),
                (
                    np.array([(diam_x + hole_margin) / 2, 0]),
                    np.array([(diam_x + hole_margin) / 2 + protrude_length, 0]),
                ),
                (
                    np.array([-(diam_x + hole_margin) / 2, 0]),
                    np.array([-(diam_x + hole_margin) / 2 - protrude_length, 0]),
                ),
            ):
                msp.add_line(
                    a + point_ref,
                    b + point_ref,
                    dxfattribs={"layer": "Layer", "color": 7},
                )
            if do_ribset:
                for offset in ribset_line_offsets:
                    add_line_inside_foil(
                        msp, dat_out, (offset, 0), 0, point_ref, 2, 2, peephole=True
                    )
            # 番号（白=7）
            text_location = np.array([0.25 * chord - spar_x, 0])
            text_interval = 7.5
            text_height = 10
            for digit in str(id):
                write_num_with_lines(
                    msp,
                    digit,
                    text_location + point_ref,
                    text_height,
                    color=7,
                    layer="NumLabel",
                )
                text_location += np.array([text_interval, 0])

        if mode == "jig":
            if all_at_once:
                msp.add_lwpolyline(
                    dat_out + point_ref,
                    format="xy",
                    close=True,
                    dxfattribs={"layer": "Layer"},
                )
            if do_ribset:
                for j, offset in enumerate(ribset_line_offsets):
                    intersection = find_line_intersection(dat_out, (offset, 0), 0)
                    if len(intersection) == 0:
                        continue
                    intersection = intersection[1]
                    dat_section = divide_dat(
                        dat_out, offset - channel_width / 2, offset + channel_width / 2
                    )
                    if is_half and j == 1:
                        dat_section = np.array(
                            [
                                find_line_intersection(
                                    dat_out, (offset - channel_width / 2, 0), 0
                                )[1],
                                (
                                    find_line_intersection(
                                        dat_out, (offset + channel_width / 2, 0), 0
                                    )[1]
                                    if find_line_intersection(
                                        dat_out, (offset + channel_width / 2, 0), 0
                                    ).size
                                    != 0
                                    else dat_out[-1]
                                ),
                            ]
                        )
                    jig_points = np.vstack(
                        [
                            [[-jig_width / 2, 0], [-jig_width / 2, jig_height]],
                            dat_section
                            + np.array([-channel_distances[j], spar_height]),
                            [
                                [jig_width / 2, jig_height],
                                [jig_width / 2, 0],
                                [channel_width / 2 + torelance / 2, 0],
                                [channel_width / 2 + torelance / 2, channel_height],
                                [-channel_width / 2 - torelance / 2, channel_height],
                                [-channel_width / 2 - torelance / 2, 0],
                            ],
                        ]
                    )
                    peak_line = np.array(
                        [intersection, intersection + np.array([0, -protrude_length])]
                    ) + np.array([-channel_distances[j], spar_height])

                    space_between = np.array([jig_width * j, 0])

                    if all_at_once:
                        add_line_inside_foil(msp, dat_out, (offset, 0), 0, point_ref)
                        jig_points += np.array([channel_distances[j], -spar_height])
                        peak_line += np.array([channel_distances[j], -spar_height])
                        space_between = np.array([0, 0])

                    msp.add_lwpolyline(
                        jig_points + point_ref + space_between,
                        format="xy",
                        close=True,
                        dxfattribs={"layer": "Layer", "color": 1},
                    )
                    msp.add_line(
                        peak_line[0] + point_ref + space_between,
                        peak_line[1] + point_ref + space_between,
                        dxfattribs={"layer": "Layer", "color": 1},
                    )

                    text_location = np.array([15.0, 35.0])
                    text_interval = 7.5
                    text_height = 5
                    if all_at_once:
                        text_location += np.array([channel_distances[j], -spar_height])
                    color = 1
                    write_num_with_lines(
                        msp,
                        ("L" if j == 0 else "T"),
                        text_location + point_ref + space_between,
                        text_height,
                        color,
                        layer="NumLabel",
                    )
                    text_location[0] += text_interval
                    for digit in str(id):
                        write_num_with_lines(
                            msp,
                            digit,
                            text_location + point_ref + space_between,
                            text_height,
                            color,
                            layer="NumLabel",
                        )
                        text_location += np.array([text_interval, 0])

        # ====== 保存と分割 ======
        if not all_at_once:
            file_name = os.path.join(output_dir, f"rib_{id}.dxf")
            doc.saveas(file_name)

            # PNG/SVG出力（非インタラクティブ）
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1])
            ctx = RenderContext(doc)
            out = MatplotlibBackend(ax)
            Frontend(ctx, out).draw_layout(msp, finalize=True)
            fig.savefig(os.path.join(output_dir, f"rib_{id}.png"))
            if mode in ("lasercut", "jig", "stilo_lazer"):
                fig.savefig(os.path.join(output_dir, f"rib_{id}.svg"), format="svg")
            plt.close(fig)

            # --- 50%コード位置に分割が必要か？ ---
            need_split = (mode == "lasercut") and (chord > lazercutter_workspace_width)

            # 分割線（可視）は SplitMark レイヤに描く。あとで消す。
            p0 = p1 = None
            if need_split:
                x_cut_local = 0.5 * chord - spar_x
                inters = find_line_intersection(dat_out, (x_cut_local, 0.0), 0)
                if inters.shape[0] >= 2:
                    idx = np.argsort(inters[:, 1])
                    p0 = inters[idx[0]] + point_ref
                    p1 = inters[idx[-1]] + point_ref
                    try:
                        doc.layers.new(name="SplitMark", dxfattribs={"color": 3})
                    except ezdxf.DXFError:
                        pass
                    msp.add_line(p0, p1, dxfattribs={"layer": "SplitMark"})

            # 仮保存（分割ユーティリティは読み込みベース）
            doc.saveas(file_name)

            if need_split and (p0 is not None):
                x_cut_world = 0.5 * (p0[0] + p1[0])

                # 出力ファイル名
                fn_front = os.path.join(output_dir, f"rib_{id}_前縁側.dxf")
                fn_rear = os.path.join(output_dir, f"rib_{id}_後縁側.dxf")

                # 分割実行（外形だけ三角キー / その他は直線クリップ）
                split_dxf_with_tri_key(
                    src_path=file_name,
                    dst_left=fn_front,
                    dst_right=fn_rear,
                    seam_x=x_cut_world,
                    layer_name="Layer",
                    tri_depth=TRI_DEPTH,
                    tri_apex_pos=TRI_APEX_POS,
                )

                # 分割線を元DXFから削除
                for e in list(msp.query('LINE[layer=="SplitMark"]')):
                    e.destroy()
                doc.saveas(file_name)

                # 左右のPNG/SVG
                for path, suffix in ((fn_front, "前縁側"), (fn_rear, "後縁側")):
                    d = ezdxf.readfile(path)
                    ms = d.modelspace()
                    fig = plt.figure()
                    ax = fig.add_axes([0, 0, 1, 1])
                    ctx = RenderContext(d)
                    out = MatplotlibBackend(ax)
                    Frontend(ctx, out).draw_layout(ms, finalize=True)
                    fig.savefig(os.path.join(output_dir, f"rib_{id}_{suffix}.png"))
                    if mode == "lasercut":
                        fig.savefig(
                            os.path.join(output_dir, f"rib_{id}_{suffix}.svg"),
                            format="svg",
                        )
                    plt.close(fig)

    if all_at_once:
        doc.saveas(file_name)

    print("ファイルが保存されました。")

    if preview:
        # プレビューは可能な環境のみ（失敗しても無視）
        try:
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1])
            ctx = RenderContext(doc)
            out = MatplotlibBackend(ax)
            Frontend(ctx, out).draw_layout(msp, finalize=True)
            plt.show()
        except Exception as e:
            print(f"プレビュー表示に失敗しました: {e}")


def nearest_next_index(dat, x):
    index = next((i for i, point in enumerate(dat) if point[0] > x), None)
    return index


def rotate_points(points, center, angle_degrees):
    angle_radians = np.radians(angle_degrees)
    rotation_matrix = np.array(
        [
            [np.cos(angle_radians), -np.sin(angle_radians)],
            [np.sin(angle_radians), np.cos(angle_radians)],
        ]
    )
    translated_points = points - center
    rotated_points = np.dot(translated_points, rotation_matrix)
    rotated_points += center
    return rotated_points


def find_line_intersection(curve, point, alpha):
    intersections = []

    # alpha==0 は垂直線
    if alpha == 0:
        x_vertical = point[0]
        for i in range(len(curve) - 1):
            x1, y1 = curve[i]
            x2, y2 = curve[i + 1]
            if (x1 <= x_vertical <= x2) or (x2 <= x_vertical <= x1):
                # 線形補間
                if x2 != x1:
                    y_intersection = y1 + (y2 - y1) * (x_vertical - x1) / (x2 - x1)
                else:
                    y_intersection = y1
                intersections.append((x_vertical, y_intersection))
    else:
        a = np.tan(np.radians(90 + alpha))
        b = point[1] - a * point[0]
        for i in range(len(curve) - 1):
            x1, y1 = curve[i]
            x2, y2 = curve[i + 1]
            denom = a * (x2 - x1) - (y2 - y1)
            if abs(denom) < 1e-12:
                continue
            x_intersection = (x2 * (y1 - b) - x1 * (y2 - b)) / denom
            y_intersection = a * x_intersection + b
            # 区間内判定
            if (
                min(x1, x2) - 1e-9 <= x_intersection <= max(x1, x2) + 1e-9
                and min(y1, y2) - 1e-9 <= y_intersection <= max(y1, y2) + 1e-9
            ):
                intersections.append((x_intersection, y_intersection))

    return np.array(intersections)


def find_circle_intersection(curve, center, radius):
    intersections = []
    for i in range(len(curve) - 1):
        x1, y1 = curve[i]
        x2, y2 = curve[i + 1]
        d1 = np.hypot(x1 - center[0], y1 - center[1])
        d2 = np.hypot(x2 - center[0], y2 - center[1])
        if (d1 < radius and d2 > radius) or (d1 > radius and d2 < radius):
            t = (radius - d1) / (d2 - d1 + 1e-12)
            x_intersection = (1 - t) * x1 + t * x2
            y_intersection = (1 - t) * y1 + t * y2
            intersections.append((x_intersection, y_intersection))
    return np.array(intersections)


def divide_dat(dat, start, end):
    """翼型下部のあるx位置からあるx位置までを分割して返す。範囲外は最近傍に丸める"""
    dat_lower = dat[np.argmin(dat[:, 0]) :].copy()
    start_index = np.argmin(np.abs(dat_lower[:, 0] - start))
    if dat_lower[start_index][0] < start:
        start_index += 1
    end_index = np.argmin(np.abs(dat_lower[:, 0] - end))
    if dat_lower[end_index][0] > end:
        end_index -= 1
    return dat_lower[start_index : end_index + 1]


def add_line_inside_foil(
    msp,
    dat,
    point,
    alpha,
    point_ref,
    inward_length=1,
    outward_length=1,
    peephole=False,
    peephole_max=3,  # ← 追加：窓の最大個数（1,2,3を想定） 2.5は３個の時のみぎはしをけした。
):
    intersections = find_line_intersection(dat, point, alpha)
    if len(intersections) == 0:
        return

    idx = (
        np.argsort(intersections[:, 1])
        if alpha == 0
        else np.argsort(intersections[:, 0])
    )
    a_local = intersections[idx[0]]
    b_local = intersections[idx[-1]]

    seg = b_local - a_local
    seg_len = np.linalg.norm(seg)
    if seg_len < 1e-9:
        return
    vec = seg / seg_len
    ortho = np.array([-vec[1], vec[0]])

    line_attrs = (
        {"layer": "Layer", "color": 7} if mode != "stilo_lazer" else {"layer": "Layer"}
    )
    msp.add_line(a_local + point_ref, b_local + point_ref, dxfattribs=line_attrs)

    if not peephole or peephole_max <= 0:
        return

    # 端の余白（はみ出し防止）
    margin_abs = peephole_length / 2.0 + protrude_length + 0.5
    m = min(0.45, margin_abs / seg_len)  # 相対マージン
    usable = 1 - 2 * m

    # 目標個数に応じた基準配置（相対座標）
    if peephole_max >= 3:
        base = np.array([0.2, 0.5, 0.8])
    elif peephole_max == 2:
        base = np.array([0.35, 0.65])
    elif peephole_max == 2.5:
        base = np.array([0.2, 0.45])
    else:
        base = np.array([0.5])

    # 長さが短いときは自動で中央寄せ＆個数削減
    if usable <= 0.0:
        ts = np.array([0.5])
    else:
        ts = m + usable * base
        if usable < 0.35:
            ts = np.array([m + usable * 0.5])  # 1個に削減

    for t in ts:
        p = a_local + t * seg
        p0 = p - (peephole_length / 2.0 + protrude_length) * vec
        p1 = p + (peephole_length / 2.0 + protrude_length) * vec
        msp.add_line(p0 + point_ref, p1 + point_ref, dxfattribs=line_attrs)

        hh = peephole_length / 2.0
        hw = peephole_length / 2.0
        c = p + point_ref
        left_side = c + hw * ortho
        top_left = c + hh * vec
        right_side = c - hw * ortho
        bottom_side = c - hh * vec
        square = np.array([left_side, top_left, right_side, bottom_side, left_side])
        msp.add_lwpolyline(square, dxfattribs={"layer": "Layer"})


def add_square(msp, square_center, vec, height, width):
    """vecはheight方向の単位ベクトル"""
    half_height = height / 2
    half_width = width / 2
    ortho = np.array([-vec[1], vec[0]])
    top_right = square_center + half_height * vec + half_width * ortho
    top_left = square_center + half_height * vec - half_width * ortho
    bottom_right = square_center - half_height * vec + half_width * ortho
    bottom_left = square_center - half_height * vec - half_width * ortho
    points = np.array([top_right, top_left, bottom_left, bottom_right, top_right])
    msp.add_lwpolyline(points, dxfattribs={"layer": "Layer"})


def add_tangedsquare(msp, geo, point_ref, gap, x, width, depth, spar_center, alpha_rib):
    x = x * geo.chord_ref
    nvec = geo.nvec(x)
    square_center = rotate_points(
        np.array([abs(x), geo.y(x)]) + nvec * (gap + depth / 2) - spar_center,
        (0, 0),
        alpha_rib,
    )
    vec = rotate_points(nvec, (0, 0), alpha_rib)
    add_square(msp, square_center + point_ref, vec, depth, width)


def add_TEarc(msp, geo, point_ref, radius, spar_center, alpha_rib):
    TE_center = np.array([geo.chord_ref, 0])
    intersections = find_circle_intersection(geo.dat_out, TE_center, radius)
    rotated_TE_center = rotate_points(TE_center - spar_center, (0, 0), alpha_rib)
    rotated_intersections = rotate_points(
        intersections - spar_center, (0, 0), alpha_rib
    )
    start = rotated_intersections[0]
    start_angle = np.arctan2(
        start[1] - rotated_TE_center[1], start[0] - rotated_TE_center[0]
    )
    end = rotated_intersections[1]
    end_angle = np.arctan2(end[1] - rotated_TE_center[1], end[0] - rotated_TE_center[0])

    if mode == "print":
        msp.add_arc(
            center=rotated_TE_center + point_ref,
            radius=radius,
            start_angle=np.rad2deg(start_angle),
            end_angle=np.rad2deg(end_angle),
            dxfattribs={"layer": "Layer"},
        )
    elif mode in ("lasercut", "stilo_lazer"):
        start_end = start + protrude_length * np.array(
            [-np.sin(start_angle), np.cos(start_angle)]
        )
        end_start = end - protrude_length * np.array(
            [-np.sin(end_angle), np.cos(end_angle)]
        )
        msp.add_line(
            start + point_ref, start_end + point_ref, dxfattribs={"layer": "Layer"}
        )
        msp.add_line(
            end_start + point_ref, end + point_ref, dxfattribs={"layer": "Layer"}
        )


def write_num_with_lines(
    msp, num, text_location, text_height, color=0, layer="NumLabel"
):
    """
    番号をポリラインで描く。layer と color は引数で指定可能。
    """

    def add(coords):
        msp.add_lwpolyline(
            text_height * coords + text_location,
            format="xy",
            close=False,
            dxfattribs={"layer": layer, "color": int(color)},
        )

    if num == "0":
        add(np.array([[0.4, 2], [0, 2], [0, 0], [0.4, 0]]) / 2)
        add(np.array([[0.6, 0], [1, 0], [1, 2], [0.6, 2]]) / 2)
    elif num == "1":
        add(np.array([[0.5, 2], [0.5, 0]]) / 2)
    elif num == "2":
        add(np.array([[0, 2], [1, 2], [1, 1], [0, 1], [0, 0], [1, 0]]) / 2)
    elif num == "3":
        add(np.array([[0, 2], [1, 2], [1, 1], [0, 1], [1, 1], [1, 0], [0, 0]]) / 2)
    elif num == "4":
        add(np.array([[0, 2], [0, 1], [1, 1], [1, 2], [1, 0]]) / 2)
    elif num == "5":
        add(np.array([[1, 2], [0, 2], [0, 1], [1, 1], [1, 0], [0, 0]]) / 2)
    elif num == "6":
        add(np.array([[1, 2], [0.6, 2]]) / 2)
        add(np.array([[0.4, 2], [0, 2], [0, 0], [0.4, 0]]) / 2)
        add(np.array([[0.6, 0], [1, 0], [1, 1], [0.6, 1]]) / 2)
        add(np.array([[0.4, 1], [0, 1]]) / 2)
    elif num == "7":
        add(np.array([[0, 1], [0, 2], [1, 2], [1, 0]]) / 2)
    elif num == "8":
        add(np.array([[0.4, 2], [0, 2], [0, 0], [0.4, 0]]) / 2)
        add(np.array([[0.6, 0], [1, 0], [1, 2], [0.6, 2]]) / 2)
        add(np.array([[0, 1], [0.4, 1]]) / 2)
        add(np.array([[0.6, 1], [1, 1]]) / 2)
    elif num == "9":
        add(np.array([[1, 1], [0.6, 1]]) / 2)
        add(np.array([[0.4, 1], [0, 1], [0, 2], [0.4, 2]]) / 2)
        add(np.array([[0.6, 2], [1, 2], [1, 0], [0.6, 0]]) / 2)
        add(np.array([[0.4, 0], [0, 0]]) / 2)
    elif num == "L":
        add(np.array([[0, 2], [0, 0], [1, 0]]) / 2)
    elif num == "T":
        add(np.array([[0, 2], [1, 2], [0.5, 2], [0.5, 0]]) / 2)
    else:
        raise ValueError("Invalid number")


# ====== 分割ユーティリティ ======
def _poly_area_signed(pts):
    P = pts[:-1] if (len(pts) >= 2 and np.allclose(pts[0], pts[-1])) else pts
    if len(P) < 3:
        return 0.0
    s = 0.0
    for i in range(len(P)):
        x1, y1 = P[i]
        x2, y2 = P[(i + 1) % len(P)]
        s += x1 * y2 - x2 * y1
    return 0.5 * s


def _lwpoly_to_pts(e):
    pts = np.array([(v[0], v[1]) for v in e.get_points("xy")], float)
    if pts.size and not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    return pts


def _outline_and_others(msp, layer_name="Layer"):
    outline = None
    others = []
    polys = list(msp.query(f'LWPOLYLINE[layer=="{layer_name}"]'))
    if not polys:
        polys = [e for e in msp.query("LWPOLYLINE") if e.dxf.layer != "NumLabel"]
    for e in polys:
        pts = _lwpoly_to_pts(e)
        if pts.shape[0] < 3:
            others.append(pts)
            continue
        if not np.allclose(pts[0], pts[-1]):
            others.append(pts)
            continue
        if (outline is None) or (
            abs(_poly_area_signed(pts)) > abs(_poly_area_signed(outline))
        ):
            if outline is not None:
                others.append(outline)
            outline = pts
        else:
            others.append(pts)
    return outline, others


def _xcut_intersections(poly, xcut):
    P = poly[:-1] if (len(poly) >= 2 and np.allclose(poly[0], poly[-1])) else poly
    ys = []
    for i in range(len(P)):
        x1, y1 = P[i]
        x2, y2 = P[(i + 1) % len(P)]
        dx = x2 - x1
        if abs(dx) < 1e-12:
            if abs(x1 - xcut) < 1e-9:
                ys += [y1, y2]
            continue
        t = (xcut - x1) / dx
        if (
            -1e-12 <= t <= 1 + 1e-12
            and min(x1, x2) - 1e-9 <= xcut <= max(x1, x2) + 1e-9
        ):
            ys.append(float(y1 + t * (y2 - y1)))
    if len(ys) < 2:
        return None, None
    ys.sort()
    return ys[0], ys[-1]


def _tri_key_x(y, y0, y1, xbase, depth, apex_pos):
    if y1 <= y0:
        return xbase
    ym = y0 + (y1 - y0) * apex_pos
    if y <= ym:
        t = 0.0 if ym == y0 else (y - y0) / (ym - y0)
        return xbase + depth * t
    else:
        t = 0.0 if y1 == ym else (y1 - y) / (y1 - ym)
        return xbase + depth * t


def _inside_tri_side(p, y0, y1, xbase, depth, apex_pos, side):
    xk = _tri_key_x(p[1], y0, y1, xbase, depth, apex_pos)
    return (p[0] <= xk + 1e-9) if side == "left" else (p[0] >= xk - 1e-9)


def _seg_intersect_tri(p0, p1, y0, y1, xbase, depth, apex_pos, side, it=28):
    def g(p):
        xk = _tri_key_x(p[1], y0, y1, xbase, depth, apex_pos)
        return (p[0] - xk) if side == "right" else (xk - p[0])

    g0, g1 = g(p0), g(p1)
    if g0 * g1 > 0:
        return None
    a, fa = p0.copy(), g0
    b, fb = p1.copy(), g1
    for _ in range(it):
        m = 0.5 * (a + b)
        fm = g(m)
        if abs(fm) < 1e-12:
            return m
        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b)


def _clip_closed_by_tri(poly, y0, y1, xbase, depth, apex_pos, side):
    P = poly.copy()
    if np.allclose(P[0], P[-1]):
        P = P[:-1]
    if len(P) < 3:
        return np.zeros((0, 2))
    out = []
    S = P[-1]
    Sin = _inside_tri_side(S, y0, y1, xbase, depth, apex_pos, side)
    for E in P:
        Ein = _inside_tri_side(E, y0, y1, xbase, depth, apex_pos, side)
        if Sin and Ein:
            out.append(E)
        elif Sin and not Ein:
            I = _seg_intersect_tri(S, E, y0, y1, xbase, depth, apex_pos, side)
            if I is not None:
                out.append(I)
        elif (not Sin) and Ein:
            I = _seg_intersect_tri(S, E, y0, y1, xbase, depth, apex_pos, side)
            if I is not None:
                out.append(I)
            out.append(E)
        S, Sin = E, Ein
    if len(out) < 3:
        return np.zeros((0, 2))
    out = np.array(out, float)
    if not np.allclose(out[0], out[-1]):
        out = np.vstack([out, out[0]])
    return out


def _inside_x_half(p, xcut, side):
    return (p[0] <= xcut + 1e-9) if side == "left" else (p[0] >= xcut - 1e-9)


def _seg_intersect_x(p0, p1, xcut):
    x1, y1 = p0
    x2, y2 = p1
    dx = x2 - x1
    if abs(dx) < 1e-12:
        return None
    t = (xcut - x1) / dx
    if t < -1e-12 or t > 1 + 1e-12:
        return None
    return np.array([xcut, y1 + t * (y2 - y1)], float)


def _clip_closed_by_vertical(poly, xcut, side):
    P = poly.copy()
    if np.allclose(P[0], P[-1]):
        P = P[:-1]
    if len(P) < 3:
        return np.zeros((0, 2))
    out = []
    S = P[-1]
    Sin = _inside_x_half(S, xcut, side)
    for E in P:
        Ein = _inside_x_half(E, xcut, side)
        if Sin and Ein:
            out.append(E)
        elif Sin and not Ein:
            I = _seg_intersect_x(S, E, xcut)
            if I is not None:
                out.append(I)
        elif (not Sin) and Ein:
            I = _seg_intersect_x(S, E, xcut)
            if I is not None:
                out.append(I)
            out.append(E)
        S, Sin = E, Ein
    if len(out) < 3:
        return np.zeros((0, 2))
    out = np.array(out, float)
    if not np.allclose(out[0], out[-1]):
        out = np.vstack([out, out[0]])
    return out


def _poly_all_on_side(poly, xcut, side):
    P = poly[:-1] if (len(poly) >= 2 and np.allclose(poly[0], poly[-1])) else poly
    if len(P) == 0:
        return False
    if side == "left":
        return bool(np.all(P[:, 0] <= xcut + 1e-9))
    else:
        return bool(np.all(P[:, 0] >= xcut - 1e-9))


def _clip_line_by_vertical(p0, p1, xcut, side):
    x0, y0 = p0
    x1, y1 = p1
    onL0 = x0 <= xcut + 1e-9
    onL1 = x1 <= xcut + 1e-9
    onR0 = x0 >= xcut - 1e-9
    onR1 = x1 >= xcut - 1e-9

    def at():
        if abs(x1 - x0) < 1e-12:
            return None
        t = (xcut - x0) / (x1 - x0)
        y = y0 + t * (y1 - y0)
        return np.array([xcut, y], float)

    res = []
    if side == "left":
        if onL0 and onL1:
            res = [(p0, p1)]
        elif onL0 and (not onL1):
            ip = at()
            res = [(p0, ip)] if ip is not None else []
        elif (not onL0) and onL1:
            ip = at()
            res = [(ip, p1)] if ip is not None else []
    else:
        if onR0 and onR1:
            res = [(p0, p1)]
        elif onR0 and (not onR1):
            ip = at()
            res = [(p0, ip)] if ip is not None else []
        elif (not onR0) and onR1:
            ip = at()
            res = [(ip, p1)] if ip is not None else []
    return res


def split_dxf_with_tri_key(
    src_path,
    dst_left,
    dst_right,
    seam_x,
    layer_name="Layer",
    tri_depth=6.0,
    tri_apex_pos=0.5,
):
    """
    - 最大面積の閉ポリライン(=外形)は蟻継ぎで左右分割（失敗時は垂直クリップ）。
    - その他の LWPOLYLINE は垂直半平面クリップ、LINE は直線クリップ。
    - TEXT/MTEXT/INSERT/MINSERT/DIMENSION/LEADER は x 位置で左右へ振り分け。
    - ★NumLabel レイヤは『文字全体を1グループ』として bbox 中心で対称移動コピー。
      → 23 が 32 になる問題を回避（鏡映ではなく平行移動のみ）。
    - 垂直の分割ガイド線は出力しない（非表示）。
    """
    import itertools
    import numpy as np
    import ezdxf
    from ezdxf.addons.importer import Importer

    NUM_LAYER = "NumLabel"
    eps = 1e-9

    # ---------- 入力 ----------
    src = ezdxf.readfile(src_path)
    msp = src.modelspace()

    # ---------- 出力 ----------
    docL = ezdxf.new("R2007", setup=True)
    mspL = docL.modelspace()
    docR = ezdxf.new("R2007", setup=True)
    mspR = docR.modelspace()
    # 追加：元レイヤ色を取得
    src_layer_color = 256
    try:
        src_layer_color = src.layers.get(layer_name).dxf.color
    except Exception:
        pass

    num_layer_color = 256
    try:
        num_layer_color = src.layers.get(NUM_LAYER).dxf.color
    except Exception:
        pass

    for d in (docL, docR):
        for lname, color, lw in (
            (layer_name, src_layer_color, 50),
            (NUM_LAYER, num_layer_color, 15),
        ):
            try:
                d.layers.new(name=lname, dxfattribs={"color": color, "lineweight": lw})
            except ezdxf.DXFError:
                pass

    impL = Importer(src, docL)
    impR = Importer(src, docR)

    # ---------- ユーティリティ ----------
    def _lwpoly_to_pts(e):
        pts = np.array([(v[0], v[1]) for v in e.get_points("xy")], float)
        if pts.size and not np.allclose(pts[0], pts[-1]):
            pts = np.vstack([pts, pts[0]])
        return pts

    def _poly_area_signed(pts):
        P = pts[:-1] if (len(pts) >= 2 and np.allclose(pts[0], pts[-1])) else pts
        if len(P) < 3:
            return 0.0
        s = 0.0
        for i in range(len(P)):
            x1, y1 = P[i]
            x2, y2 = P[(i + 1) % len(P)]
            s += x1 * y2 - x2 * y1
        return 0.5 * s

    # 外形とその他
    polys = [e for e in msp.query(f'LWPOLYLINE[layer=="{layer_name}"]')]
    if not polys:
        polys = [
            e
            for e in msp.query("LWPOLYLINE")
            if e.dxf.layer not in (NUM_LAYER, "SplitMark")
        ]
    outline = None
    others = []  # -> (entity, pts) を入れる
    for e in polys:
        pts = _lwpoly_to_pts(e)
        if pts.shape[0] < 3:
            continue
        if (outline is None) or (
            abs(_poly_area_signed(pts))
            > abs(_poly_area_signed(outline[1] if outline else pts))
        ):
            if outline is not None:
                others.append(outline)
            outline = (e, pts)
        else:
            others.append((e, pts))

    # seam の縦範囲を決定
    def _xcut_intersections(poly, xcut):
        P = poly[:-1] if (len(poly) >= 2 and np.allclose(poly[0], poly[-1])) else poly
        ys = []
        for i in range(len(P)):
            x1, y1 = P[i]
            x2, y2 = P[(i + 1) % len(P)]
            dx = x2 - x1
            if abs(dx) < 1e-12:
                if abs(x1 - xcut) < 1e-9:
                    ys += [y1, y2]
                continue
            t = (xcut - x1) / dx
            if (
                -1e-12 <= t <= 1 + 1e-12
                and min(x1, x2) - 1e-9 <= xcut <= max(x1, x2) + 1e-9
            ):
                ys.append(float(y1 + t * (y2 - y1)))
        if len(ys) < 2:
            return None, None
        ys.sort()
        return ys[0], ys[-1]

    if outline is not None:
        e_out, pts_out = outline
        y0, y1 = _xcut_intersections(pts_out, seam_x)
        if y0 is None:
            y0 = float(np.min(pts_out[:, 1]))
            y1 = float(np.max(pts_out[:, 1]))
    else:
        ymin, ymax = +1e18, -1e18
        for e in msp.query("LWPOLYLINE"):
            if e.dxf.layer in (NUM_LAYER, "SplitMark"):
                continue
            pts = _lwpoly_to_pts(e)
            if pts.size:
                ymin = min(ymin, float(np.min(pts[:, 1])))
                ymax = max(ymax, float(np.max(pts[:, 1])))
        y0, y1 = ymin, ymax

    # 蟻継ぎキー
    def _tri_x(y, y0, y1, xbase, depth, apex):
        if y1 <= y0:
            return xbase
        ym = y0 + (y1 - y0) * apex
        if y <= ym:
            t = 0.0 if ym == y0 else (y - y0) / (ym - y0)
            return xbase + depth * t
        else:
            t = 0.0 if y1 == ym else (y1 - y) / (y1 - ym)
            return xbase + depth * t

    def _inside_tri(p, side):
        xk = _tri_x(p[1], y0, y1, seam_x, tri_depth, tri_apex_pos)
        return (p[0] <= xk + 1e-9) if side == "left" else (p[0] >= xk - 1e-9)

    def _seg_inter_tri(p0, p1, side):
        def g(p):
            xk = _tri_x(p[1], y0, y1, seam_x, tri_depth, tri_apex_pos)
            return (p[0] - xk) if side == "right" else (xk - p[0])

        a, b = p0.copy(), p1.copy()
        fa, fb = g(a), g(b)
        if fa * fb > 0:
            return None
        for _ in range(28):
            m = 0.5 * (a + b)
            fm = g(m)
            if abs(fm) < 1e-12:
                return m
            if fa * fm <= 0:
                b, fb = m, fm
            else:
                a, fa = m, fm
        return 0.5 * (a + b)

    def _clip_closed_by_tri(poly, side):
        P = poly.copy()
        if np.allclose(P[0], P[-1]):
            P = P[:-1]
        if len(P) < 3:
            return np.zeros((0, 2))
        out = []
        S = P[-1]
        Sin = _inside_tri(S, side)
        for E in P:
            Ein = _inside_tri(E, side)
            if Sin and Ein:
                out.append(E)
            elif Sin and not Ein:
                I = _seg_inter_tri(S, E, side)
                out.append(I) if I is not None else None
            elif (not Sin) and Ein:
                I = _seg_inter_tri(S, E, side)
                out.append(I) if I is not None else None
                out.append(E)
            S, Sin = E, Ein
        if len(out) < 3:
            return np.zeros((0, 2))
        out = np.array(out, float)
        if not np.allclose(out[0], out[-1]):
            out = np.vstack([out, out[0]])
        return out

    def _clip_closed_by_vertical(poly, xcut, side):
        P = poly.copy()
        if np.allclose(P[0], P[-1]):
            P = P[:-1]
        if len(P) < 3:
            return np.zeros((0, 2))

        def inside(p):
            return (p[0] <= xcut + 1e-9) if side == "left" else (p[0] >= xcut - 1e-9)

        def inter(p0, p1):
            x1, y1 = p0
            x2, y2 = p1
            dx = x2 - x1
            if abs(dx) < 1e-12:
                return None
            t = (xcut - x1) / dx
            if t < -1e-12 or t > 1 + 1e-12:
                return None
            return np.array([xcut, y1 + t * (y2 - y1)], float)

        out = []
        S = P[-1]
        Sin = inside(S)
        for E in P:
            Ein = inside(E)
            if Sin and Ein:
                out.append(E)
            elif Sin and not Ein:
                I = inter(S, E)
                out.append(I) if I is not None else None
            elif (not Sin) and Ein:
                I = inter(S, E)
                out.append(I) if I is not None else None
                out.append(E)
            S, Sin = E, Ein
        if len(out) < 3:
            return np.zeros((0, 2))
        out = np.array(out, float)
        if not np.allclose(out[0], out[-1]):
            out = np.vstack([out, out[0]])
        return out

    def _poly_all_on_side(pts, xcut, side):
        P = pts[:-1] if (len(pts) >= 2 and np.allclose(pts[0], pts[-1])) else pts
        if len(P) == 0:
            return False
        return (
            bool(np.all(P[:, 0] <= xcut + 1e-9))
            if side == "left"
            else bool(np.all(P[:, 0] >= xcut - 1e-9))
        )

    def _col(ent):
        c = getattr(ent.dxf, "color", 256)
        return c if (c not in (0, None)) else 256

    # ---------- 1) 外形（蟻継ぎ） ----------
    if outline is not None:
        e_out, pts_out = outline
        L = _clip_closed_by_tri(pts_out, "left")
        R = _clip_closed_by_tri(pts_out, "right")
        if L.shape[0] < 3:
            L = _clip_closed_by_vertical(pts_out, seam_x, "left")
        if R.shape[0] < 3:
            R = _clip_closed_by_vertical(pts_out, seam_x, "right")
        if L.shape[0] >= 3:
            mspL.add_lwpolyline(
                L,
                format="xy",
                close=True,
                dxfattribs={"layer": layer_name, "color": _col(e_out)},
            )
        if R.shape[0] >= 3:
            mspR.add_lwpolyline(
                R,
                format="xy",
                close=True,
                dxfattribs={"layer": layer_name, "color": _col(e_out)},
            )

    # ---------- 2) NumLabel を“グループ対称移動コピー” ----------
    num_entities = list(msp.query(f'LWPOLYLINE[layer=="{NUM_LAYER}"]'))
    if num_entities:
        # グループ（全文字）の BBox を取る
        xs = []
        for e in num_entities:
            pts = np.array([(v[0], v[1]) for v in e.get_points("xy")], float)
            if pts.size:
                xs += [float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))]
        cx_group = 0.5 * (min(xs) + max(xs)) if xs else seam_x

        # どちら側に“元”があるか判定
        on_left = cx_group <= seam_x + eps
        on_right = cx_group >= seam_x - eps

        # 反対側へは “鏡映でなく”、Δx=2*(seam_x - cx_group) の**平行移動**
        dx = 2.0 * (seam_x - cx_group)

        def _translate_lwpoly(e, dx):
            c = e.copy()
            new_pts = []
            for v in c.get_points("xy"):
                new_pts.append((v[0] + dx, v[1]))
            c.set_points(new_pts, format="xy")
            return c

        # 元の配置は元の側へ、その**対称位置の複製**を反対側へ
        if on_left:
            impL.import_entities([e.copy() for e in num_entities], mspL)
            impR.import_entities([_translate_lwpoly(e, dx) for e in num_entities], mspR)
        if on_right:
            impR.import_entities([e.copy() for e in num_entities], mspR)
            impL.import_entities([_translate_lwpoly(e, dx) for e in num_entities], mspL)

    # ---------- 3) その他 LWPOLYLINE ----------
    for e_src, pts in others:
        if pts.shape[0] < 2:
            continue
        c = _col(e_src)

        if _poly_all_on_side(pts, seam_x, "left") and not _poly_all_on_side(
            pts, seam_x, "right"
        ):
            mspL.add_lwpolyline(
                pts,
                format="xy",
                close=np.allclose(pts[0], pts[-1]),
                dxfattribs={"layer": layer_name, "color": c},
            )
            continue
        if _poly_all_on_side(pts, seam_x, "right") and not _poly_all_on_side(
            pts, seam_x, "left"
        ):
            mspR.add_lwpolyline(
                pts,
                format="xy",
                close=np.allclose(pts[0], pts[-1]),
                dxfattribs={"layer": layer_name, "color": c},
            )
            continue
        lp = _clip_closed_by_vertical(pts, seam_x, "left")
        rp = _clip_closed_by_vertical(pts, seam_x, "right")
        if lp.shape[0] >= 3:
            mspL.add_lwpolyline(
                lp,
                format="xy",
                close=True,
                dxfattribs={"layer": layer_name, "color": c},
            )
        if rp.shape[0] >= 3:
            mspR.add_lwpolyline(
                rp,
                format="xy",
                close=True,
                dxfattribs={"layer": layer_name, "color": c},
            )

    # ---------- 4) LINE ----------
    def _clip_line_by_vertical(p0, p1, xcut, side):
        x0, y0 = p0
        x1, y1 = p1
        onL0 = x0 <= xcut + 1e-9
        onL1 = x1 <= xcut + 1e-9
        onR0 = x0 >= xcut - 1e-9
        onR1 = x1 >= xcut - 1e-9

        def inter():
            if abs(x1 - x0) < 1e-12:
                return None
            t = (xcut - x0) / (x1 - x0)
            y = y0 + t * (y1 - y0)
            return np.array([xcut, y], float)

        if side == "left":
            if onL0 and onL1:
                return [(p0, p1)]
            if onL0 and not onL1:
                ip = inter()
                return [(p0, ip)] if ip is not None else []
            if (not onL0) and onL1:
                ip = inter()
                return [(ip, p1)] if ip is not None else []
        else:
            if onR0 and onR1:
                return [(p0, p1)]
            if onR0 and not onR1:
                ip = inter()
                return [(p0, ip)] if ip is not None else []
            if (not onR0) and onR1:
                ip = inter()
                return [(ip, p1)] if ip is not None else []
        return []

    for e in msp.query('LINE[layer!="SplitMark"]'):
        s, t = e.dxf.start, e.dxf.end
        p0 = np.array([float(s.x), float(s.y)], float)
        p1 = np.array([float(t.x), float(t.y)], float)
        col = e.dxf.color if (e.dxf.color not in (0, None)) else 256
        for a, b in _clip_line_by_vertical(p0, p1, seam_x, "left"):
            mspL.add_line(
                tuple(a), tuple(b), dxfattribs={"layer": layer_name, "color": col}
            )
        for a, b in _clip_line_by_vertical(p0, p1, seam_x, "right"):
            mspR.add_line(
                tuple(a), tuple(b), dxfattribs={"layer": layer_name, "color": col}
            )

    # ---------- 5) TEXT/MTEXT/INSERT/MINSERT/DIMENSION/LEADER ----------
    def _clone_on_side(ent, x):
        if x <= seam_x + 1e-9:
            impL.import_entities([ent.copy()], mspL)
        if x >= seam_x - 1e-9:
            impR.import_entities([ent.copy()], mspR)

    for e in itertools.chain(
        msp.query('TEXT[layer!="SplitMark"]'), msp.query('MTEXT[layer!="SplitMark"]')
    ):
        ins = getattr(e.dxf, "insert", None)
        if ins is None:
            continue
        _clone_on_side(e, float(ins.x))

    for e in itertools.chain(
        msp.query('INSERT[layer!="SplitMark"]'),
        msp.query('MINSERT[layer!="SplitMark"]'),
    ):
        ins = getattr(e.dxf, "insert", None)
        if ins is None:
            continue
        _clone_on_side(e, float(ins.x))

    for e in msp.query('DIMENSION[layer!="SplitMark"]'):
        ins = getattr(e.dxf, "insert", None)
        if ins is None:
            impL.import_entities([e.copy()], mspL)
            impR.import_entities([e.copy()], mspR)
        else:
            _clone_on_side(e, float(ins.x))

    for e in msp.query('LEADER[layer!="SplitMark"]'):
        try:
            verts = list(e.vertices())
            if not verts:
                # 頂点無し→両方に複製（またはスキップでも可）
                impL.import_entities([e.copy()], mspL)
                impR.import_entities([e.copy()], mspR)
                continue
            _clone_on_side(e, float(verts[0].x))
        except Exception:
            pass

    # finalize & 保存
    impL.finalize()
    impR.finalize()
    docL.saveas(dst_left)
    docR.saveas(dst_right)


if __name__ == "__main__":
    wb_path = FindWorkbookUpOne("*.xlsm")
    wb = xw.Book(str(wb_path))  # ← インスタンス
    wb.set_mock_caller()  # ← この wb を caller にセット
    main()
