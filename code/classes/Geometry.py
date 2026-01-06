"""
幾何クラス

- GeometricalAirfoil


"""

import xlwings as xw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate


class GeometricalAirfoil:
    """
    翼型の幾何特性を表すクラス

    Parameters
    ----------
    dat : numpy.ndarray
        翼型の座標データ
    chord_ref : float
        規格コード長

    Attributes
    ----------
    dat : numpy.ndarray
        翼型の座標データ（正規化されてなくてもよい）
    chord_ref : float
        規格コード長
    dat_norm : numpy.ndarray
        翼型の座標データを0~1に正規化したもの
    dat_ref : numpy.ndarray
        翼型の座標データを規格コード長に合わせて拡大したもの
    dat_extended : numpy.ndarray
        翼型の上半分を左右反転させたdatデータ
    interp : scipy.interpolate.interpolate.interp1d
        dat_extendedをスプライン曲線で補間した関数。
        翼型の任意xでのy座標を、定義域を-chord_ref~chord_refとして.interp([x])で取得できる。
    """

    def __init__(self, dat, chord_ref=1):
        self.dat = dat.copy()
        self.chord_ref = chord_ref
        # datを0~1に正規化
        xmin = np.amin(self.dat[:, 0])
        xmax = np.amax(self.dat[:, 0])
        self.chord_act = xmax - xmin
        self.dat_norm = (self.dat - xmin) / self.chord_act
        # 規格コード長に合わせて拡大
        self.dat_ref = self.dat_norm * self.chord_ref
        # y座標が初めて負になるインデックスを取得
        first_negative_y_index = np.where(self.dat_norm[:, 1] < 0)[0][0]
        # 上側の点データを取得
        upper_side_data = self.dat_ref[:first_negative_y_index].copy()
        # x座標を左右反転
        upper_side_data[:, 0] = -upper_side_data[:, 0]
        # 結合
        _x = np.concatenate(
            [upper_side_data[:, 0], self.dat_ref[first_negative_y_index:][:, 0]]
        )
        _y = np.concatenate(
            [upper_side_data[:, 1], self.dat_ref[first_negative_y_index:][:, 1]]
        )
        self.dat_extended = np.array([_x, _y]).T
        # モデル作成
        self.interp = interpolate.interp1d(
            self.dat_extended[:, 0],
            self.dat_extended[:, 1],
            kind="linear",
            fill_value="extrapolate",
        )

    def y(self, x):
        # 任意xにおける翼型のy座標を返す
        return self.interp([x])[0]

    def thickness(self, x):
        # 任意xにおける翼厚を返す
        return self.y(-x) - self.y(x)

    def camber(self, x):
        # 任意xにおけるキャンバーを返す
        return (self.y(-x) + self.y(x)) / 2

    def perimeter(self, start, end):
        # startからendまでの周囲長を返す
        start_index = np.argmin(np.abs(self.normalized_dat[:, 0] - start))
        end_index = np.argmin(np.abs(self.normalized_dat[:, 0] - end))
        perimeter = 0
        for i in range(start_index, end_index):
            perimeter += np.sqrt(
                (self.dat[i + 1][1] - self.dat[i][1]) ** 2
                + (self.dat[i + 1][0] - self.dat[i][0]) ** 2
            )
        return perimeter

    def nvec(self, x):
        # 任意xにおける翼型の法線ベクトルを返す
        delta = 0.000001
        x_elem = self.interp(x) - self.interp(x + delta)
        y_elem = np.sign(x) * delta
        size = np.sqrt(x_elem**2 + y_elem**2)
        return np.array([x_elem, y_elem] / size).T

    def _polyline_outward_normals(dat: np.ndarray) -> np.ndarray:
        """dat: (N,2) with columns [x, y]. Return unit normals that point 'outward' (away from centroid)."""
        # 接線（両側差分の平均で安定化）
        v = np.diff(dat, axis=0)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        # 端点は隣の接線で補完
        t = np.vstack([v[0], (v[:-1] + v[1:]) * 0.5, v[-1]])
        t /= np.linalg.norm(t, axis=1, keepdims=True)

        # +90度回転で法線（x,y)->(-y, x）
        n = np.column_stack([-t[:, 1], t[:, 0]])

        # 外向きに揃える：重心から各点へのベクトルと法線の内積で符号判定
        c = dat.mean(axis=0)
        s = np.einsum("ij,ij->i", n, dat - c)  # 正なら外向き
        n[s < 0] *= -1.0

        return n

    def offset_foil(
        self,
        offset_base,
        offset_arr=[],
        carbon_length=None,  # カーボンチップ長さ [mm]（コード方向）
        carbon_depth=0.0,  # カーボン部の「追加」オフセット量
    ):
        import numpy as np

        # --- 基本データ ---
        dat = self.dat_extended.copy()  # [-chord..+chord] 上下面つなげたやつ
        depth_arr = np.ones(len(dat)) * float(offset_base)

        # =========================================================
        # ① 既存：プランクなど（下面側だけ） ← 元のロジックを維持
        # =========================================================
        if len(offset_arr) != 0:
            for i in range(len(offset_arr)):
                start_frac, end_frac, thickness = offset_arr[i]

                x_start = start_frac * self.chord_ref
                x_end = end_frac * self.chord_ref

                start = np.array([x_start, self.y(x_start)])
                end = np.array([x_end, self.y(x_end)])

                idx_start = np.searchsorted(dat[:, 0], start[0])
                idx_end = np.searchsorted(dat[:, 0], end[0])

                # dat に「2点ずつ」挿入して角を作る
                dat = np.insert(dat, [idx_start, idx_start], [start, start], axis=0)
                dat = np.insert(dat, [idx_end + 2, idx_end + 2], [end, end], axis=0)

                # depth_arr も同じ場所に 0 を挿入してから上書き
                depth_arr = np.insert(
                    depth_arr, [idx_start, idx_start, idx_end, idx_end], 0.0
                )
                depth_arr[idx_start] = depth_arr[idx_start - 1]
                depth_arr[idx_start + 1 : idx_end + 3] = thickness
                depth_arr[idx_end + 3] = depth_arr[idx_end + 4]

        # =========================================================
        # ② カーボンチップ：上面（x<0）・下面（x>0）両方に「角付き」で入れる
        # =========================================================
        if (
            carbon_length is not None
            and carbon_length > 0.0
            and abs(carbon_depth) > 1e-9
        ):
            chord = float(self.chord_ref)

            # TE x=chord から前縁側へ carbon_length 戻った位置まで
            x_end = chord
            x_start = max(0.0, chord - float(carbon_length))

            # ---------- まず下面側（x>0） ----------
            lower_start = np.array([x_start, self.y(x_start)])
            lower_end = np.array([x_end, self.y(x_end)])

            idx_start = np.searchsorted(dat[:, 0], lower_start[0])  # x>0 側
            idx_end = np.searchsorted(dat[:, 0], lower_end[0])

            dat = np.insert(
                dat, [idx_start, idx_start], [lower_start, lower_start], axis=0
            )
            dat = np.insert(
                dat, [idx_end + 2, idx_end + 2], [lower_end, lower_end], axis=0
            )

            depth_arr = np.insert(
                depth_arr, [idx_start, idx_start, idx_end, idx_end], 0.0
            )

            # 既存値→カーボン厚→元に戻す、という階段形
            depth_arr[idx_start] = depth_arr[idx_start - 1]
            depth_arr[idx_start + 1 : idx_end + 3] = depth_arr[idx_start] + float(
                carbon_depth
            )
            depth_arr[idx_end + 3] = depth_arr[idx_end + 4]

            # ---------- 次に上面側（x<0） ----------
            # 上面は dat_extended では x が「負側」に並んでいるので、
            # chordwise の同じ位置は -x_end ～ -x_start に対応
            x_start = max(0.0, chord - float(carbon_length - 1.55))
            upper_start = np.array([-x_end, self.y(-x_end)])
            upper_end = np.array([-x_start, self.y(-x_start)])

            # x の昇順なので、-x_end < -x_start となる：こちらが start / end
            idx_start_u = np.searchsorted(
                dat[:, 0], upper_start[0]
            )  # もっと負側（後縁寄り）
            idx_end_u = np.searchsorted(
                dat[:, 0], upper_end[0]
            )  # 0 に近い側（前縁寄り）

            dat = np.insert(
                dat, [idx_start_u, idx_start_u], [upper_start, upper_start], axis=0
            )
            dat = np.insert(
                dat, [idx_end_u + 2, idx_end_u + 2], [upper_end, upper_end], axis=0
            )

            depth_arr = np.insert(
                depth_arr, [idx_start_u, idx_start_u, idx_end_u, idx_end_u], 0.0
            )

            depth_arr[idx_start_u] = depth_arr[idx_start_u - 1]
            depth_arr[idx_start_u + 1 : idx_end_u + 3] = depth_arr[idx_start_u] + float(
                carbon_depth
            )
            depth_arr[idx_end_u + 3] = depth_arr[idx_end_u + 4]

        # =========================================================
        # ③ 法線方向へオフセット（元のロジック）
        # =========================================================
        n = self.nvec(dat[:, 0])  # (N,2) 想定

        # 正規化
        n_norm = np.linalg.norm(n, axis=1, keepdims=True)
        n_norm[n_norm == 0.0] = 1.0
        n = n / n_norm

        # 法線の向きが急にひっくり返っている点だけ反転してならす
        flip_cos_threshold = 0.0  # 90deg 以上ズレていたら反転

        N = len(n)
        if N >= 3:
            for i in range(1, N - 1):
                cos_prev = float(np.dot(n[i], n[i - 1]))
                cos_next = float(np.dot(n[i], n[i + 1]))
                if (cos_prev < flip_cos_threshold) and (cos_next < flip_cos_threshold):
                    n[i] *= -1.0
        elif N == 2:
            if float(np.dot(n[1], n[0])) < flip_cos_threshold:
                n[1] *= -1.0

        move = n * depth_arr[:, np.newaxis]

        # x は左右対称に畳む元々の仕様
        dat[:, 0] = np.abs(dat[:, 0])

        self.dat_out = dat + move
        return self.dat_out

    @property
    def tmax(self):
        # 最大翼厚
        self.tmax_at, self.thickness_max = max(
            [(x, self.thickness(x)) for x in np.arange(0, 1, 0.001)], key=lambda x: x[1]
        )
        return self

    @property
    def tmax_at(self):
        # 最大翼厚位置
        return self

    @property
    def cmax(self):
        # 最大キャンバー
        self.cmax_at, self.camber_max = max(
            [(x, self.camber(x)) for x in np.arange(0, 1, 0.001)], key=lambda x: x[1]
        )
        return self

    @property
    def cmax_at(self):
        # 最大キャンバー位置
        return self

    @property
    def curvature(
        self,
        dx_dt,
        dy_dt,
        d2x_dt2,
        d2y_dt2,
    ):
        # 任意xにおける翼型の曲率を返す
        return np.abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / np.power(
            dx_dt**2 + dy_dt**2, 3 / 2
        )

    @property
    def area(self):
        # 面積
        area = 0
        for i in range(int(len(self.dat) / 2) - 1):
            area += (
                (
                    (self.dat[i + 1][1] - self.dat[-(i + 1)][1])
                    + (self.dat[i][1] - self.dat[-i][1])
                )
                * (self.dat[i][0] - self.dat[i + 1][0])
                / 2
            )
        return area

    @property
    def leading_edge_radius(self):
        # 前縁半径（曲率の逆数）
        leading_edge_radius = 1 / self.curvature
        return 1  # leading_edge_radius / self.max_thickness()

    @property
    def trailing_edge_angle(self):
        # 後縁角
        return 0.1

    def outline(self):
        # 翼型の輪郭をfigで返す
        dpi = 72  # 画像の解像度
        figsize = (10, 2)  # 画像のサイズ
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, aspect="equal")
        ax.plot([r[0] for r in self.dat], [r[1] for r in self.dat], label="original")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        return fig
