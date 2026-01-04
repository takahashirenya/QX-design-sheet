import numpy as np


def calculation_ypzp(Wing, state, yp, zp, ymp, zmp, cp, dihedral_angle):
    n2 = 2 * int(Wing["span_div"])
    rad = np.pi / 180.0

    # 参照（読み取り専用）
    ycp = cp[1, :]  # (n2,)
    zcp = cp[2, :]  # (n2,)

    if (
        ycp.shape[0] != n2
        or zcp.shape[0] != n2
        or dihedral_angle.shape[0] != n2
        or yp.shape != (n2, n2)
        or zp.shape != (n2, n2)
        or ymp.shape != (n2, n2)
        or zmp.shape != (n2, n2)
    ):
        raise ValueError("形がおかしいです。配列の大きさを確認してください。")

    for j in range(n2):
        a = dihedral_angle[j] * rad
        c = np.cos(a)
        s = np.sin(a)

        # Δy, Δz（元の座標差）
        dy = ycp - ycp[j]  # (n2,)
        dz = zcp - zcp[j]  # (n2,)

        # オリジナルを j 列へ（回転）
        yp[:, j] = dy * c + dz * s
        zp[:, j] = -dy * s + dz * c

        # 鏡像：相手点は (y, z') with z' = -z_j - 2*hE
        dzm = zcp - (-zcp[j] - 2.0 * float(state["hE"]))
        # 角度は -a：cos(-a)=cos(a), sin(-a)=-sin(a)
        ymp[:, j] = dy * c - dzm * s
        zmp[:, j] = dy * s + dzm * c
