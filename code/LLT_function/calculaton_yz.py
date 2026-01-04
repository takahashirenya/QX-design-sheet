import numpy as np


def calculation_yz(
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
):
    n = Wing["span_div"]
    rad = np.pi / 180.0

    # 中央リブの初期値（VBA相当なら 0 でも可。必要なら外で与える）
    y[n] = y_center
    z[n] = z_center

    # dy: スカラー or 2n配列のどちらでも動くように
    dy_all = Wing["dy"]
    is_scalar_dy = np.isscalar(dy_all) or (np.ndim(dy_all) == 0)

    for i in range(n):
        # 左翼: index (n-1-i)
        li = n - 1 - i
        # 右翼: index (n+i)
        ri = n + i

        # αs, Γd の更新に相当（角度は[deg]のまま保持）
        setting_angle[li] = setting_angle0[li] + 0.5 * (phi[li + 1] + phi[li])  # 左
        dihedral_angle[li] = dihedral_angle0[li] - 0.5 * (
            theta[li + 1] + theta[li]
        )  # 左は負

        setting_angle[ri] = setting_angle0[ri] + 0.5 * (phi[ri] + phi[ri + 1])  # 右
        dihedral_angle[ri] = dihedral_angle0[ri] + 0.5 * (
            theta[ri] + theta[ri + 1]
        )  # 右

        # パネル幅 dy（VBAはスカラー .dy を使っている）
        if is_scalar_dy:
            dyL = float(dy_all)
            dyR = float(dy_all)
        else:
            # 左へ進む区間は li のパネル幅、右へ進む区間は ri のパネル幅と対応付け
            # （あなたのデータ構造により li/ri の対応が異なるなら調整してください）
            dyL = float(dy_all[li])
            dyR = float(dy_all[ri])

        # 左に一つ進む
        y[li] = y[li + 1] - dyL * np.cos(-rad * dihedral_angle[li])
        z[li] = z[li + 1] + dyL * np.sin(-rad * dihedral_angle[li])

        # 右に一つ進む
        y[ri + 1] = y[ri] + dyR * np.cos(rad * dihedral_angle[ri])
        z[ri + 1] = z[ri] + dyR * np.sin(rad * dihedral_angle[ri])

    # VBAのSubに合わせて戻り値は無し（破壊的更新）
    return None
