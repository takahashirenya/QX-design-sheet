import numpy as np

_RAD = np.pi / 180.0
_DEG = 180.0 / np.pi


def calculation_alpha_effective(
    Wing,
    state,
    alpha_effective,  # (2n,) 出力: 有効迎角[deg]
    alpha_induced,  # (2n,) 入力: 誘導迎角[deg]
    cp,  # (3,2n) 入力: コントロールポイント座標 (y は cp[1,:])
    dihedral_angle,  # (2n,) 入力: 上反角[deg]
    setting_angle,  # (2n,) 入力: 取り付け角[deg]
    alpha_max,  # スカラー[deg]
    alpha_min,  # スカラー[deg]
):

    n2 = 2 * Wing["span_div"]

    y = cp[1, :]  # スパン方向座標
    denom = state["Vair"] - _RAD * state["r"] * y  # 有効主流（ヨー角速度補正）
    roll_term = _DEG * np.arctan((_RAD * state["p"] * y) / np.maximum(denom, 1e-12))
    sideslip_term = _DEG * np.arctan(
        (state["Vair"] * np.sin(_RAD * state["beta"]) * np.sin(_RAD * dihedral_angle))
        / np.maximum(denom, 1e-12)
    )

    # 有効迎角（度）
    # α_eff = α（機体） + 取付角 + 誘導迎角 + ロール項 + 横滑り/上反角項
    alpha_effective[:] = (
        state["alpha"] + setting_angle - alpha_induced + roll_term + sideslip_term
    )

    # 収束安定のためのクリップ
    np.clip(alpha_effective, alpha_min, alpha_max, out=alpha_effective)
