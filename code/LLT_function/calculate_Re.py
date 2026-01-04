import numpy as np

_RAD = np.pi / 180.0


def calculate_Re(Wing, state, Re, cp, chord_cp, Re_max, Re_min):
    y = cp[1, :]
    Ueff = state["Vair"] - _RAD * state["r"] * y

    Re[:] = (Ueff * chord_cp) / state["mu"]

    # 収束安定のためにクリップ
    np.clip(Re, Re_min, Re_max, out=Re)
