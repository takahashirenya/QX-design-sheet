import numpy as np
import math

_RAD = math.pi / 180.0


def calculation_force(
    Wing,
    state,
    iteration: int,
    Lift: np.ndarray,
    Induced_Drag: np.ndarray,
    cp: np.ndarray,
    circulation: np.ndarray,
    dihedral_angle: np.ndarray,
    wi: np.ndarray,
    CL: np.ndarray,
    chord_cp: np.ndarray,
    last_call: bool = False,
):
    """
    VBA Sub calculation_Force の Python 版（ベクトル化）。
    Lift/Induced_Drag は iteration 番目の要素に“加算”します（VBAと同じ累積）。
    """
    m = chord_cp.shape[0]
    assert cp.shape[1] == m and cp.shape[0] in (2, 3), "cp must be (2,m) or (3,m)"
    assert dihedral_angle.shape == (m,)
    assert wi.shape == (m,)
    assert CL.shape == (m,)
    assert circulation.shape == (m,)

    # --- 取り出し（dict/属性両対応） ---
    Vair = float(getattr(state, "Vair", state["Vair"]))
    rho = float(getattr(state, "rho", state["rho"]))
    r = float(getattr(state, "r", state["r"]))

    # dy: _dy_full_cache -> dy -> Wing['dy']
    if hasattr(Wing, "_dy_full_cache"):
        dy_full = np.asarray(Wing._dy_full_cache, dtype=float)
    else:
        dy_in = getattr(Wing, "dy", None)
        if dy_in is None and isinstance(Wing, dict):
            dy_in = Wing.get("dy", None)
        if dy_in is None:
            raise RuntimeError(
                "dy が見つかりません（Wing._dy_full_cache / Wing.dy / Wing['dy']）。"
            )
        dy_full = (
            np.full(m, float(dy_in), dtype=float)
            if np.isscalar(dy_in)
            else np.asarray(dy_in, dtype=float)
        )

    if dy_full.shape != (m,):
        raise ValueError(
            "Wing.dy はスカラーまたは shape=(2n,) の配列である必要があります。"
        )

    # --- 局所速度・循環 ---
    y_cp = cp[1, :]
    V_local = Vair - _RAD * r * y_cp  # (m,)

    if last_call:
        pass
    else:
        circulation[:] = 0.5 * chord_cp * V_local * CL  # (m,)

    # --- 力の累積 ---
    cos_dih = np.cos(_RAD * dihedral_angle)
    # Lift(iteration) += ρ * V_local * Γ * dy * cos(dihedral)
    Lift[iteration] += float(np.sum(rho * V_local * circulation * dy_full * cos_dih))
    # Induced_Drag(iteration) += ρ * wi * Γ * dy
    dDi = rho * circulation * wi * dy_full  # [N]
    Wing["dDi"] = dDi  # 最後の呼び出し時に保存
    Induced_Drag[iteration] += float(np.sum(dDi))

    return Lift, Induced_Drag, circulation
