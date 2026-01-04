import numpy as np


def update_circulation(Wing, state, iteration, circulation, circulation_old):
    coef = 0.18  # 緩和係数 これ以上は収束しにくくなるよ。

    n2 = 2 * int(Wing["span_div"])

    # 形チェック（(n2,) か (n2,1) を許容）
    def _view1d(a, name):
        if a.ndim == 2 and 1 in a.shape:
            return a.reshape(-1)
        if a.ndim != 1:
            raise ValueError(f"{name} は1次元配列にしてください（今は {a.shape} ）")
        if a.shape[0] != n2:
            raise ValueError(
                f"{name} の長さは {n2} にしてください（今は {a.shape[0]} ）"
            )
        return a

    circulation = _view1d(circulation, "circulation")
    circulation_old = _view1d(circulation_old, "circulation_old")

    if iteration > 1:
        # Γ_new = Γ_old + coef*(Γ - Γ_old)
        circulation[:] = circulation_old + coef * (circulation - circulation_old)

    # Γ_old を更新（VBと同じ順序で常に代入）
    circulation_old[:] = circulation
