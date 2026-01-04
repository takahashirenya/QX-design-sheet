import numpy as np

def calculation_Rij(Wing, state, ds, yp, zp, ymp, zmp, Rpij, Rmij, Rpijm, Rmijm):

    n2 = 2 * int(Wing["span_div"])

    # 形チェック
    expected = (n2, n2)
    for name, arr in [
        ("yp", yp), ("zp", zp), ("ymp", ymp), ("zmp", zmp),
        ("Rpij", Rpij), ("Rmij", Rmij), ("Rpijm", Rpijm), ("Rmijm", Rmijm),
    ]:
        if arr.shape != expected:
            raise ValueError(f"{name} の形が {arr.shape} です。期待は {expected} です。")
    # Rij の計算
    np.hypot(yp - ds, zp, out=Rpij)
    np.hypot(yp + ds, zp, out=Rmij)

    # 鏡像
    np.hypot(ymp - ds, zmp, out=Rpijm)
    np.hypot(ymp + ds, zmp, out=Rmijm)
