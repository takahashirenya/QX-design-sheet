import numpy as np


def calculation_Qij(
    Wing, state, ds, yp, zp, ymp, zmp, Rpij, Rmij, Rpijm, Rmijm, dihedral_angle, Qij
):

    n2 = 2 * int(Wing["span_div"])
    rad = np.pi / 180.0

    # 角度配列
    a = dihedral_angle * rad  # shape (n2,)

    ds_arr = np.asarray(ds, dtype=float)
    if ds_arr.ndim == 0:
        ds_arr = np.full(n2, float(ds_arr))
    elif ds_arr.shape[0] != n2:
        raise ValueError(f"ds length mismatch: {ds_arr.shape[0]} != {n2}")

    for j in range(n2):
        dsj = ds_arr[j]  # ★ パネル j の半幅

        a_diff = a - a[j]
        a_sum = a + a[j]
        c1 = np.cos(a_diff)
        s1 = np.sin(a_diff)
        c2 = np.cos(a_sum)
        s2 = np.sin(a_sum)

        Rp2 = Rpij[:, j] * Rpij[:, j]
        Rm2 = Rmij[:, j] * Rmij[:, j]
        Rp2m = Rpijm[:, j] * Rpijm[:, j]
        Rm2m = Rmijm[:, j] * Rmijm[:, j]

        yp_j = yp[:, j]
        zp_j = zp[:, j]
        ymp_j = ymp[:, j]
        zmp_j = zmp[:, j]

        term1 = (-(yp_j - dsj) / Rp2 + (yp_j + dsj) / Rm2) * c1
        term2 = (-zp_j / Rp2 + zp_j / Rm2) * s1
        term3 = ((ymp_j - dsj) / Rp2m - (ymp_j + dsj) / Rm2m) * c2
        term4 = (zmp_j / Rp2m - zmp_j / Rm2m) * s2

        Qij[:, j] = term1 + term2 + term3 + term4
