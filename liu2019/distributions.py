"""Spanwise distributions and exit-plane curves from Liu et al. 2019 Section 1.1.

- Ma(z): second-order polynomial (paper Eq. 1)
- y(z):  cubic freestream-surface trailing edge (paper Eq. 2)
- y_s(z): piecewise-quartic shock curve in base plane (paper Eq. 3)
- Coefficients derived via paper Eqs. 4-8.
"""

import numpy as np


def Ma_distribution(z, W, Ma_center, Ma_tip):
    """Paper Eq. 1: Ma(z) = m*z^2 + n.

    Coefficients are chosen so that Ma(0) = Ma_center and Ma(+/- W/2) = Ma_tip.
    Accepts scalar or array z (in metres).
    """
    n = Ma_center
    m = (Ma_tip - Ma_center) / (W / 2.0) ** 2
    return m * np.asarray(z, dtype=float) ** 2 + n


def upper_surface_trailing_edge(z, a, b, c, d):
    """Paper Eq. 2: y(z) = a*z^3 + b*z^2 + c*z + d (freestream TE at base plane)."""
    z = np.asarray(z, dtype=float)
    return a * z ** 3 + b * z ** 2 + c * z + d


def shock_curve(z, A, L_s):
    """Paper Eq. 3 — piecewise quartic shock curve (ICC) in the base plane.

        y_s(z) = A*(z - L_s)^4   for z >=  L_s
               = 0               for -L_s < z < L_s
               = A*(z + L_s)^4   for z <= -L_s
    """
    z_arr = np.atleast_1d(np.asarray(z, dtype=float))
    y = np.zeros_like(z_arr)
    pos = z_arr >=  L_s
    neg = z_arr <= -L_s
    y[pos] = A * (z_arr[pos] - L_s) ** 4
    y[neg] = A * (z_arr[neg] + L_s) ** 4
    return y if y.shape != () else float(y)


def shock_curve_coefficient(y5, z5, L_s):
    """Paper Eq. 4: A = y5 / (z5 - L_s)^4  (requires z5 > L_s)."""
    if z5 <= L_s:
        raise ValueError(
            f"z5 (={z5}) must exceed L_s (={L_s}) for the quartic coefficient "
            "to be well defined."
        )
    return y5 / (z5 - L_s) ** 4


def upper_surface_coefficients(y5, z5, y6, z6, delta5, delta6, L_w, beta_deg):
    """Paper Eqs. 5-8 — derive (a, b, c, d) for the cubic freestream TE.

    Eq. 5:  d = L_w * tan(beta)
    Eq. 6:  c = d * tan(delta6)
    Eq. 7:  b = [3*(y5 - d - c*z5) - z5*(d*tan(delta5) + z5*c)] / z5^2
    Eq. 8:  a = (y5 - d - c*z5 - b*z5^2) / z5^3

    y6, z6 describe the centreline apex of the upper surface — they do not
    enter Eqs. 5-8 but are kept for validation (y6 must equal d when
    delta6 = 0 and the centreline is at z6 = 0).
    """
    beta = np.radians(beta_deg)
    d = L_w * np.tan(beta)
    c = d * np.tan(delta6)
    b = (3.0 * (y5 - d - c * z5)
         - z5 * (d * np.tan(delta5) + z5 * c)) / z5 ** 2
    a = (y5 - d - c * z5 - b * z5 ** 2) / z5 ** 3
    return a, b, c, d
