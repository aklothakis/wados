"""Inviscid waverider aerodynamic coefficients (PSWR-1 §5.6).

Reference frame as in :mod:`pswr.geometry.variable_wedge`:
    +x downstream, +z up. Lift is +z.

Lower-surface pressure coefficient at each spanwise station:
    C_p,low(y) = 4 (M_inf^2 sin^2 beta(y) - 1) / ((gamma+1) M_inf^2)

Upper surface is at freestream (planar, parallel to x), so C_p,up = 0.

The lower-surface elemental area dA can be decomposed into a planform
projection ``dy dx`` and a vertical-projection ``tan(theta(y)) dy dx``. The
outward normal of the lower surface is

    n_hat = ( sin(theta(y)), 0, -cos(theta(y)) )    (points down/forward)

so the lift component is ``-n_hat . z_hat = cos(theta)`` and the drag
component is ``-n_hat . x_hat = -sin(theta)`` (force on body from fluid is
``-Cp * n_hat * dA`` per unit dynamic pressure). After multiplying by the
slant area ``dA = dy dx / cos(theta)``, the lift and drag integrals reduce to

    L/q_inf = int Cp_low(y) (x_b - x_LE(y)) dy
    D/q_inf = int Cp_low(y) (x_b - x_LE(y)) tan(theta(y)) dy

The dependence on x integrates to the chord length at each station, hence the
result is closed-form per §5.6 of the spec.
"""

from __future__ import annotations

import math
import numpy as np

from ..geometry.variable_wedge import VariableWedgeWaverider
from ..geometry.volume import planform_area
from ..thermo.oblique_shock import cp_lower_wedge

_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def _cp_lower_array(M_inf: float, beta_y: np.ndarray,
                    gamma: float) -> np.ndarray:
    s = np.sin(beta_y)
    return 4.0 * (M_inf*M_inf * s*s - 1.0) / ((gamma + 1.0) * M_inf*M_inf)


def inviscid_coefficients(wr: VariableWedgeWaverider,
                          x_ref_frac: float = 0.5) -> dict:
    """Closed-form inviscid C_L, C_D,wave, C_m about ``x_ref = x_b * x_ref_frac``.

    Reference area = planform area (full vehicle).

    Returns dict with keys: ``CL``, ``CD``, ``Cm``, ``LD``, ``Cp_low_y``,
    ``S_ref``, ``L_over_q``, ``D_over_q``, ``M_over_q``.
    """
    y = wr.y_grid
    chord = wr.body_length - wr.leading_edge[:, 0]
    cp = _cp_lower_array(wr.M_inf, wr.beta_y, wr.gamma)
    tan_th = np.tan(wr.theta_y)

    # Force per unit dynamic pressure (full vehicle integrals)
    L_q = float(_trapz(cp * chord, y))                    # lift  [m^2]
    D_q = float(_trapz(cp * chord * tan_th, y))           # drag  [m^2]

    # Pitching moment about (x_ref, 0, 0). The lower surface contribution
    # comes from the pressure force acting at chord midpoint (constant Cp
    # along x at each y). Force at station y is per unit y:
    #   dF/dy = q_inf * Cp(y) * chord(y) * (-n_hat) * (1/cos theta)*cos theta
    # giving per-unit-y vertical force +cos(theta) Cp chord (for Cp>0 below)
    # and horizontal force +sin(theta) Cp chord. Moment arm to x_ref is
    # x_mid(y) - x_ref where x_mid = (x_LE + x_b)/2.
    x_ref = wr.body_length * x_ref_frac
    x_mid = 0.5 * (wr.leading_edge[:, 0] + wr.body_length)
    M_q = float(_trapz(cp * chord * (x_mid - x_ref), y))   # nose-up positive

    S_ref = planform_area(wr)
    if S_ref <= 0.0:
        raise ValueError("Planform area is zero — degenerate geometry")
    L_ref = wr.body_length

    CL = L_q / S_ref
    CD = D_q / S_ref
    Cm = M_q / (S_ref * L_ref)
    LD = CL / CD if CD > 1e-12 else math.inf

    return {
        "CL": CL,
        "CD": CD,
        "Cm": Cm,
        "LD": LD,
        "Cp_low_y": cp,
        "S_ref": S_ref,
        "L_over_q": L_q,
        "D_over_q": D_q,
        "M_over_q": M_q,
        "x_ref": x_ref,
    }


def cl_cd_caret_analytic(M_inf: float, beta_deg: float,
                         Lambda_deg: float,
                         L: float = 10.0,
                         gamma: float = 1.4) -> dict:
    """Closed-form Nonweiler caret CL, CD reference (no integration).

    For constant beta and constant Lambda, the integrals reduce to:
        L/q_inf = Cp_low * S_planform
        D/q_inf = Cp_low * S_planform * tan(theta)

    so:
        CL = Cp_low
        CD = Cp_low * tan(theta)
        L/D = 1/tan(theta)
    """
    beta = math.radians(beta_deg)
    Lam = math.radians(Lambda_deg)
    s = math.sin(beta)
    cp = 4.0 * (M_inf*M_inf * s*s - 1.0) / ((gamma + 1.0) * M_inf*M_inf)
    num = 2.0 / math.tan(beta) * (M_inf*M_inf * s*s - 1.0)
    den = M_inf*M_inf * (gamma + math.cos(2.0 * beta)) + 2.0
    theta = math.atan(num / den)
    tan_theta = math.tan(theta)
    return {
        "CL": cp,
        "CD": cp * tan_theta,
        "LD": 1.0 / tan_theta if tan_theta > 0 else math.inf,
        "Cp_low": cp,
        "theta_deg": math.degrees(theta),
    }
