"""Volume / planform / volumetric efficiency (PSWR-1 §5.1)."""

from __future__ import annotations

import numpy as np

from .variable_wedge import VariableWedgeWaverider

# numpy 2.x renamed trapz -> trapezoid; keep working on both
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def planform_area(wr: VariableWedgeWaverider) -> float:
    """Full-vehicle planform area S = int_{-y_tip}^{y_tip} (x_b - x_LE(y)) dy."""
    y = wr.y_grid
    chord = wr.body_length - wr.leading_edge[:, 0]
    return float(_trapz(chord, y))


def body_volume(wr: VariableWedgeWaverider) -> float:
    """Full-vehicle volume.

    For the variable-wedge family, at fixed y the wedge thickness at chord
    position s = x - x_LE(y) is

        t(x, y) = (x - x_LE(y)) * tan(theta(y))

    so the spanwise volume contribution is

        V(y) = int_{x_LE}^{x_b} t(x, y) dx
             = (x_b - x_LE(y))^2 / 2 * tan(theta(y))

    and the body volume is V_body = int V(y) dy across the full span.
    """
    y = wr.y_grid
    chord = wr.body_length - wr.leading_edge[:, 0]
    V_y = 0.5 * chord**2 * np.tan(wr.theta_y)
    return float(_trapz(V_y, y))


def volume_efficiency(wr: VariableWedgeWaverider) -> float:
    """eta_V = V^(2/3) / S_planform."""
    V = body_volume(wr)
    S = planform_area(wr)
    if S <= 0.0 or V <= 0.0:
        return 0.0
    return V ** (2.0 / 3.0) / S


def caret_analytic(M_inf: float, beta_deg: float, Lambda_deg: float,
                   L: float, gamma: float = 1.4) -> dict:
    """Closed-form caret reference for the Phase-1 validation gate.

    Returns volume, planform area, and eta_V using the *exact* analytical
    integrals (no numerical quadrature) for a constant-beta sweep.
    """
    import math
    beta = math.radians(beta_deg)
    Lam = math.radians(Lambda_deg)
    # theta from theta-beta-M
    s = math.sin(beta)
    num = 2.0 / math.tan(beta) * (M_inf*M_inf * s*s - 1.0)
    den = M_inf*M_inf * (gamma + math.cos(2.0 * beta)) + 2.0
    theta = math.atan(num / den)
    tan_theta = math.tan(theta)
    tan_L = math.tan(Lam)

    # Full body (spec's V is full-span)
    V = tan_theta * L**3 / (3.0 * tan_L)
    S = L*L / tan_L
    eta = V ** (2.0/3.0) / S
    cp_low = 4.0 * (M_inf*M_inf * s*s - 1.0) / ((gamma + 1.0) * M_inf*M_inf)
    return {"V": V, "S": S, "eta_V": eta, "theta": theta, "Cp_low": cp_low}
