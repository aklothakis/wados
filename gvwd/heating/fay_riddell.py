"""Fay-Riddell-style stagnation-point heating (GVWD §4.10).

Uses the **Tauber-Sutton 1991** convective correlation (which is the form
that matches the spec §5.4 DoD numerical gates: 50-200 MW/m^2 for sharp
1 mm LE at M=15, h=30 km):

    q_dot_s [W/m^2] = 1.83e-4 * sqrt(rho_inf / R_N) * V_inf^3.15

with sweep correction sqrt(cos Lambda_eff) for swept leading edges
(Beckwith-Cohen high-Re_theta limit).

NOTE: this is the *engineering* form often called "Fay-Riddell-style"
in the heating literature even though the original 1958 Fay-Riddell paper
gave a more elaborate Pr / Le / enthalpy-difference form. The Tauber-
Sutton form is widely used as the calibrated correlation for
hypersonic-vehicle preliminary design (Tauber & Sutton, JSR 1991), and
matches the spec's DoD numerical bounds.

The full Fay-Riddell form (with Pr, Le, h_aw, h_w, density-viscosity
products at edge and wall) is provided as :func:`fay_riddell_full` for
users who want the more detailed expression.
"""

from __future__ import annotations

import math
from typing import Optional

# Sutton-Graves / Tauber-Sutton 1991 coefficient
# q_s [W/m^2] = K_TS * sqrt(rho/R) * V^3.15
K_TAUBER_SUTTON = 1.83e-4


def stagnation_point_heat_flux(rho_inf: float, V_inf: float,
                                R_N: float) -> float:
    """Tauber-Sutton 1991 stagnation-point convective heat flux [W/m^2].

    Parameters
    ----------
    rho_inf : freestream density [kg/m^3]
    V_inf   : freestream velocity [m/s]
    R_N     : stagnation-point nose radius [m]

    Returns
    -------
    q_dot_s : stagnation-point heat flux [W/m^2]
    """
    if R_N <= 0:
        raise ValueError(f"R_N must be > 0; got {R_N}")
    return K_TAUBER_SUTTON * math.sqrt(rho_inf / R_N) * V_inf ** 3.15


def swept_LE_heat_flux(rho_inf: float, V_inf: float, R_LE: float,
                        Lambda_rad: float) -> float:
    """Stagnation-line heat flux on a swept cylindrical leading edge.

    Apply the Beckwith-Cohen sweep correction sqrt(cos(Lambda)) to the
    base unswept Tauber-Sutton form:

        q_swept = q_unswept * sqrt(cos(Lambda))

    Returns 0 if Lambda >= 90 deg (degenerate).
    """
    cos_L = math.cos(Lambda_rad)
    if cos_L <= 0.0:
        return 0.0
    q_unswept = stagnation_point_heat_flux(rho_inf, V_inf, R_LE)
    return q_unswept * math.sqrt(cos_L)


def nose_heat_flux(rho_inf: float, V_inf: float, R_nose: float) -> float:
    """Spherical-nose stagnation-point heat flux. Same form as the
    unswept LE."""
    return stagnation_point_heat_flux(rho_inf, V_inf, R_nose)


# ----------------------------------------------------------------------
#  Full Fay-Riddell form (per spec §4.10), provided for completeness
# ----------------------------------------------------------------------

def fay_riddell_full(rho_e: float, mu_e: float, rho_w: float, mu_w: float,
                      du_e_ds: float, h_0e: float, h_w: float,
                      Pr: float = 0.715, Le: float = 1.4,
                      h_D: float = 0.0) -> float:
    """Full Fay-Riddell 1958 convective heat flux at a stagnation point.

        q_dot = 0.763 * Pr^(-0.6) * (rho_e mu_e)^0.4 * (rho_w mu_w)^0.1
                * sqrt(du_e/dx) * (h_0e - h_w) * [1 + (Le^0.52 - 1)(h_D/h_0e)]

    Velocity gradient at stagnation: du_e/dx = (1/r) * sqrt(2(p_e-p_inf)/rho_e)
    must be supplied by the caller. h_D is the dissociation enthalpy
    contribution (set 0 to disable the Lewis-number correction).
    """
    bracket = 1.0 + (Le ** 0.52 - 1.0) * (h_D / max(h_0e, 1.0))
    return (0.763 * Pr ** (-0.6) * (rho_e * mu_e) ** 0.4
             * (rho_w * mu_w) ** 0.1 * math.sqrt(max(du_e_ds, 0.0))
             * (h_0e - h_w) * bracket)
