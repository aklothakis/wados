"""Final aero-coefficient assembly (GVWD §4.8, §5.4 DoD).

Combines panel-method inviscid (CD_wave) with viscous correction
(CD_friction) and assembles a clean breakdown dict.
"""

from __future__ import annotations

import math
from typing import Optional

from gvwd.geometry.mesh import Mesh
from gvwd.aero.panel_method import panel_aero_coefficients, PanelAeroResult
from gvwd.aero.viscous import panel_viscous_drag, ViscousResult


def aero_coefficients_full(
    mesh: Mesh,
    M_inf: float,
    alpha_rad: float = 0.0,
    *,
    altitude_km: float = 30.0,
    T_w: float = 1500.0,
    Re_x_tr: float = 1.0e6,
    p_inf: Optional[float] = None,
    T_inf: Optional[float] = None,
    gamma: float = 1.4,
    S_ref: Optional[float] = None,
    L_ref: Optional[float] = None,
    x_ref: Optional[float] = None,
) -> dict:
    """Full coefficient breakdown for a closed mesh at (M_inf, alpha).

    Returns
    -------
    dict with keys:
      M_inf, alpha_deg
      CL, CD_total, CD_wave, CD_friction, Cm, L_over_D
      panel: PanelAeroResult
      viscous: ViscousResult
      altitude_km, T_w
      Re_chord_max, delta_BL_max
    """
    inv = panel_aero_coefficients(
        mesh, M_inf, alpha_rad,
        gamma=gamma, S_ref=S_ref, L_ref=L_ref, x_ref=x_ref,
    )
    visc = panel_viscous_drag(
        mesh, M_inf, alpha_rad,
        altitude_km=altitude_km, T_w=T_w, Re_x_tr=Re_x_tr,
        p_inf=p_inf, T_inf=T_inf, gamma=gamma,
        S_ref=inv.S_ref,
    )
    CD_total = inv.CD + visc.CD_friction
    LD = inv.CL / CD_total if CD_total > 1e-12 else math.inf
    return {
        "M_inf": M_inf,
        "alpha_deg": math.degrees(alpha_rad),
        "CL": inv.CL,
        "CD_total": CD_total,
        "CD_wave": inv.CD,
        "CD_friction": visc.CD_friction,
        "Cm": inv.Cm,
        "LD": LD,
        "panel": inv,
        "viscous": visc,
        "altitude_km": altitude_km,
        "T_w": T_w,
        "Re_chord_max": visc.Re_chord_max,
        "delta_BL_max": visc.delta_BL_max,
        "S_ref": inv.S_ref,
        "L_ref": inv.L_ref,
    }
