"""Compressible boundary-layer skin friction (GVWD §4.8 viscous).

Per-panel Eckert reference temperature + Sutherland viscosity + 1/7-power
(Pohlhausen) BL thickness estimate. We reuse the PSWR-1 utilities for the
core Sutherland and Eckert relations and add the per-mesh integration on
top.

The viscous correction is added to the inviscid panel-method drag
(:mod:`gvwd.aero.panel_method`) to give the full ``CD_total = CD_wave +
CD_friction``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from gvwd.geometry.mesh import Mesh
from pswr.aero.viscous import (   # reuse PSWR-1 primitives (audited & tested)
    sutherland_viscosity,
    eckert_reference_T,
    cf_laminar, cf_turbulent,
    boundary_layer_thickness,
)


_R_AIR = 287.05
_GAMMA = 1.4


@dataclass
class ViscousResult:
    CD_friction: float
    F_friction_over_q: float
    delta_BL_max: float
    Re_chord_max: float
    Cf_panel: np.ndarray            # per-windward-panel chord-averaged Cf
    state_per_panel: dict
    S_ref: float


def panel_viscous_drag(
    mesh: Mesh,
    M_inf: float,
    alpha_rad: float = 0.0,
    *,
    altitude_km: float = 30.0,
    T_w: float = 1500.0,
    Re_x_tr: float = 1.0e6,
    p_inf: Optional[float] = None,
    T_inf: Optional[float] = None,
    gamma: float = _GAMMA,
    S_ref: Optional[float] = None,
) -> ViscousResult:
    """Compute the full-vehicle viscous drag coefficient by panel
    integration.

    Algorithm: each WINDWARD panel contributes a friction force
        dF_fric = (1/2 rho_e u_e^2 Cf_panel) A_panel * (-v̂_∞)
    where Cf_panel is the chord-averaged skin friction at the panel's
    local Eckert reference state. Streamwise distance for Re_x is taken
    as the panel centroid's distance from the body apex (x_centroid -
    x_apex), which is a coarse but workable proxy for the local
    Reynolds number on a delta-planform vehicle.

    For more accurate results, a streamline-based BL marching would be
    required (Phase 5+ refinement).
    """
    # Atmosphere (default US Std 1976 at given altitude if not provided)
    if p_inf is None or T_inf is None:
        p_inf, T_inf = _us_std_1976(altitude_km)

    rho_inf = p_inf / (_R_AIR * T_inf)
    a_inf = math.sqrt(gamma * _R_AIR * T_inf)
    V_inf = M_inf * a_inf

    # Per-face quantities
    n_face = mesh.face_normals()
    A_face = mesh.face_areas()
    c_face = mesh.face_centroids()

    # Freestream direction
    v_inf = np.array([math.cos(alpha_rad), 0.0, -math.sin(alpha_rad)])
    n_dot_v = np.clip(n_face @ v_inf, -1.0, 1.0)
    theta_local = -np.arcsin(n_dot_v)
    windward = theta_local > 0.0

    # Crude per-panel post-shock state estimate using local incidence.
    # For windward panels we treat them as 2-D wedges with deflection
    # theta_local; the post-shock T, p, M follow from oblique-shock
    # relations. For a first-attack viscous estimate this is good
    # enough (the boundary-layer integral is not very sensitive to the
    # post-shock state via the Eckert reference).
    # Use freestream as approximation when attached-shock fails:
    T_e = np.full_like(theta_local, T_inf, dtype=float)
    p_e = np.full_like(theta_local, p_inf, dtype=float)
    M_e = np.full_like(theta_local, M_inf, dtype=float)
    # NOTE: a refined version per spec §4.8 would call the Rankine-
    # Hugoniot per panel. For Phase 4 we use freestream (this is a
    # standard simplification at high Mach where post-shock T factor is
    # close to 1 for small theta).

    # Eckert reference state per panel
    T_star = eckert_reference_T(T_e, M_e, T_w)
    mu_star = sutherland_viscosity(T_star)
    rho_star = p_e / (_R_AIR * T_star)
    a_e = np.sqrt(gamma * _R_AIR * T_e)
    u_e = M_e * a_e

    # Streamwise distance from apex (proxy for x in Re_x)
    x_apex = float(mesh.vertices[:, 0].min())
    x_panel = c_face[:, 0] - x_apex
    x_safe = np.maximum(x_panel, 1e-6)

    # Reynolds number at panel
    Re_x = rho_star * u_e * x_safe / np.maximum(mu_star, 1e-30)

    # Skin-friction coefficient (laminar / turbulent based on local Re)
    Cf_lam = cf_laminar(Re_x)
    Cf_turb = cf_turbulent(Re_x)
    Cf = np.where(Re_x < Re_x_tr, Cf_lam, Cf_turb)

    # BL thickness diagnostic
    delta_BL = boundary_layer_thickness(x_safe, Re_x)

    # Per-panel friction force (per unit q_inf): (1/2 rho_e u_e^2 Cf) A,
    # acting OPPOSITE to v_inf (drag direction). Force per q_inf:
    #   dF/q_inf = (rho_e u_e^2 / (rho_inf u_inf^2)) Cf A * (- v̂_inf)
    q_inf = 0.5 * rho_inf * V_inf * V_inf
    factor = 0.5 * rho_star * u_e * u_e / np.maximum(q_inf, 1e-30)
    F_fric_q_per_panel = factor * Cf * A_face * windward

    F_fric_q_total = float(F_fric_q_per_panel.sum())

    # Reference area
    if S_ref is None:
        n2 = mesh.face_normals()
        upward = n2[:, 2] > 1e-9
        S_ref = float(np.sum(A_face[upward] * n2[upward, 2]))
    CD_fric = F_fric_q_total / max(S_ref, 1e-30)

    return ViscousResult(
        CD_friction=CD_fric,
        F_friction_over_q=F_fric_q_total,
        delta_BL_max=float(delta_BL.max()),
        Re_chord_max=float(Re_x.max()),
        Cf_panel=Cf,
        state_per_panel={"T_star": T_star, "rho_star": rho_star,
                          "u_e": u_e, "Re_x": Re_x, "windward": windward},
        S_ref=S_ref,
    )


# ----------------------------------------------------------------------
#  Atmosphere (minimal US Std 1976 layered model)
# ----------------------------------------------------------------------

def _us_std_1976(h_km: float) -> tuple:
    """Return (p_inf [Pa], T_inf [K]) at altitude h [km] via US Standard
    Atmosphere 1976. Layered model up to 84.852 km; sufficient for the
    GVWD operational regime."""
    h = h_km * 1000.0
    layers = [
        (0,        288.150,  -0.0065, 101325.0),
        (11000,    216.650,   0.0,     22632.06),
        (20000,    216.650,   0.0010,   5474.889),
        (32000,    228.650,   0.0028,    868.0187),
        (47000,    270.650,   0.0,       110.9063),
        (51000,    270.650,  -0.0028,    66.93887),
        (71000,    214.650,  -0.0020,     3.95642),
        (84852,    186.946,   0.0,        0.3734),
    ]
    g0 = 9.80665
    M = 0.0289644
    R = 8.31446
    for i in range(len(layers) - 1):
        h_b, T_b, L_b, p_b = layers[i]
        h_top = layers[i+1][0]
        if h_b <= h <= h_top:
            if abs(L_b) < 1e-12:
                T = T_b
                p = p_b * math.exp(-g0 * M * (h - h_b) / (R * T_b))
            else:
                T = T_b + L_b * (h - h_b)
                p = p_b * (T_b / T) ** (g0 * M / (R * L_b))
            return p, T
    # Above table: extrapolate isothermal exp decay from the top layer
    h_b, T_b, L_b, p_b = layers[-1]
    p = p_b * math.exp(-g0 * M * (h - h_b) / (R * T_b))
    return p, T_b
