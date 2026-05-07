"""Surface-panel aero evaluator (GVWD §4.8).

Algorithm per spec §4.8:
  1. Decompose mesh into triangular panels.
  2. Per panel:
     - outward normal n̂
     - local incidence theta_local = -arcsin(n̂ · v̂_inf)  (windward = +)
     - if theta_local <= 0   : Cp = 0      (shadow)
     - if 0 < theta_local <= theta_max(M_inf) : tangent-wedge attached
     - if theta_local > theta_max(M_inf)      : modified Newtonian
  3. Force per panel (per unit q_inf): dF = -Cp A n̂.
  4. Sum and project onto lift / drag / pitching-moment axes.

Frame: body x forward, z up (gvwd geometry frame).
AoA convention: positive alpha = nose up. v̂_inf in body frame =
(cos alpha, 0, -sin alpha).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from gvwd.geometry.mesh import Mesh
from gvwd.thermo.tangent_wedge import tangent_wedge_cp_array


@dataclass
class PanelAeroResult:
    """Output of :func:`panel_aero_coefficients`."""
    CL: float
    CD: float
    Cm: float
    LD: float
    L_over_q: float                 # full-vehicle lift force / q_inf [m^2]
    D_over_q: float                 # full-vehicle drag force / q_inf [m^2]
    M_over_q: float                 # full-vehicle pitching moment / q_inf [m^3]
    Cp: np.ndarray                  # per-face Cp (n_faces,)
    regime_code: np.ndarray         # per-face regime: 0=shadow, 1=attached, 2=Newtonian
    F_per_face: np.ndarray          # (n_faces, 3) force per face / q_inf
    S_ref: float                    # reference area
    L_ref: float                    # reference length
    M_inf: float
    alpha_deg: float


def freestream_direction(alpha_rad: float) -> np.ndarray:
    """v̂_inf (freestream propagation direction) in body frame for
    nose-up positive alpha.

    GVWD body-frame convention: apex at x=0 (upstream end), body extends
    to +x (downstream). Freestream propagates in +x_body direction at
    alpha = 0. Pitching the body nose-up by alpha means the body x-axis
    rotates UP relative to the inertial frame, equivalent to the
    freestream direction rotating DOWN... wait no — the standard result
    is that v_inf in body frame has a +sin(alpha) z-component (the
    freestream "passes over the top of the nose" when the body is pitched
    nose-up, so in body frame the wind has a +z component).

    Verified by the unit tests: with this convention, the LOWER surface
    of a delta wedge body becomes windward at positive alpha (giving
    positive CL, as expected) and the UPPER surface enters shadow.
    """
    return np.array([math.cos(alpha_rad), 0.0, math.sin(alpha_rad)])


def lift_direction(alpha_rad: float) -> np.ndarray:
    """ẑ_lift in body frame: perpendicular to v̂_inf in the x-z plane,
    pointing 'up' (= +z_body when alpha = 0). For nose-up positive
    alpha, this tilts the lift vector slightly toward -x (forward of
    the body's vertical)."""
    return np.array([-math.sin(alpha_rad), 0.0, math.cos(alpha_rad)])


def panel_aero_coefficients(
    mesh: Mesh,
    M_inf: float,
    alpha_rad: float = 0.0,
    *,
    gamma: float = 1.4,
    S_ref: Optional[float] = None,
    L_ref: Optional[float] = None,
    x_ref: Optional[float] = None,
) -> PanelAeroResult:
    """Inviscid surface-panel aero on a closed Mesh.

    Parameters
    ----------
    mesh        : closed surface mesh
    M_inf       : freestream Mach
    alpha_rad   : angle of attack [rad], positive nose-up
    gamma       : specific heat ratio
    S_ref       : reference area for nondimensionalisation [m^2]. If None,
                  the planform projection of the mesh on z=0 is used.
    L_ref       : reference length for the moment coefficient [m]. If None,
                  the streamwise extent of the mesh (max x - min x) is used.
    x_ref       : streamwise position of the moment reference [m]. If None,
                  the streamwise midpoint is used.
    """
    v_inf = freestream_direction(alpha_rad)
    z_lift = lift_direction(alpha_rad)

    n = mesh.face_normals()       # (M, 3) outward unit normals
    A = mesh.face_areas()         # (M,)
    c = mesh.face_centroids()     # (M, 3)

    # Local windward deflection (positive = windward, negative = shadow).
    n_dot_v = np.clip(n @ v_inf, -1.0, 1.0)
    theta_local = -np.arcsin(n_dot_v)

    # Cp via tangent-wedge with Newtonian fallback.
    Cp, regime_code = tangent_wedge_cp_array(M_inf, theta_local, gamma)

    # Per-face force per unit dynamic pressure: dF_aero = -Cp A n̂
    F_pf = -Cp[:, None] * A[:, None] * n
    F_total = F_pf.sum(axis=0)

    L_q = float(F_total @ z_lift)
    # Drag is the component of the aerodynamic force in the freestream
    # propagation direction (drag opposes body motion; body moves in
    # -v_inf in inertial frame, so drag-on-body is in +v_inf direction).
    D_q = float(F_total @ v_inf)

    # Reference geometry
    if S_ref is None:
        S_ref = _projected_planform_area(mesh)
    if L_ref is None or x_ref is None:
        x_min = float(mesh.vertices[:, 0].min())
        x_max = float(mesh.vertices[:, 0].max())
        if L_ref is None:
            L_ref = x_max - x_min
        if x_ref is None:
            x_ref = 0.5 * (x_min + x_max)

    # Pitching moment about (x_ref, 0, 0): nose-up positive.
    # M = sum (r - r_ref) x dF, take y-component.
    r_rel = c - np.array([x_ref, 0.0, 0.0])
    moments = np.cross(r_rel, F_pf)
    M_q = float(moments[:, 1].sum())

    CL = L_q / S_ref
    CD = D_q / S_ref
    Cm = M_q / (S_ref * L_ref)
    LD = CL / CD if CD > 1e-12 else math.inf

    return PanelAeroResult(
        CL=CL, CD=CD, Cm=Cm, LD=LD,
        L_over_q=L_q, D_over_q=D_q, M_over_q=M_q,
        Cp=Cp, regime_code=regime_code, F_per_face=F_pf,
        S_ref=S_ref, L_ref=L_ref,
        M_inf=M_inf, alpha_deg=math.degrees(alpha_rad),
    )


def _projected_planform_area(mesh: Mesh) -> float:
    """Projected planform area on z=0 plane (sum of upward-facing triangle
    projections)."""
    n = mesh.face_normals()
    A = mesh.face_areas()
    # The total projected area of upward-facing faces equals the planform
    # area (silhouette).
    upward = n[:, 2] > 1e-9
    return float(np.sum(A[upward] * n[upward, 2]))
