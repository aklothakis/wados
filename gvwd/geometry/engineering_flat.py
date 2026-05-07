"""Engineering glide-vehicle (flat-bottom) headline geometry (GVWD §4.4).

HTV-2 / Fattah-2 / Avangard archetype:
  - Triangular flat-bottom forebody (single inclined lower plane,
    horizontal-or-inclined upper plane)
  - Prismatic / linearly-interpolated centerbody connecting forebody TE
    to a rectangular base
  - Rectangular base of dimensions (2 b_base) x h_base
  - Optional nose blunting (r_nose) and LE blunting (r_LE) — Phase-3
    ships these as informational fields; cylindrical/spherical fillet
    meshes are added by a follow-on refinement (the DoD asks for "sharp
    1 mm" cases anyway).
  - Optional aft control fins (see :mod:`gvwd.geometry.fins`)

Topology (no blunting, no fins): 9 vertices, 14 triangles
  - Apex (1)
  - Forebody-TE corners at x=L_fore (4)  : F_UR, F_UL, F_LR, F_LL
  - Base corners at x=L_total = L_fore+L_center (4) : B_UR, B_UL, B_LR, B_LL
Faces:
  - Forebody (4 triangles) : upper, lower, side_right, side_left
  - Centerbody (8 triangles): upper-quad, lower-quad, side_right-quad,
    side_left-quad — each split into two triangles
  - Base (2 triangles, rectangular)

Frame: x downstream from apex, y spanwise, z vertical.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from gvwd.geometry.mesh import Mesh
from gvwd.thermo.oblique_shock import (
    obtain_beta, ShockDetachedError,
)


@dataclass
class EngineeringFlat:
    """HTV-2 / Fattah-2 / Avangard-archetype flat-bottom glide vehicle.

    Inputs (per spec §4.4 12-parameter set):

    M_design        : design Mach for forebody shock-attachment      (default 15)
    theta_fore      : forebody lower-surface inclination [rad]       (default 8 deg)
    Lambda          : LE sweep from spanwise axis [rad]              (default 75 deg)
    L_fore          : forebody (compression-surface) length [m]      (default 2.5 m)
    L_center        : centerbody length [m]                          (default 1.5 m)
    b_base          : half-width of base [m]                         (default 0.5 m)
    h_base          : base height [m]                                (default 0.4 m)
    h_fore_nose     : nose drop below freestream plane [m]           (default 0)
    r_LE            : leading-edge radius [m]                        (default 5e-3)
    r_nose          : nose radius [m]                                (default 10e-3)
    theta_upper     : upper-surface inclination [rad]                (default 0)
    gamma           : specific heat ratio                            (default 1.4)

    The geometry is constrained by:
      - 0 < theta_fore < theta_max(M_design)  (attached oblique shock)
      - M_design * cos(Lambda) > 1            (LE attachment)
      - b_base <= b_LE_fore                   (centerbody tapers inward)
      - 0 <= h_fore_nose <= 0.3 * L_fore
    """

    M_design: float = 15.0
    theta_fore: float = math.radians(8.0)
    Lambda: float = math.radians(75.0)
    L_fore: float = 2.5
    L_center: float = 1.5
    b_base: float = 0.5
    h_base: float = 0.4
    h_fore_nose: float = 0.0
    r_LE: float = 5e-3
    r_nose: float = 10e-3
    theta_upper: float = 0.0
    gamma: float = 1.4

    # Populated by build()
    mesh: Optional[Mesh] = field(default=None, init=False, repr=False)
    beta_design: float = field(default=0.0, init=False)
    b_LE_fore: float = field(default=0.0, init=False)
    L_total: float = field(default=0.0, init=False)
    warnings: list = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self._validate_inputs()
        self._build()

    def _validate_inputs(self) -> None:
        # Geometric bounds
        if not (0.0 < self.Lambda < math.pi / 2):
            raise ValueError("Lambda must be in (0, pi/2)")
        if self.theta_fore <= 0.0:
            raise ValueError("theta_fore must be positive")
        if self.L_fore <= 0.0 or self.L_center < 0.0:
            raise ValueError("L_fore must be > 0 and L_center >= 0")
        if self.h_fore_nose < 0 or self.h_fore_nose > 0.3 * self.L_fore:
            raise ValueError(
                f"h_fore_nose must be in [0, 0.3 * L_fore]; "
                f"got {self.h_fore_nose} (max {0.3 * self.L_fore})"
            )
        # Attached-shock checks
        try:
            self.beta_design = obtain_beta(self.theta_fore, self.M_design,
                                            self.gamma)
        except ShockDetachedError as e:
            raise ShockDetachedError(
                f"forebody shock detached: M={self.M_design}, "
                f"theta_fore={math.degrees(self.theta_fore):.2f} deg: {e}")
        # LE attachment
        M_perp = self.M_design * math.cos(self.Lambda)
        if M_perp <= 1.0:
            raise ShockDetachedError(
                f"LE detached: M cos(Lambda) = {M_perp:.3f} <= 1")
        # Inward-taper check
        self.b_LE_fore = self.L_fore / math.tan(self.Lambda)
        if self.b_base > self.b_LE_fore + 1e-9:
            raise ValueError(
                f"b_base ({self.b_base:.4f}) > b_LE_fore "
                f"({self.b_LE_fore:.4f}); centerbody can taper inward only"
            )

    def _build(self) -> None:
        Lf = self.L_fore
        Lc = self.L_center
        Lt = Lf + Lc
        self.L_total = Lt
        b_f = self.b_LE_fore
        b_b = self.b_base
        h_b = self.h_base
        tan_th = math.tan(self.theta_fore)
        tan_thu = math.tan(self.theta_upper)
        z_apex = -self.h_fore_nose

        # Forebody-TE z values (upper at theta_upper plane, lower at
        # theta_fore plane, both through apex)
        z_F_upper = z_apex - Lf * tan_thu
        z_F_lower = z_apex - Lf * tan_th
        # Base z-position: base upper edge continues the upper-surface
        # plane; base height = h_base downward from there.
        z_B_upper = z_apex - Lt * tan_thu
        z_B_lower = z_B_upper - h_b

        # Vertices (9 total)
        A    = np.array([0.0, 0.0, z_apex])
        F_UR = np.array([Lf, +b_f, z_F_upper])
        F_UL = np.array([Lf, -b_f, z_F_upper])
        F_LR = np.array([Lf, +b_f, z_F_lower])
        F_LL = np.array([Lf, -b_f, z_F_lower])
        B_UR = np.array([Lt, +b_b, z_B_upper])
        B_UL = np.array([Lt, -b_b, z_B_upper])
        B_LR = np.array([Lt, +b_b, z_B_lower])
        B_LL = np.array([Lt, -b_b, z_B_lower])

        vertices = np.array([A, F_UR, F_UL, F_LR, F_LL,
                              B_UR, B_UL, B_LR, B_LL])
        # Indices (for readability)
        iA  = 0
        iFUR, iFUL, iFLR, iFLL = 1, 2, 3, 4
        iBUR, iBUL, iBLR, iBLL = 5, 6, 7, 8

        # Faces (CCW from outside body for outward normals; verified by
        # signed-volume test). 14 triangles.
        faces = []
        labels = []

        # ---- Forebody (4 triangles) ------------------------------
        # Upper: A -> F_UL -> F_UR  (normal +z)
        faces.append([iA, iFUL, iFUR]); labels.append("forebody_upper")
        # Lower: A -> F_LR -> F_LL  (normal -z)
        faces.append([iA, iFLR, iFLL]); labels.append("forebody_lower")
        # Side right: A -> F_UR -> F_LR  (normal +y)
        faces.append([iA, iFUR, iFLR]); labels.append("forebody_side_right")
        # Side left: A -> F_LL -> F_UL  (normal -y)
        faces.append([iA, iFLL, iFUL]); labels.append("forebody_side_left")

        # ---- Centerbody (8 triangles) ---------------------------
        # Upper quad: F_UR -> F_UL -> B_UL -> B_UR (CCW from above)
        # Triangulate as F_UR-F_UL-B_UL and F_UR-B_UL-B_UR
        faces.append([iFUR, iFUL, iBUL]); labels.append("centerbody_upper")
        faces.append([iFUR, iBUL, iBUR]); labels.append("centerbody_upper")
        # Lower quad: F_LR -> B_LR -> B_LL -> F_LL (CCW from below)
        faces.append([iFLR, iBLR, iBLL]); labels.append("centerbody_lower")
        faces.append([iFLR, iBLL, iFLL]); labels.append("centerbody_lower")
        # Side right (y = +b): F_UR -> B_UR -> B_LR -> F_LR (CCW from +y)
        faces.append([iFUR, iBUR, iBLR]); labels.append("centerbody_side_right")
        faces.append([iFUR, iBLR, iFLR]); labels.append("centerbody_side_right")
        # Side left (y = -b): F_UL -> F_LL -> B_LL -> B_UL (CCW from -y)
        faces.append([iFUL, iFLL, iBLL]); labels.append("centerbody_side_left")
        faces.append([iFUL, iBLL, iBUL]); labels.append("centerbody_side_left")

        # ---- Base (2 triangles) ----------------------------------
        # Rectangle viewed from +x (outward direction): UR -> UL -> LL -> LR
        # is CCW. Triangulate as (UR, UL, LL) + (UR, LL, LR).
        faces.append([iBUR, iBUL, iBLL]); labels.append("base")
        faces.append([iBUR, iBLL, iBLR]); labels.append("base")

        self.mesh = Mesh(
            vertices=vertices,
            faces=np.array(faces),
            labels=np.array(labels, dtype=object),
            metadata={
                "kind": "engineering_flat",
                "M_design": self.M_design,
                "theta_fore_deg": math.degrees(self.theta_fore),
                "Lambda_deg": math.degrees(self.Lambda),
                "L_fore": self.L_fore,
                "L_center": self.L_center,
                "b_base": self.b_base,
                "h_base": self.h_base,
                "b_LE_fore": self.b_LE_fore,
                "L_total": self.L_total,
                "beta_design_deg": math.degrees(self.beta_design),
                "r_LE": self.r_LE,
                "r_nose": self.r_nose,
                "theta_upper_deg": math.degrees(self.theta_upper),
                "h_fore_nose": self.h_fore_nose,
            },
        )

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def closed_form_volume(self) -> float:
        """Closed-form volume (forebody + centerbody + base; theta_upper=0
        case). Used for cross-checking the numerical mesh volume."""
        Lf = self.L_fore; Lc = self.L_center
        b_f = self.b_LE_fore; b_b = self.b_base
        tan_th = math.tan(self.theta_fore)
        tan_thu = math.tan(self.theta_upper)
        # Forebody: triangular wedge with apex at origin, planform area
        # b_f * Lf, thickness at TE (Lf tan(theta_fore) - Lf tan(theta_upper)).
        # V = (1/3) * b_f * Lf * thickness ... actually for a wedge swept
        # by Lambda with cross-section growing as ξ tan(theta_fore):
        # V_fore = (2/3) Lf^3 (tan(theta_fore) - tan(theta_upper)) / tan(Lambda).
        V_fore = ((2.0/3.0) * Lf**3
                  * (tan_th - tan_thu) / math.tan(self.Lambda))
        # Centerbody: linearly-interpolated frustum between two rectangles
        # (forebody TE rectangle and base rectangle).
        h_F_TE = Lf * (tan_th - tan_thu)         # forebody TE thickness
        A_fore_TE = 2.0 * b_f * h_F_TE           # rectangle area
        A_base = 2.0 * b_b * self.h_base
        V_center = 0.5 * (A_fore_TE + A_base) * Lc
        return V_fore + V_center

    def closed_form_planform_area(self) -> float:
        """Closed-form S_planform: forebody triangle + centerbody trapezoid."""
        S_fore = self.L_fore * self.b_LE_fore
        S_center = 0.5 * (2.0 * self.b_LE_fore + 2.0 * self.b_base) * self.L_center
        return S_fore + S_center

    @property
    def Lambda_deg(self) -> float:
        return math.degrees(self.Lambda)

    @property
    def theta_fore_deg(self) -> float:
        return math.degrees(self.theta_fore)

    @property
    def beta_design_deg(self) -> float:
        return math.degrees(self.beta_design)
