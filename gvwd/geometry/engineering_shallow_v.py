"""Engineering glide-vehicle (shallow-V) variant (GVWD §4.5).

Identical to the flat-bottom mode except the lower surface is a shallow V
with a prescribed dihedral angle. When ``dihedral_lower = 0`` this reduces
exactly to :class:`EngineeringFlat`. When dihedral_lower approaches the
value at which the V trough reaches the LE plane, this approaches a full
Nonweiler caret with a finite base.

Geometric construction:

The lower surface centerline (y = 0) drops further below the flat-bottom
case by the trough depth:
    d_trough(x) = (b_y_at_x) * tan(dihedral_lower)
where ``b_y_at_x`` is the local half-width of the lower surface at x.
The lower surface is two flat panels meeting at a V-trough centerline
that runs apex -> base centerline at increasing depth.

The forebody therefore has 6 vertices instead of 5 (apex + 4 TE corners +
1 TE V-trough vertex), and the centerbody propagates the trough through
its lower surface to a base trough vertex.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from gvwd.geometry.mesh import Mesh
from gvwd.thermo.oblique_shock import obtain_beta, ShockDetachedError


@dataclass
class EngineeringShallowV:
    """Shallow-V variant of the engineering flat-bottom forebody."""

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
    dihedral_lower: float = math.radians(5.0)   # NEW vs EngineeringFlat
    gamma: float = 1.4

    mesh: Optional[Mesh] = field(default=None, init=False, repr=False)
    beta_design: float = field(default=0.0, init=False)
    b_LE_fore: float = field(default=0.0, init=False)
    L_total: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if not (0.0 < self.Lambda < math.pi / 2):
            raise ValueError("Lambda must be in (0, pi/2)")
        if self.theta_fore <= 0.0:
            raise ValueError("theta_fore must be positive")
        if not (0.0 <= self.dihedral_lower < math.radians(45.0)):
            raise ValueError(
                f"dihedral_lower must be in [0, 45) deg; got "
                f"{math.degrees(self.dihedral_lower):.2f}"
            )
        try:
            self.beta_design = obtain_beta(self.theta_fore, self.M_design,
                                            self.gamma)
        except ShockDetachedError as e:
            raise ShockDetachedError(f"forebody shock detached: {e}")
        if self.M_design * math.cos(self.Lambda) <= 1.0:
            raise ShockDetachedError("LE detached: M cos(Lambda) <= 1")
        self.b_LE_fore = self.L_fore / math.tan(self.Lambda)
        if self.b_base > self.b_LE_fore + 1e-9:
            raise ValueError("b_base must be <= b_LE_fore")
        self._build()

    def _build(self) -> None:
        Lf = self.L_fore; Lc = self.L_center
        Lt = Lf + Lc; self.L_total = Lt
        b_f = self.b_LE_fore; b_b = self.b_base; h_b = self.h_base
        tan_th = math.tan(self.theta_fore)
        tan_thu = math.tan(self.theta_upper)
        tan_dh = math.tan(self.dihedral_lower)
        z_apex = -self.h_fore_nose

        # Z values at forebody TE
        z_F_upper = z_apex - Lf * tan_thu
        z_F_lower_LE = z_apex - Lf * tan_th                  # at wingtip y = b_f
        # Trough drops further below by b_f * tan(dihedral)
        z_F_trough = z_F_lower_LE - b_f * tan_dh

        # Z values at base
        z_B_upper = z_apex - Lt * tan_thu
        z_B_lower_LE = z_B_upper - h_b
        z_B_trough = z_B_lower_LE - b_b * tan_dh

        # Vertices (11 total: apex + 5 forebody TE + 5 base)
        A = np.array([0.0, 0.0, z_apex])
        F_UR = np.array([Lf, +b_f, z_F_upper])
        F_UL = np.array([Lf, -b_f, z_F_upper])
        F_LR = np.array([Lf, +b_f, z_F_lower_LE])
        F_LL = np.array([Lf, -b_f, z_F_lower_LE])
        F_TR = np.array([Lf, 0.0, z_F_trough])         # forebody-TE trough
        B_UR = np.array([Lt, +b_b, z_B_upper])
        B_UL = np.array([Lt, -b_b, z_B_upper])
        B_LR = np.array([Lt, +b_b, z_B_lower_LE])
        B_LL = np.array([Lt, -b_b, z_B_lower_LE])
        B_TR = np.array([Lt, 0.0, z_B_trough])         # base trough

        vertices = np.array([A, F_UR, F_UL, F_LR, F_LL, F_TR,
                              B_UR, B_UL, B_LR, B_LL, B_TR])
        iA = 0
        iFUR, iFUL, iFLR, iFLL, iFTR = 1, 2, 3, 4, 5
        iBUR, iBUL, iBLR, iBLL, iBTR = 6, 7, 8, 9, 10

        faces = []
        labels = []

        # ---- Forebody (5 triangles: 1 upper, 2 lower-V, 2 sides) ----
        faces.append([iA, iFUL, iFUR]); labels.append("forebody_upper")
        # Lower V: two panels meeting at the centerline trough
        # Right panel: A -> F_LR -> F_TR  (normal points -z and -y)
        faces.append([iA, iFLR, iFTR]); labels.append("forebody_lower_right")
        # Left panel: A -> F_TR -> F_LL  (normal points -z and +y)
        faces.append([iA, iFTR, iFLL]); labels.append("forebody_lower_left")
        # Side panels (apex -> upper-tip -> lower-tip)
        faces.append([iA, iFUR, iFLR]); labels.append("forebody_side_right")
        faces.append([iA, iFLL, iFUL]); labels.append("forebody_side_left")

        # ---- Centerbody (10 triangles) ------------------------------
        # Upper quad (outward +z)
        faces.append([iFUR, iFUL, iBUL]); labels.append("centerbody_upper")
        faces.append([iFUR, iBUL, iBUR]); labels.append("centerbody_upper")
        # Lower right V-panel (outward roughly -z, +y): swap last two
        # vertices vs naive CCW so the outward normal points down/right.
        faces.append([iFLR, iBTR, iFTR]); labels.append("centerbody_lower_right")
        faces.append([iFLR, iBLR, iBTR]); labels.append("centerbody_lower_right")
        # Lower left V-panel (outward -z, -y)
        faces.append([iFTR, iBTR, iFLL]); labels.append("centerbody_lower_left")
        faces.append([iFLL, iBTR, iBLL]); labels.append("centerbody_lower_left")
        # Side right (outward +y)
        faces.append([iFUR, iBUR, iBLR]); labels.append("centerbody_side_right")
        faces.append([iFUR, iBLR, iFLR]); labels.append("centerbody_side_right")
        # Side left (outward -y)
        faces.append([iFUL, iFLL, iBLL]); labels.append("centerbody_side_left")
        faces.append([iFUL, iBLL, iBUL]); labels.append("centerbody_side_left")

        # ---- Base (3 triangles, pentagonal cross-section) -----------
        # Pentagon UR-UL-LL-TR-LR viewed from +x (outside): going CCW
        # we visit LR -> UR -> UL -> LL -> TR. Fan from B_TR with the
        # remaining 4 vertices in CCW order from +x:
        #   B_TR -> B_LR -> B_UR
        #   B_TR -> B_UR -> B_UL
        #   B_TR -> B_UL -> B_LL
        # giving outward (+x) normals on each.
        faces.append([iBTR, iBLR, iBUR]); labels.append("base")
        faces.append([iBTR, iBUR, iBUL]); labels.append("base")
        faces.append([iBTR, iBUL, iBLL]); labels.append("base")

        self.mesh = Mesh(
            vertices=vertices, faces=np.array(faces),
            labels=np.array(labels, dtype=object),
            metadata={
                "kind": "engineering_shallow_v",
                "M_design": self.M_design,
                "theta_fore_deg": math.degrees(self.theta_fore),
                "Lambda_deg": math.degrees(self.Lambda),
                "L_fore": self.L_fore, "L_center": self.L_center,
                "b_base": self.b_base, "h_base": self.h_base,
                "dihedral_lower_deg": math.degrees(self.dihedral_lower),
                "b_LE_fore": self.b_LE_fore,
                "L_total": self.L_total,
                "beta_design_deg": math.degrees(self.beta_design),
            },
        )

    @property
    def dihedral_lower_deg(self) -> float:
        return math.degrees(self.dihedral_lower)

    @property
    def theta_fore_deg(self) -> float:
        return math.degrees(self.theta_fore)
