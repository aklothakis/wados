"""Nonweiler caret reference geometry (GVWD §4.6).

The caret is a V-shape vehicle with apex at the origin, two leading-edge
rays riding on the design-Mach planar shock, and an upper surface that
is a single flat triangular plane through (apex, wingtip+, wingtip-).
The lower surface is two flat triangular panels meeting at a centerline
V-trough that drops to z = -L tan(theta) at the base centerline.

Inputs (per spec §5.2 DoD):
    M_design : design Mach
    theta_d  : wedge angle (lower-surface deflection from freestream) [rad]
    Lambda   : leading-edge sweep from spanwise axis [rad]
    L        : body length (apex to base) [m]

The shock angle beta is derived from theta-beta-M (weak-shock branch).

The closed-form volume is V = (1/3) L b |z_TE,centerline| with
b = y_tip = L / tan(Lambda) and z_TE,centerline = -L tan(theta_d).

NOTE on PSWR-1 reuse: with constant beta (i.e. all three knots equal to
beta_d) and X1=0, the PSWR-1 ``VariableWedgeWaverider`` reproduces
exactly the Nonweiler caret geometry — its lower-surface ruled
construction collapses to two flat planes and the upper surface is a
single triangular plane (verified analytically: see PSWR-1 phase notes
for the coplanarity proof). We thin-wrap that geometry here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from gvwd.geometry.mesh import Mesh
from gvwd.thermo.oblique_shock import obtain_beta


@dataclass
class Caret:
    """Nonweiler caret reference geometry.

    Construction flag: ``input_kind = 'theta'`` (default — user supplies
    wedge angle) or ``'beta'`` (user supplies the shock angle directly).
    """

    M_design: float
    Lambda: float                      # rad
    L: float = 10.0                    # m
    theta_d: Optional[float] = None    # rad, wedge angle
    beta_d: Optional[float] = None     # rad, shock angle (alternative input)
    gamma: float = 1.4
    input_kind: str = "theta"

    # Populated by build()
    mesh: Optional[Mesh] = field(default=None, init=False, repr=False)
    y_tip: float = field(default=0.0, init=False)
    z_TE_centerline: float = field(default=0.0, init=False)
    z_LE_tip: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        # Resolve theta_d / beta_d inputs
        if self.input_kind == "theta":
            if self.theta_d is None:
                raise ValueError("theta_d required when input_kind='theta'")
            self.beta_d = obtain_beta(self.theta_d, self.M_design, self.gamma)
        elif self.input_kind == "beta":
            if self.beta_d is None:
                raise ValueError("beta_d required when input_kind='beta'")
            from gvwd.thermo.oblique_shock import theta_from_beta_M
            self.theta_d = theta_from_beta_M(self.beta_d, self.M_design,
                                              self.gamma)
        else:
            raise ValueError(f"unknown input_kind {self.input_kind!r}")

        if not (0.0 < self.Lambda < math.pi / 2):
            raise ValueError(
                f"Lambda must be in (0, pi/2); got "
                f"{math.degrees(self.Lambda):.2f} deg")

        self._build()

    def _build(self) -> None:
        tan_L = math.tan(self.Lambda)
        sin_L = math.sin(self.Lambda)
        tan_th = math.tan(self.theta_d)
        tan_b = math.tan(self.beta_d)

        self.y_tip = self.L / tan_L
        self.z_LE_tip = -self.y_tip * tan_b / sin_L      # at LE wingtip
        self.z_TE_centerline = -self.L * tan_th          # at base centerline

        # Caret topology: TWO upper panels meeting at a centerline ridge
        # (apex -> base-centerline-upper) and TWO lower panels meeting at
        # a centerline V-trough (apex -> base-centerline-lower). The base
        # cross-section is a 4-corner diamond (Wp, UB, Wm, LB) split into
        # two triangles.
        #
        # 5 vertices, 6 faces.
        A  = np.array([0.0, 0.0, 0.0])                          # apex
        Wp = np.array([self.L, +self.y_tip, self.z_LE_tip])     # wingtip+
        Wm = np.array([self.L, -self.y_tip, self.z_LE_tip])     # wingtip-
        UB = np.array([self.L, 0.0, 0.0])                       # base centerline upper (ridge)
        LB = np.array([self.L, 0.0, self.z_TE_centerline])      # base centerline lower (V-trough)

        vertices = np.array([A, Wp, Wm, UB, LB])
        # Faces (CCW from outside the body):
        #   upper_right: A - Wp - UB     (normal +z, +x)
        #   upper_left : A - UB - Wm     (normal +z, +x)
        #   lower_right: A - LB - Wp     (normal -z, -x)
        #   lower_left : A - Wm - LB     (normal -z, -x)
        #   base_upper : UB - Wp - LB    (rear-facing diamond top half, normal +x)
        #     (alternatively split as UB-Wp-LB and Wp-Wm-LB; we choose
        #     the symmetric pair below)
        #   base_lower : UB - LB - Wm
        # Face winding chosen so the divergence-theorem signed volume is
        # POSITIVE (outward unit normals). Verified by the
        # test_caret_signed_volume_outward_normals test.
        faces = np.array([
            [0, 3, 1],   # upper right (A, UB, Wp)  — outward normal up
            [0, 2, 3],   # upper left  (A, Wm, UB)
            [0, 1, 4],   # lower right (A, Wp, LB)  — outward normal down
            [0, 4, 2],   # lower left  (A, LB, Wm)
            [3, 4, 1],   # base right  (UB, LB, Wp) — outward normal +x
            [3, 2, 4],   # base left   (UB, Wm, LB)
        ])
        labels = np.array([
            "upper_right", "upper_left",
            "lower_right", "lower_left",
            "base_right", "base_left",
        ], dtype=object)
        self.mesh = Mesh(vertices, faces, labels, metadata={
            "kind": "caret",
            "M_design": self.M_design,
            "theta_d_deg": math.degrees(self.theta_d),
            "beta_d_deg": math.degrees(self.beta_d),
            "Lambda_deg": math.degrees(self.Lambda),
            "L": self.L,
            "y_tip": self.y_tip,
        })

    # ------------------------------------------------------------------
    #  Convenience
    # ------------------------------------------------------------------

    @property
    def beta_d_deg(self) -> float:
        return math.degrees(self.beta_d)

    @property
    def theta_d_deg(self) -> float:
        return math.degrees(self.theta_d)

    @property
    def Lambda_deg(self) -> float:
        return math.degrees(self.Lambda)
