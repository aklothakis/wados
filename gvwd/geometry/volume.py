"""Geometric quantities for GVWD reference modes (§4.6).

Closed-form analytic volumes for caret and flat-delta; numerical fallback
via the Mesh divergence-theorem integral for any closed mesh.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from .mesh import Mesh, mesh_volume_signed
from .caret import Caret
from .flat_delta import FlatDelta


def caret_analytic_volume(caret: Caret) -> float:
    """V = (1/3) L b |z_TE,centerline| (spec §4.6).

    Equivalent to the closed-form integral
    V = tan(theta_d) * L^3 / (3 tan(Lambda)),
    which is what the lower-surface ruled construction gives.
    """
    return float(math.tan(caret.theta_d) * caret.L**3
                  / (3.0 * math.tan(caret.Lambda)))


def flat_delta_analytic_volume(fd: FlatDelta) -> float:
    """Volume of the flat-bottomed delta wedge.

    Geometry is bounded by:
      - upper triangular plane at z=0, planform vertices (0,0), (L, +b), (L,-b)
      - lower y>0 panel: vertices (0,0,0), (L, +b, 0), (L, 0, -L tan theta)
      - lower y<0 panel: vertices (0,0,0), (L, 0, -L tan theta), (L, -b, 0)
      - base triangle:   (L, 0, -L tan theta), (L, +b, 0), (L, -b, 0)

    The body is a tetrahedron-like prism with apex at the origin and a
    triangular base. By the same y-quadrature as the caret, V is the
    integral of (z_upper - z_lower) over the planform:

        z_upper - z_lower(x, y) = -z_lower(x, y) since z_upper = 0
        z_lower depends on y because the lower surface is two flat panels
        joined at the centerline. At spanwise position y, the lower panel
        passes through (0,0,0) and (L, +b, 0) and (L, 0, -L tan theta);
        a point on this plane satisfies
            n . (r - r0) = 0  with normal n perpendicular to two edge vectors.

    Equivalent closed-form:
        V = (1/6) * L * b * (L tan theta) for the full body (sum of two
        tetrahedral half-volumes joined at the centerline trough).
    """
    return float((1.0 / 6.0) * fd.L * fd.y_tip * (fd.L * math.tan(fd.theta_d))
                 * 2.0)   # times 2 for both halves


def numerical_volume(mesh: Mesh) -> float:
    """Numerical volume by divergence theorem on a closed mesh."""
    return abs(mesh_volume_signed(mesh))


def planform_area_from_mesh(mesh: Mesh) -> float:
    """Project each face onto the z=0 plane and sum signed areas;
    absolute value is the planform area."""
    v = mesh.vertices[mesh.faces]   # (M, 3, 3)
    a, b, c = v[:, 0, :2], v[:, 1, :2], v[:, 2, :2]   # x-y projections
    e1 = b - a; e2 = c - a
    cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    # The planform is the SHADOW of the body on z=0 — sum of upward-facing
    # triangles' area equals the planform area, but here we just take half
    # the absolute sum which counts each surface element twice (top + bottom).
    return float(0.5 * np.abs(cross).sum() / 2.0)


def eta_V(V: float, S_planform: float) -> float:
    """eta_V = V^(2/3) / S_planform."""
    if S_planform <= 0.0 or V <= 0.0:
        return 0.0
    return V ** (2.0 / 3.0) / S_planform
