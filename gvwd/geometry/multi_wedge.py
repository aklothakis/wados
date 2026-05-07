"""Oswatitsch n-ramp multi-wedge reference geometry (GVWD §4.6).

The 2.5-D multi-wedge forebody: an n-ramp lower compression surface in
the (x, z) plane, extruded laterally either as a constant-spanwise
profile (rectangular extrusion) or as a swept-delta (convergent y span).
The ramp angles come from the Phase-1 ``equal_strength_ramps`` solver.

Inputs:
    M_design       : design Mach
    n              : number of ramps
    delta_total_deg: cumulative deflection [deg] (default 20)
    L              : body length [m]
    half_span      : wingtip half-span [m] (rectangular mode) or
                     wingtip half-span at base (delta mode)
    extrusion      : 'rectangular' or 'delta'
    height         : prism height [m] (rectangular mode upper-surface offset)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from gvwd.geometry.mesh import Mesh
from gvwd.thermo.oswatitsch import equal_strength_ramps, OswatitschResult


@dataclass
class MultiWedge:
    """Oswatitsch n-ramp 2.5-D multi-wedge forebody.

    Lower-surface profile in the (x, z) plane is a sequence of straight
    line segments at the cumulative ramp angles delta_1, delta_2, ...,
    delta_n. Ramp lengths are chosen so the overall body length equals L.
    By default the n ramps have equal streamwise length L/n; can be
    customised via ``ramp_lengths_frac``.
    """

    M_design: float
    n: int = 2
    delta_total_deg: float = 20.0
    L: float = 8.0
    half_span: float = 1.0
    extrusion: str = "rectangular"
    height: float = 0.6                    # prism height (upper-z offset)
    ramp_lengths_frac: Optional[List[float]] = None
    gamma: float = 1.4

    osw: Optional[OswatitschResult] = field(default=None, init=False, repr=False)
    mesh: Optional[Mesh] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("n must be >= 1")
        if self.extrusion not in ("rectangular", "delta"):
            raise ValueError(
                f"extrusion must be 'rectangular' or 'delta', "
                f"got {self.extrusion!r}")
        # Solve for the equal-strength ramp angles
        self.osw = equal_strength_ramps(
            M_inf=self.M_design,
            n=self.n,
            gamma=self.gamma,
            delta_total_deg=self.delta_total_deg,
        )
        # Default to equal streamwise lengths
        if self.ramp_lengths_frac is None:
            self.ramp_lengths_frac = [1.0 / self.n] * self.n
        if abs(sum(self.ramp_lengths_frac) - 1.0) > 1e-9:
            raise ValueError("ramp_lengths_frac must sum to 1")
        self._build()

    def _ramp_profile(self) -> np.ndarray:
        """Return (n+1, 2) array of (x, z) profile vertices on the lower
        surface, starting at the apex (0, 0) and stepping through each
        ramp at its cumulative angle delta_i."""
        pts = [(0.0, 0.0)]
        x_running = 0.0
        for i in range(self.n):
            dx = self.L * self.ramp_lengths_frac[i]
            delta_i = math.radians(self.osw.deltas_cum_deg[i])
            x_new = x_running + dx
            # z on the i-th ramp uses the cumulative deflection from freestream
            # measured from the previous ramp endpoint
            x_prev, z_prev = pts[-1]
            z_new = z_prev - dx * math.tan(delta_i)
            pts.append((x_new, z_new))
            x_running = x_new
        return np.array(pts)

    def _build(self) -> None:
        prof = self._ramp_profile()         # (n+1, 2) x-z lower-surface points

        if self.extrusion == "rectangular":
            self._build_rectangular(prof)
        else:
            self._build_delta(prof)

    def _build_rectangular(self, prof: np.ndarray) -> None:
        """Rectangular prismatic extrusion: lower-surface profile is
        replicated at y=+half_span and y=-half_span; upper surface is a
        flat rectangular plane at z = height (above the apex)."""
        b = self.half_span
        h_top = self.height
        n_seg = len(prof) - 1   # = n ramps
        # Vertex layout:
        #   bottom right edge: prof at y=+b   (n+1 verts)
        #   bottom left edge:  prof at y=-b   (n+1 verts)
        #   top right edge:    (x, +b, h_top)  for x at start and end of body (2 verts)
        #   top left edge:     (x, -b, h_top)  (2 verts)
        verts = []
        for x, z in prof:
            verts.append([x, +b, z])    # bottom right
            verts.append([x, -b, z])    # bottom left
        # Top corners
        x0, x1 = prof[0, 0], prof[-1, 0]
        verts += [
            [x0, +b, h_top], [x0, -b, h_top],
            [x1, +b, h_top], [x1, -b, h_top],
        ]
        verts = np.array(verts)
        # Index helpers
        BR = lambda i: 2*i + 0     # bottom right at profile node i
        BL = lambda i: 2*i + 1     # bottom left  at profile node i
        TR0 = 2*(n_seg+1) + 0
        TL0 = 2*(n_seg+1) + 1
        TR1 = 2*(n_seg+1) + 2
        TL1 = 2*(n_seg+1) + 3

        faces = []
        labels = []
        # Bottom surface: n_seg quads, each split into 2 triangles
        for i in range(n_seg):
            # Quad: BR(i)  BR(i+1)  BL(i+1)  BL(i)   (viewed from below)
            faces.append([BR(i), BR(i+1), BL(i+1)])
            faces.append([BR(i), BL(i+1), BL(i)])
            labels.append(f"lower_ramp_{i+1}")
            labels.append(f"lower_ramp_{i+1}")

        # Side panels (right y=+b): connect bottom profile to top edge
        # Right side is a polygon with vertices TR0, TR1, BR(n_seg), ..., BR(0).
        # Triangulate as a fan from TR0.
        for i in range(n_seg):
            # Quad TR0  BR(i)  BR(i+1)  TR1 (for the rear-most)... easier:
            # Triangulate the right side as fan from TR0 and TR1.
            faces.append([TR0, BR(i+1), BR(i)])    # top wedge if i=0 is special
            labels.append("side_right")
        # The polygon TR0 -> TR1 -> BR(n_seg) needs one more triangle:
        faces.append([TR0, TR1, BR(n_seg)])
        labels.append("side_right")

        # Side panels (left y=-b): mirror
        for i in range(n_seg):
            faces.append([TL0, BL(i), BL(i+1)])
            labels.append("side_left")
        faces.append([TL0, BL(n_seg), TL1])
        labels.append("side_left")

        # Top surface (rectangular): TR0-TR1-TL1-TL0 -> 2 triangles
        faces.append([TR0, TR1, TL1])
        faces.append([TR0, TL1, TL0])
        labels.append("upper"); labels.append("upper")

        # Front face: triangle BR(0) - BL(0) - TL0 - TR0 (zero-area unless
        # they aren't coincident; here apex is at z=0 so x0=0). Quad split:
        faces.append([BR(0), TR0, TL0])
        faces.append([BR(0), TL0, BL(0)])
        labels.append("nose"); labels.append("nose")

        # Base (rear, x=x1): polygon BR(n) - BL(n) - TL1 - TR1
        faces.append([BR(n_seg), TR1, TL1])
        faces.append([BR(n_seg), TL1, BL(n_seg)])
        labels.append("base"); labels.append("base")

        self.mesh = Mesh(
            vertices=verts,
            faces=np.array(faces),
            labels=np.array(labels, dtype=object),
            metadata={
                "kind": "multi_wedge_rectangular",
                "M_design": self.M_design,
                "n": self.n,
                "deltas_cum_deg": list(self.osw.deltas_cum_deg),
                "L": self.L,
                "half_span": self.half_span,
                "height": self.height,
            },
        )

    def _build_delta(self, prof: np.ndarray) -> None:
        """Swept-delta extrusion: the ramp profile is the centerline; the
        wingtips taper from y=0 at the apex to y=+/-half_span at the base.
        Upper surface is a single plane through (apex, wingtip+, wingtip-)."""
        b = self.half_span
        n_seg = len(prof) - 1
        # Linear interpolation of half-span along the body length:
        spans = np.linspace(0.0, b, len(prof))
        # Lower surface: 2*(n+1) bottom-edge vertices (right and left).
        # Upper surface: triangle through (apex, wingtip+, wingtip-) — only
        # 3 unique vertices.
        verts = []
        for (x, z), s in zip(prof, spans):
            verts.append([x, +s, z])
            verts.append([x, -s, z])
        # Add upper-surface tip and apex (apex shared with lower)
        # Apex is verts[0] = (0, 0, 0); we override apex coords to ensure
        # span at apex is exactly 0 (numerically ok since spans[0] = 0).
        # Wingtips at base (last profile point): verts[2*n_seg] and verts[2*n_seg+1].
        # For the upper triangular plane we need a SINGLE upper-surface vertex
        # at the apex; we already have (0, 0, prof[0,1]). The two wingtips at
        # base are also on the upper plane.
        verts = np.array(verts)
        # Check spans[0] = 0 -> verts[0] == verts[1]; collapse them:
        # Actually we can keep the duplicated apex vertex at index 0 and 1
        # since both have y=0. The mesh will have a degenerate triangle if
        # we connect them, so we need to special-case the first ramp segment.

        n_v = len(verts)
        BR = lambda i: 2*i + 0
        BL = lambda i: 2*i + 1
        # Apex single vertex (we use BR(0) since spans[0]=0 makes BR(0)=BL(0))
        # All upper-surface faces use vertex 0 as the apex (whose y=0).
        # Wingtips at base: BR(n_seg), BL(n_seg).

        faces = []
        labels = []
        # Lower surface: n_seg quads -> triangles. At i=0 the quad collapses
        # to a triangle (apex point).
        for i in range(n_seg):
            if i == 0:
                # First ramp: triangle (apex, BR(1), BL(1))
                faces.append([BR(0), BR(1), BL(1)])
                labels.append(f"lower_ramp_{i+1}")
            else:
                faces.append([BR(i), BR(i+1), BL(i+1)])
                faces.append([BR(i), BL(i+1), BL(i)])
                labels.append(f"lower_ramp_{i+1}")
                labels.append(f"lower_ramp_{i+1}")

        # Upper surface: single triangle (apex, wingtip+, wingtip-)
        faces.append([BR(0), BL(n_seg), BR(n_seg)])
        labels.append("upper")

        # Base: triangle (BR(n_seg), BL(n_seg), centerline_at_base)
        # The centerline at the base is profile point n_seg, but we don't
        # have a centerline-y=0 vertex at the base unless we add one.
        # The closure is: at x=L, the cross-section is bounded by:
        #   upper edge: line from BL(n_seg) to BR(n_seg) (both on upper plane,
        #   wingtip-tips)
        #   lower edge: bottom-of-ramp profile (at y=0, follows centerline)
        # For a simple closure, treat the base as a single triangle from
        # the lower-centerline (at z = prof[-1, 1]) to the two wingtips.
        # Add the centerline vertex:
        verts2 = np.vstack([verts, [[prof[-1, 0], 0.0, prof[-1, 1]]]])
        cb_idx = len(verts)
        faces.append([cb_idx, BR(n_seg), BL(n_seg)])
        labels.append("base")
        verts = verts2

        self.mesh = Mesh(
            vertices=verts,
            faces=np.array(faces),
            labels=np.array(labels, dtype=object),
            metadata={
                "kind": "multi_wedge_delta",
                "M_design": self.M_design,
                "n": self.n,
                "deltas_cum_deg": list(self.osw.deltas_cum_deg),
                "L": self.L,
                "half_span": self.half_span,
            },
        )
