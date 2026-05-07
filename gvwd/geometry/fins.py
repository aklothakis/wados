"""Aft-fin generator for engineering glide-vehicle modes (GVWD §4.7).

Diamond (double-wedge) airfoil cross-section with parameterized
max-thickness location. Each fin is a swept-tapered planform attached
to the centerbody at a prescribed root-LE chordwise location.

Fin coordinate frame (local to fin):
  - x_f: chordwise (root LE at x_f = 0, root TE at x_f = c_root)
  - y_f: spanwise (from root y_f = 0 to tip y_f = b_fin)
  - z_f: thickness direction

The local fin geometry is rotated and translated into the body frame
according to the attachment point on the centerbody and the fin
dihedral angle (for 4-fin layouts).

Topology per fin (sharp LE, diamond cross-section):
  - 4 vertices per spanwise station x 2 stations (root, tip) = 8 verts
    minus 1 each at LE and TE (which are line edges, not 4 verts)
  - Actually: at each of (root, tip), the diamond cross-section has
    4 corners (LE, TE, upper, lower). At root: 4 verts. At tip: 4 verts.
    But LE and TE are sharp edges, so LE_root and LE_tip are points and
    likewise for TE. -> 8 verts total per fin.
  - 12 triangles per fin: 4 surface quads (2 upper, 2 lower) split into
    8 triangles + root cap (4 triangles) + tip cap (4 triangles) ...
    actually for a sharp-LE diamond, the cross-section is a 4-vertex
    diamond, root is a 4-vert diamond, tip is a 4-vert diamond, and the
    side surfaces are 4 quads (LE-upper, LE-lower, TE-upper, TE-lower
    for both root and tip). Closed mesh = 4 side quads (8 tri) + 1 root
    diamond cap (2 tri) + 1 tip diamond cap (2 tri) = 12 triangles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from gvwd.geometry.mesh import Mesh


@dataclass
class FinParams:
    """Geometric parameters for an aft fin.

    Defaults match spec §4.7 for a 4-fin / X-tail at 45 deg dihedral.
    """

    n_fins: int = 4                        # 0 (off), 2 (vertical pair), 4 (X / +)
    root_chord: float = 0.3
    tip_chord: float = 0.1
    span: float = 0.4
    sweep_LE: float = math.radians(45.0)   # LE sweep [rad]
    dihedral: float = math.radians(45.0)   # 4-fin only; 0 = +-tail, 45 = X
    t_c: float = 0.05                      # thickness/chord ratio
    max_thickness_loc: float = 0.5         # x_t/c
    LE_style: str = "sharp"                # 'sharp' or 'blunt_cylinder'
    LE_radius: float = 1e-3                # used if LE_style='blunt_cylinder'
    attach_x_frac: float = 0.5             # fraction of L_center

    def __post_init__(self) -> None:
        if self.n_fins not in (0, 2, 4):
            raise ValueError(f"n_fins must be 0, 2 or 4; got {self.n_fins}")
        if not (0.02 <= self.t_c <= 0.10):
            raise ValueError(
                f"t_c must be in [0.02, 0.10]; got {self.t_c}"
            )
        if not (0.3 <= self.max_thickness_loc <= 0.7):
            raise ValueError(
                f"max_thickness_loc must be in [0.3, 0.7]; got "
                f"{self.max_thickness_loc}"
            )
        if self.tip_chord > self.root_chord:
            raise ValueError("tip_chord must be <= root_chord")
        if self.LE_style not in ("sharp", "blunt_cylinder"):
            raise ValueError(
                f"LE_style must be 'sharp' or 'blunt_cylinder'; got "
                f"{self.LE_style}")


def _fin_local_mesh(params: FinParams) -> tuple:
    """Return (vertices, faces, labels) of a single fin in its LOCAL
    frame (root LE at origin, span +y_f, chord +x_f, thickness +-z_f).

    The diamond cross-section at a chord c has:
        LE at (0, 0)
        TE at (c, 0)
        upper at (x_t, +t/2)  with t = t_c * c, x_t = max_thickness_loc * c
        lower at (x_t, -t/2)

    Root cross-section (y_f = 0) uses c = root_chord.
    Tip cross-section (y_f = span) uses c = tip_chord, but its LE is
    swept back by ``span * tan(sweep_LE)`` in the chordwise direction.

    Returns 8 vertices, 12 triangles for a sharp-LE diamond.
    """
    c_r = params.root_chord
    c_t = params.tip_chord
    b = params.span
    x_t_r = params.max_thickness_loc * c_r
    x_t_t = params.max_thickness_loc * c_t
    t_r = params.t_c * c_r
    t_t = params.t_c * c_t
    sweep_offset = b * math.tan(params.sweep_LE)

    # 8 vertices: 4 per cross-section (LE, TE, upper, lower) at root & tip
    LE_r  = np.array([0.0,                          0.0, 0.0])
    TE_r  = np.array([c_r,                          0.0, 0.0])
    UP_r  = np.array([x_t_r,                        0.0, +t_r / 2.0])
    LO_r  = np.array([x_t_r,                        0.0, -t_r / 2.0])
    LE_t  = np.array([sweep_offset,                 b,   0.0])
    TE_t  = np.array([sweep_offset + c_t,           b,   0.0])
    UP_t  = np.array([sweep_offset + x_t_t,         b,   +t_t / 2.0])
    LO_t  = np.array([sweep_offset + x_t_t,         b,   -t_t / 2.0])

    vertices = np.array([LE_r, TE_r, UP_r, LO_r,
                          LE_t, TE_t, UP_t, LO_t])
    iLEr, iTEr, iUPr, iLOr = 0, 1, 2, 3
    iLEt, iTEt, iUPt, iLOt = 4, 5, 6, 7

    faces = []; labels = []

    # Side surfaces (4 quads, 8 tri)
    # Upper-LE wedge (front-upper): LE_r - UP_r - UP_t - LE_t
    faces.append([iLEr, iUPr, iUPt]); labels.append("fin_upper_LE")
    faces.append([iLEr, iUPt, iLEt]); labels.append("fin_upper_LE")
    # Upper-TE wedge: UP_r - TE_r - TE_t - UP_t
    faces.append([iUPr, iTEr, iTEt]); labels.append("fin_upper_TE")
    faces.append([iUPr, iTEt, iUPt]); labels.append("fin_upper_TE")
    # Lower-LE wedge: LE_r - LE_t - LO_t - LO_r
    faces.append([iLEr, iLEt, iLOt]); labels.append("fin_lower_LE")
    faces.append([iLEr, iLOt, iLOr]); labels.append("fin_lower_LE")
    # Lower-TE wedge: LO_r - LO_t - TE_t - TE_r
    faces.append([iLOr, iLOt, iTEt]); labels.append("fin_lower_TE")
    faces.append([iLOr, iTEt, iTEr]); labels.append("fin_lower_TE")

    # Root cap (diamond at y_f = 0): LE_r - UP_r - TE_r - LO_r
    # (CCW from -y_f outside)
    faces.append([iLEr, iLOr, iTEr]); labels.append("fin_root")
    faces.append([iLEr, iTEr, iUPr]); labels.append("fin_root")

    # Tip cap (diamond at y_f = span): LE_t - TE_t - UP_t / LO_t
    # (CCW from +y_f outside)
    faces.append([iLEt, iTEt, iLOt]); labels.append("fin_tip")
    faces.append([iLEt, iUPt, iTEt]); labels.append("fin_tip")

    return vertices, np.array(faces), np.array(labels, dtype=object)


def _rotate_x(theta: float) -> np.ndarray:
    """Rotation matrix about the x-axis (chordwise)."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c],
    ])


def generate_fins(params: FinParams, *,
                   attach_xyz: tuple) -> Optional[Mesh]:
    """Construct a fin assembly mesh in the body frame.

    Parameters
    ----------
    params      : FinParams
    attach_xyz  : (x, y, z) in the body frame. Each fin's LOCAL ROOT-LE is
                  placed here, then the fin is rotated into its dihedral.
                  For 2-fin (vertical pair): both fins emerge from this
                  point, one to +z and one to -z, dihedral=0 (vertical).
                  For 4-fin: four fins at +-dihedral around the body
                  centerline, two on each side (top and bottom).

    Returns
    -------
    Mesh combining all fins, or None if n_fins == 0.
    """
    if params.n_fins == 0:
        return None
    v_loc, f_loc, l_loc = _fin_local_mesh(params)

    # Build per-fin rotations
    if params.n_fins == 2:
        # Vertical pair (top + bottom)
        rotations = [
            _rotate_x(math.radians(90.0)),    # top fin (span +z)
            _rotate_x(math.radians(-90.0)),   # bottom fin (span -z)
        ]
    elif params.n_fins == 4:
        d = params.dihedral
        # Four fins at +-d around the +-y axes
        rotations = [
            _rotate_x(+d),           # upper-right
            _rotate_x(math.pi - d),  # upper-left
            _rotate_x(-d),           # lower-right
            _rotate_x(math.pi + d),  # lower-left
        ]
    else:
        rotations = []

    all_v: List[np.ndarray] = []
    all_f: List[np.ndarray] = []
    all_l: List[np.ndarray] = []
    offset = np.array(attach_xyz, dtype=float)
    n_per_fin = v_loc.shape[0]

    for fin_idx, R in enumerate(rotations):
        v_rot = (R @ v_loc.T).T + offset
        all_v.append(v_rot)
        all_f.append(f_loc + fin_idx * n_per_fin)
        # Tag labels with fin index
        all_l.append(np.array([f"{lbl}_fin{fin_idx + 1}" for lbl in l_loc],
                                dtype=object))

    return Mesh(
        vertices=np.vstack(all_v),
        faces=np.vstack(all_f),
        labels=np.concatenate(all_l),
        metadata={
            "kind": "fins",
            "n_fins": params.n_fins,
            "root_chord": params.root_chord,
            "tip_chord": params.tip_chord,
            "span": params.span,
            "sweep_LE_deg": math.degrees(params.sweep_LE),
            "dihedral_deg": math.degrees(params.dihedral),
            "t_c": params.t_c,
            "max_thickness_loc": params.max_thickness_loc,
            "LE_style": params.LE_style,
            "attach_xyz": tuple(attach_xyz),
        },
    )


def diamond_LE_TE_half_angles(params: FinParams) -> tuple:
    """Compute the leading- and trailing-wedge half-angles of the diamond
    airfoil. Per spec §4.7:

        LE half-angle = atan((t/2) / x_t)
        TE half-angle = atan((t/2) / (c - x_t))

    Returns ``(LE_half_angle_rad, TE_half_angle_rad)`` evaluated at the
    root chord (the larger of the two).
    """
    c = params.root_chord
    t = params.t_c * c
    x_t = params.max_thickness_loc * c
    LE_half = math.atan((t / 2.0) / x_t)
    TE_half = math.atan((t / 2.0) / (c - x_t))
    return LE_half, TE_half


def merge_meshes(meshes: list) -> Mesh:
    """Merge a list of Mesh objects into a single mesh (vertex+face index
    offsets handled). Used to combine body + fins."""
    if not meshes:
        raise ValueError("at least one mesh required")
    all_v: List[np.ndarray] = []
    all_f: List[np.ndarray] = []
    all_l: List[np.ndarray] = []
    offset = 0
    for m in meshes:
        if m is None:
            continue
        all_v.append(m.vertices)
        all_f.append(m.faces + offset)
        all_l.append(m.labels)
        offset += m.n_vertices
    return Mesh(
        vertices=np.vstack(all_v),
        faces=np.vstack(all_f),
        labels=np.concatenate(all_l),
        metadata={"kind": "merged",
                   "components": [m.metadata.get("kind", "?") for m in meshes if m is not None]},
    )
