"""Fin geometry tests (GVWD §5.3 DoD, §5.7 fin diamond geometry).

Includes the spec's validation cases #14, #15, #16 for the diamond
airfoil and LE-style switch.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from gvwd.geometry import FinParams, generate_fins, merge_meshes
from gvwd.geometry.fins import diamond_LE_TE_half_angles
from gvwd.geometry.mesh import mesh_volume_signed
from gvwd.geometry import EngineeringFlat, numerical_volume


def test_diamond_symmetric_LE_TE_half_angles():
    """Validation case #14: t/c=0.05, x_t/c=0.5 -> both half-angles
    equal atan((t/2)/x_t)."""
    p = FinParams(t_c=0.05, max_thickness_loc=0.5, root_chord=0.3,
                   tip_chord=0.1, span=0.4, n_fins=4)
    LE_h, TE_h = diamond_LE_TE_half_angles(p)
    expected = math.atan((p.t_c * p.root_chord / 2.0)
                          / (0.5 * p.root_chord))
    assert math.isclose(LE_h, expected, rel_tol=1e-12)
    assert math.isclose(TE_h, expected, rel_tol=1e-12)


def test_diamond_asymmetric_LE_TE_half_angles_distinct():
    """Validation case #15: t/c=0.05, x_t/c=0.4 -> distinct LE/TE."""
    p = FinParams(t_c=0.05, max_thickness_loc=0.4, root_chord=0.3,
                   tip_chord=0.1, span=0.4, n_fins=4)
    LE_h, TE_h = diamond_LE_TE_half_angles(p)
    assert not math.isclose(LE_h, TE_h, rel_tol=1e-3)
    # LE half-angle should be larger (smaller x_t -> steeper LE wedge)
    assert LE_h > TE_h


def test_fin_params_validation():
    """t/c outside [0.02, 0.10] is rejected."""
    with pytest.raises(ValueError):
        FinParams(t_c=0.5, max_thickness_loc=0.5,
                   root_chord=0.3, tip_chord=0.1, span=0.4, n_fins=2)
    with pytest.raises(ValueError):
        FinParams(t_c=0.05, max_thickness_loc=0.1,   # too far forward
                   root_chord=0.3, tip_chord=0.1, span=0.4, n_fins=2)


def test_n_fins_zero_returns_none():
    p = FinParams(n_fins=0)
    fm = generate_fins(p, attach_xyz=(0, 0, 0))
    assert fm is None


def test_two_fin_construction():
    p = FinParams(n_fins=2, root_chord=0.3, tip_chord=0.1, span=0.4,
                   sweep_LE=math.radians(45), t_c=0.05,
                   max_thickness_loc=0.5)
    fm = generate_fins(p, attach_xyz=(3.0, 0.0, 0.0))
    assert fm is not None
    # 2 fins x 8 verts = 16 verts; 2 fins x 12 tri = 24 tri
    assert fm.n_vertices == 16
    assert fm.n_faces == 24


def test_four_fin_construction():
    p = FinParams(n_fins=4, root_chord=0.3, tip_chord=0.1, span=0.4,
                   sweep_LE=math.radians(45),
                   dihedral=math.radians(45),
                   t_c=0.05, max_thickness_loc=0.5)
    fm = generate_fins(p, attach_xyz=(3.0, 0.0, 0.0))
    assert fm.n_vertices == 32
    assert fm.n_faces == 48


def test_single_fin_signed_volume_positive():
    """Each individual fin has consistent outward normals -> positive
    signed volume."""
    p = FinParams(n_fins=2, root_chord=0.3, tip_chord=0.1, span=0.4,
                   sweep_LE=math.radians(30), t_c=0.05,
                   max_thickness_loc=0.5)
    fm = generate_fins(p, attach_xyz=(3.0, 0.0, 0.0))
    V = mesh_volume_signed(fm)
    # Two fins of approximate volume ~ 0.5 * (root_c + tip_c)*span*t/2
    # = 0.5 * 0.4 * 0.4 * 0.0075 ~ 6e-4 m^3 per fin -> 1.2e-3 total.
    # Sign is the diagnostic; magnitude roughly bounded.
    assert V > 0
    assert 1e-5 < V < 1e-1


def test_fin_LE_style_blunt_cylinder_flag_only():
    """For Phase 3, blunt_cylinder is currently a flag only; the mesh is
    the same as sharp. (Detailed cylindrical-fillet construction is a
    Phase-3-stretch refinement.)"""
    p = FinParams(n_fins=2, t_c=0.05, max_thickness_loc=0.5,
                   root_chord=0.3, tip_chord=0.1, span=0.4,
                   LE_style="blunt_cylinder", LE_radius=1e-3)
    fm = generate_fins(p, attach_xyz=(3.0, 0.0, 0.0))
    assert fm is not None
    assert fm.metadata["LE_style"] == "blunt_cylinder"


def test_engineering_flat_with_fins_merged_closes():
    """HTV-2-class body + 4 fins: the merged mesh has proper vertex
    counts and a positive (combined-component) volume."""
    body = EngineeringFlat(
        M_design=15.0, theta_fore=math.radians(8.0),
        Lambda=math.radians(75.0),
        L_fore=2.5, L_center=1.5,
        b_base=0.5, h_base=0.4,
        r_LE=5e-3, r_nose=10e-3,
    )
    fp = FinParams(n_fins=4, root_chord=0.3, tip_chord=0.1,
                    span=0.4, sweep_LE=math.radians(45),
                    dihedral=math.radians(45), t_c=0.05,
                    max_thickness_loc=0.5)
    fins_mesh = generate_fins(fp,
                                attach_xyz=(body.L_fore + body.L_center * 0.5,
                                             0.0, -0.1))
    merged = merge_meshes([body.mesh, fins_mesh])
    assert merged.n_vertices == body.mesh.n_vertices + fins_mesh.n_vertices
    assert merged.n_faces == body.mesh.n_faces + fins_mesh.n_faces
    # Combined signed volume = body + sum(fins) (each component is
    # closed, so the merged "mesh" is two disjoint closed surfaces).
    V_merged = mesh_volume_signed(merged)
    assert V_merged > 0
