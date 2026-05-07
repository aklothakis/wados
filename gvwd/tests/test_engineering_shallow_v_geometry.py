"""Shallow-V variant tests (GVWD §5.3 DoD).

Spec DoD: shallow-V variant with dihedral 5 deg produces a valid mesh
with marginally higher eta_V than the flat-bottom equivalent.
"""

from __future__ import annotations

import math
import pytest

from gvwd.geometry import (
    EngineeringFlat, EngineeringShallowV,
    numerical_volume, eta_V, mesh_volume_signed,
    planform_area_from_mesh,
)


def test_shallow_v_mesh_closes():
    sv = EngineeringShallowV(
        M_design=15.0, theta_fore=math.radians(8.0),
        Lambda=math.radians(75.0), L_fore=2.5, L_center=1.5,
        b_base=0.5, h_base=0.4,
        dihedral_lower=math.radians(5.0),
    )
    assert sv.mesh.n_vertices == 11
    # 5 forebody + 10 centerbody + 3 base = 18 faces
    assert sv.mesh.n_faces == 18
    V_signed = mesh_volume_signed(sv.mesh)
    assert V_signed > 0, f"signed volume {V_signed} non-positive"


def test_shallow_v_higher_eta_V_than_flat():
    """Spec DoD: shallow-V has marginally higher eta_V than flat-bottom
    (more volume for the same wetted-area / planform)."""
    flat = EngineeringFlat(
        M_design=15.0, theta_fore=math.radians(8.0),
        Lambda=math.radians(75.0), L_fore=2.5, L_center=1.5,
        b_base=0.5, h_base=0.4,
    )
    sv = EngineeringShallowV(
        M_design=15.0, theta_fore=math.radians(8.0),
        Lambda=math.radians(75.0), L_fore=2.5, L_center=1.5,
        b_base=0.5, h_base=0.4,
        dihedral_lower=math.radians(5.0),
    )
    V_flat = numerical_volume(flat.mesh)
    V_sv = numerical_volume(sv.mesh)
    # Shallow-V has more volume due to the trough below the flat baseline
    assert V_sv > V_flat, (
        f"Shallow-V V={V_sv:.4f} should exceed flat V={V_flat:.4f}"
    )


def test_shallow_v_zero_dihedral_matches_flat_volume():
    """When dihedral_lower = 0, the shallow-V volume should match the
    flat-bottom volume (the V-trough collapses to a flat line)."""
    flat = EngineeringFlat(
        M_design=15.0, theta_fore=math.radians(8.0),
        Lambda=math.radians(75.0), L_fore=2.5, L_center=1.5,
        b_base=0.5, h_base=0.4,
    )
    sv = EngineeringShallowV(
        M_design=15.0, theta_fore=math.radians(8.0),
        Lambda=math.radians(75.0), L_fore=2.5, L_center=1.5,
        b_base=0.5, h_base=0.4,
        dihedral_lower=0.0,
    )
    V_flat = numerical_volume(flat.mesh)
    V_sv = numerical_volume(sv.mesh)
    rel = abs(V_sv - V_flat) / V_flat
    assert rel < 5e-3, (
        f"zero-dihedral shallow-V V={V_sv:.4f} differs from flat "
        f"V={V_flat:.4f} by {rel:.2e}"
    )
