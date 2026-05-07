"""Caret geometry + analytic-volume tests (GVWD §5.2 DoD).

Spec DoD:
- M_d = 6, theta_d = 14 deg, Lambda = 70 deg, L = 10 m
- closed manifold mesh
- eta_V matches analytic to 0.5%
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from gvwd.geometry import (
    Caret, caret_analytic_volume, numerical_volume, eta_V,
    mesh_volume_signed,
)
from gvwd.geometry.volume import planform_area_from_mesh


@pytest.fixture
def caret_dod():
    return Caret(
        M_design=6.0,
        theta_d=math.radians(14.0),
        Lambda=math.radians(70.0),
        L=10.0,
    )


def test_caret_construction(caret_dod):
    """Mesh has 5 vertices and 6 triangles (two-panel upper, two-panel
    lower meeting at centerline ridges, diamond base split into 2)."""
    m = caret_dod.mesh
    assert m.n_vertices == 5
    assert m.n_faces == 6
    labels = set(m.labels)
    assert "upper_right" in labels
    assert "lower_right" in labels
    assert "base_right" in labels


def test_caret_analytic_volume_dod(caret_dod):
    """Analytic V = (1/3) L b z_TE. Cross-checked vs the direct mesh
    divergence-theorem volume to within 0.5%."""
    V_ana = caret_analytic_volume(caret_dod)
    V_num = numerical_volume(caret_dod.mesh)
    rel_err = abs(V_num - V_ana) / V_ana
    assert rel_err < 5e-3, (
        f"caret V_num={V_num:.4f} vs V_ana={V_ana:.4f} (rel err {rel_err:.2e})"
    )


def test_caret_eta_V_within_tolerance(caret_dod):
    """eta_V from analytic V matches eta_V from numerical V to 0.5%."""
    V_ana = caret_analytic_volume(caret_dod)
    V_num = numerical_volume(caret_dod.mesh)
    S = planform_area_from_mesh(caret_dod.mesh)
    eta_ana = eta_V(V_ana, S)
    eta_num = eta_V(V_num, S)
    rel = abs(eta_num - eta_ana) / eta_ana
    assert rel < 5e-3, f"eta_V rel err {rel:.2e}"


def test_caret_input_kind_beta():
    """Caret can accept beta_d directly; resulting theta_d should match
    theta-beta-M for that beta."""
    from gvwd.thermo.oblique_shock import theta_from_beta_M
    M, b_d, Lam, L = 6.0, math.radians(20.0), math.radians(70.0), 10.0
    c = Caret(M_design=M, beta_d=b_d, Lambda=Lam, L=L, input_kind="beta")
    assert math.isclose(
        c.theta_d, theta_from_beta_M(b_d, M), rel_tol=1e-9
    )


def test_caret_y_tip_formula(caret_dod):
    """y_tip = L / tan(Lambda)."""
    expected = caret_dod.L / math.tan(caret_dod.Lambda)
    assert math.isclose(caret_dod.y_tip, expected, rel_tol=1e-12)


def test_caret_signed_volume_outward_normals():
    """If the mesh is constructed with consistent outward normals, the
    signed-volume integral is positive (and equals the absolute volume)."""
    c = Caret(M_design=6.0, theta_d=math.radians(14.0),
               Lambda=math.radians(70.0), L=10.0)
    V_signed = mesh_volume_signed(c.mesh)
    V_ana = caret_analytic_volume(c)
    # Expect outward normals -> positive signed volume close to V_ana
    # (within the same 0.5% tolerance).
    assert V_signed > 0
    rel = abs(V_signed - V_ana) / V_ana
    assert rel < 5e-3, (
        f"signed V = {V_signed:.4f}, analytic V = {V_ana:.4f}, "
        f"rel err {rel:.2e}"
    )
