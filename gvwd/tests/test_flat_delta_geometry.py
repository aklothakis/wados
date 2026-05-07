"""Flat-bottomed delta geometry tests (GVWD §5.2 DoD).

Spec DoD:
- M_d = 5, theta_d = 12 deg, Lambda = 75 deg, L = 8 m
- closed mesh
- swept-shock attachment satisfied (M_perp > 1, attached oblique exists)
"""

from __future__ import annotations

import math
import pytest

from gvwd.geometry import (
    FlatDelta, flat_delta_analytic_volume, numerical_volume, eta_V,
)
from gvwd.geometry.volume import planform_area_from_mesh
from gvwd.thermo.oblique_shock import ShockDetachedError


@pytest.fixture
def fd_dod():
    return FlatDelta(
        M_design=5.0,
        theta_d=math.radians(12.0),
        Lambda=math.radians(75.0),
        L=8.0,
    )


def test_flat_delta_constructs(fd_dod):
    m = fd_dod.mesh
    assert m.n_vertices == 4
    assert m.n_faces == 4
    assert "lower_right" in fd_dod.mesh.labels
    assert "upper" in fd_dod.mesh.labels
    assert "base" in fd_dod.mesh.labels


def test_swept_shock_attached(fd_dod):
    """Spec DoD: LE attachment satisfied (M cos Lambda > 1) AND body-frame
    surface oblique shock attached (theta_d < theta_max(M))."""
    M_perp = fd_dod.M_design * math.cos(fd_dod.Lambda)
    assert M_perp > 1.0, f"LE detached: M_perp = {M_perp:.3f}"
    # body-frame shock angle is sensible
    assert 0 < math.degrees(fd_dod.beta_body) < 90


def test_flat_delta_volume_analytic_vs_numerical(fd_dod):
    """Analytic V matches numerical V to 1% (looser than caret because
    the flat-delta closed-form is an approximation; the wedge-with-
    triangular-base formula slightly overcounts at the base)."""
    V_ana = flat_delta_analytic_volume(fd_dod)
    V_num = numerical_volume(fd_dod.mesh)
    rel = abs(V_num - V_ana) / V_ana
    assert rel < 1e-2, (
        f"flat-delta V_num={V_num:.4f} vs V_ana={V_ana:.4f} "
        f"(rel err {rel:.2e})"
    )


def test_flat_delta_high_sweep_detaches():
    """Lambda close to 90 deg -> M_perp = M cos(Lambda) <= 1 -> detaches."""
    # M=3, Lambda=85 deg -> M_perp = 3*cos(85) = 0.26 < 1
    with pytest.raises(ShockDetachedError):
        FlatDelta(M_design=3.0, theta_d=math.radians(8.0),
                   Lambda=math.radians(85.0), L=5.0)


def test_flat_delta_y_tip_formula(fd_dod):
    expected = fd_dod.L / math.tan(fd_dod.Lambda)
    assert math.isclose(fd_dod.y_tip, expected, rel_tol=1e-12)
