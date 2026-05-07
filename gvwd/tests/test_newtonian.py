"""Modified Newtonian Cp_max tests (GVWD §5.1 DoD)."""

from __future__ import annotations

import math
import numpy as np
import pytest

from gvwd.thermo.newtonian import (
    cp_max_modified_newtonian, modified_newtonian_cp,
    classical_newtonian_cp,
)


def test_cp_max_M10_gamma14():
    """spec DoD: Cp_max(M=10, gamma=1.4) ~ 1.83."""
    cpm = cp_max_modified_newtonian(10.0, 1.4)
    assert abs(cpm - 1.832) < 5e-3, f"Cp_max(M=10) = {cpm:.4f}, expected ~1.832"


def test_cp_max_high_M_asymptote():
    """As M -> infinity, Cp_max -> 1.839 for gamma=1.4 (Lees limit)."""
    cpm_inf = cp_max_modified_newtonian(1000.0, 1.4)
    assert abs(cpm_inf - 1.839) < 1e-2


def test_cp_max_below_M2_finite():
    """Cp_max remains finite and positive for M >= ~2."""
    for M in [2.0, 3.0, 5.0, 10.0, 30.0]:
        cpm = cp_max_modified_newtonian(M, 1.4)
        assert cpm > 0.0
        assert cpm < 2.0
        assert math.isfinite(cpm)


def test_modified_newtonian_zero_at_zero_incidence():
    cp = modified_newtonian_cp(10.0, 0.0)
    assert cp == 0.0


def test_modified_newtonian_shadow_zero():
    """Lee-side panels (theta < 0) must return Cp = 0."""
    cp = modified_newtonian_cp(10.0, math.radians(-15.0))
    assert cp == 0.0


def test_modified_newtonian_at_pi_over_2_equals_cp_max():
    """At theta = pi/2 (face-on), Cp = Cp_max."""
    M = 10.0
    cpm = cp_max_modified_newtonian(M)
    cp_face = modified_newtonian_cp(M, math.pi / 2)
    assert math.isclose(cp_face, cpm, rel_tol=1e-12)


def test_modified_newtonian_array():
    """Vectorised call returns array with shadow handled."""
    M = 8.0
    angles = np.array([math.radians(a) for a in [-10, 0, 5, 15, 45, 90]])
    cps = modified_newtonian_cp(M, angles)
    assert cps.shape == angles.shape
    assert cps[0] == 0.0   # shadow
    assert cps[1] == 0.0   # zero incidence
    cpm = cp_max_modified_newtonian(M)
    assert math.isclose(cps[5], cpm, rel_tol=1e-12)
    # Monotone in (0, pi/2)
    assert cps[2] < cps[3] < cps[4] < cps[5]


def test_classical_newtonian_at_15_deg():
    """Classical Cp = 2 sin^2(theta) at 15 deg ~ 0.134."""
    cp = classical_newtonian_cp(math.radians(15.0))
    assert abs(cp - 2.0 * math.sin(math.radians(15.0)) ** 2) < 1e-12
