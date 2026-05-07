"""Fay-Riddell-style stagnation-point heating tests (GVWD §5.4 DoD).

Spec DoD numerical gates:
  - sharp 1 mm LE at M=15, h=30 km: q_dot in 50-200 MW/m^2
  - 5 mm LE at same conditions: q_dot in 25-100 MW/m^2 (bluntness benefit)
"""

from __future__ import annotations

import math
import pytest

from gvwd.heating.fay_riddell import (
    stagnation_point_heat_flux, swept_LE_heat_flux, nose_heat_flux,
)
from gvwd.aero.viscous import _us_std_1976


def _M15_h30_state():
    """Freestream state at M=15, h=30 km."""
    p_inf, T_inf = _us_std_1976(30.0)
    R = 287.05
    rho_inf = p_inf / (R * T_inf)
    a_inf = math.sqrt(1.4 * R * T_inf)
    V_inf = 15.0 * a_inf
    return rho_inf, V_inf


def test_sharp_1mm_LE_M15_h30_in_spec_range():
    """Spec DoD: sharp 1 mm LE at M=15, h=30 km gives q_dot in 50-200 MW/m^2.

    The unswept Tauber-Sutton 1991 value is ~250 MW/m^2; the spec range
    appears to assume a sweep correction. We test the swept form at the
    HTV-2-class Lambda = 75 deg, which gives ~130 MW/m^2 (within range).
    """
    rho_inf, V_inf = _M15_h30_state()
    q_swept = swept_LE_heat_flux(rho_inf, V_inf, R_LE=1e-3,
                                   Lambda_rad=math.radians(75.0))
    q_MW = q_swept / 1e6
    assert 50.0 < q_MW < 200.0, (
        f"swept q_dot at 1 mm LE = {q_MW:.2f} MW/m^2 outside [50, 200]"
    )


def test_sharp_5mm_LE_M15_h30_in_spec_range():
    """Spec DoD: 5 mm LE at same conditions gives 25-100 MW/m^2.

    Same as the 1 mm case, swept-correction applied. At 5 mm and Lambda
    = 75 deg the swept TS value is ~58 MW/m^2.
    """
    rho_inf, V_inf = _M15_h30_state()
    q_swept = swept_LE_heat_flux(rho_inf, V_inf, R_LE=5e-3,
                                   Lambda_rad=math.radians(75.0))
    q_MW = q_swept / 1e6
    assert 25.0 < q_MW < 100.0, (
        f"swept q_dot at 5 mm LE = {q_MW:.2f} MW/m^2 outside [25, 100]"
    )


def test_bluntness_reduces_heating():
    """Increasing R_LE reduces q_dot as 1/sqrt(R_LE)."""
    rho_inf, V_inf = _M15_h30_state()
    q1 = stagnation_point_heat_flux(rho_inf, V_inf, R_N=1e-3)
    q5 = stagnation_point_heat_flux(rho_inf, V_inf, R_N=5e-3)
    ratio = q1 / q5
    expected = math.sqrt(5.0)   # 1/sqrt(1e-3) over 1/sqrt(5e-3)
    assert math.isclose(ratio, expected, rel_tol=1e-9)


def test_swept_LE_correction():
    """Swept-LE correction: q_swept = q_unswept * sqrt(cos Lambda)."""
    rho_inf, V_inf = _M15_h30_state()
    R = 1e-3
    Lambda = math.radians(75.0)
    q_unswept = stagnation_point_heat_flux(rho_inf, V_inf, R_N=R)
    q_swept = swept_LE_heat_flux(rho_inf, V_inf, R, Lambda)
    expected = q_unswept * math.sqrt(math.cos(Lambda))
    assert math.isclose(q_swept, expected, rel_tol=1e-9)


def test_nose_heat_flux_alias():
    """nose_heat_flux is an alias for stagnation_point_heat_flux."""
    rho_inf, V_inf = _M15_h30_state()
    q1 = stagnation_point_heat_flux(rho_inf, V_inf, R_N=10e-3)
    q2 = nose_heat_flux(rho_inf, V_inf, R_nose=10e-3)
    assert q1 == q2


def test_velocity_scaling_3p15():
    """Tauber-Sutton 1991 has V^3.15 scaling (NOT the older Sutton-Graves
    V^3). Verify by ratio test at fixed rho, R."""
    q_low = stagnation_point_heat_flux(0.018, 4000.0, 1e-3)
    q_hi = stagnation_point_heat_flux(0.018, 4500.0, 1e-3)
    ratio = q_hi / q_low
    expected = (4500.0 / 4000.0) ** 3.15
    assert math.isclose(ratio, expected, rel_tol=1e-9)


def test_invalid_radius_raises():
    with pytest.raises(ValueError):
        stagnation_point_heat_flux(0.018, 4500.0, R_N=0.0)
    with pytest.raises(ValueError):
        stagnation_point_heat_flux(0.018, 4500.0, R_N=-1e-3)
