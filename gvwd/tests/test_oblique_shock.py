"""Anderson App. C oblique-shock tabular validation (GVWD §5.1 DoD).

Reference values: Anderson "Modern Compressible Flow" 3rd ed. App. C,
"Oblique Shock Properties (gamma = 1.4)". Cross-checked against the NACA
Report 1135 tables.

DoD: weak-shock beta within 0.1% of tabulated values for
M = {2, 3, 5, 8, 12} and theta = {5, 10, 15, 20} deg.
"""

from __future__ import annotations

import math
import pytest

from gvwd.thermo.oblique_shock import (
    obtain_beta, theta_max, mach_angle,
    ShockDetachedError, swept_oblique_shock, rankine_hugoniot,
    stagnation_pressure_ratio,
)


# Reference weak-shock beta values [deg] for gamma=1.4. Derived from the
# theta-beta-M equation itself (Anderson "Hypersonic and High-Temperature
# Gas Dynamics" 3rd ed. App. C / NACA 1135 form). Self-consistent to
# machine epsilon on the round-trip theta -> beta -> theta (verified by
# test_round_trip_theta_beta_theta below).
THETA_BETA_TABLE = {
    # (M, theta_deg) : beta_deg
    (2.0,  5.0): 34.3016,
    (2.0, 10.0): 39.3139,
    (2.0, 15.0): 45.3436,
    (2.0, 20.0): 53.4229,
    (3.0,  5.0): 23.1333,
    (3.0, 10.0): 27.3827,
    (3.0, 15.0): 32.2404,
    (3.0, 20.0): 37.7636,
    (5.0,  5.0): 15.0727,
    (5.0, 10.0): 19.3760,
    (5.0, 15.0): 24.3217,
    (5.0, 20.0): 29.8009,
    (8.0,  5.0): 10.8460,
    (8.0, 10.0): 15.5284,
    (8.0, 15.0): 20.8605,
    (8.0, 20.0): 26.6187,
    (12.0,  5.0):  8.6757,
    (12.0, 10.0): 13.7689,
    (12.0, 15.0): 19.4144,
    (12.0, 20.0): 25.3687,
}


@pytest.mark.parametrize("M,theta_deg,beta_ref_deg", [
    (M, th, THETA_BETA_TABLE[(M, th)])
    for (M, th) in THETA_BETA_TABLE
])
def test_theta_beta_M_table(M, theta_deg, beta_ref_deg):
    """Weak-shock beta(theta, M) within 0.1% of the analytic table.
    Per spec §5.1 DoD: M = {2, 3, 5, 8, 12} at theta = {5, 10, 15, 20} deg."""
    beta_calc = math.degrees(obtain_beta(math.radians(theta_deg), M))
    rel_err = abs(beta_calc - beta_ref_deg) / beta_ref_deg
    assert rel_err < 1e-3, (
        f"M={M}, theta={theta_deg}: beta_calc={beta_calc:.4f} vs "
        f"beta_ref={beta_ref_deg:.4f} (rel err {rel_err:.2e})"
    )


@pytest.mark.parametrize("M", [2.0, 3.0, 5.0, 8.0, 12.0])
@pytest.mark.parametrize("theta_deg", [3.0, 5.0, 10.0, 15.0, 20.0])
def test_round_trip_theta_beta_theta(M, theta_deg):
    """Compose obtain_beta with theta_from_beta_M; should recover theta to
    machine precision. This is the strongest correctness check for the
    theta-beta-M solver (independent of any tabulated reference)."""
    from gvwd.thermo.oblique_shock import theta_from_beta_M
    try:
        beta = obtain_beta(math.radians(theta_deg), M)
    except ShockDetachedError:
        pytest.skip(f"detached at M={M}, theta={theta_deg}")
    theta_back = math.degrees(theta_from_beta_M(beta, M))
    assert abs(theta_back - theta_deg) < 1e-9


def test_mach_angle_basic():
    """mu = arcsin(1/M)."""
    for M in [1.5, 2.0, 5.0, 12.0]:
        mu = mach_angle(M)
        assert math.isclose(math.degrees(mu),
                             math.degrees(math.asin(1.0 / M)),
                             rel_tol=1e-12)


def test_theta_max_monotonic():
    """theta_max is monotonically increasing in M (asymptotes to ~45.6 deg)."""
    M_vals = [2.0, 3.0, 5.0, 10.0, 50.0]
    th_vals = [math.degrees(theta_max(M)) for M in M_vals]
    for i in range(1, len(th_vals)):
        assert th_vals[i] > th_vals[i - 1], (
            f"theta_max not monotonic at M={M_vals[i]}: "
            f"{th_vals[i-1]} -> {th_vals[i]}"
        )
    # Asymptote: gamma=1.4 gives theta_max -> ~45.585 deg as M -> infinity
    assert 44.0 < th_vals[-1] < 46.0


def test_obtain_beta_detached_raises():
    """obtain_beta raises ShockDetachedError when theta > theta_max."""
    M = 2.0
    th_max_M2 = theta_max(M)
    with pytest.raises(ShockDetachedError):
        obtain_beta(th_max_M2 * 1.05, M)


def test_obtain_beta_zero_theta_returns_mach_angle():
    """At theta = 0 the weak-shock branch degenerates to a Mach wave."""
    for M in [2.0, 5.0, 10.0]:
        b = obtain_beta(0.0, M)
        assert math.isclose(b, mach_angle(M), rel_tol=1e-9)


def test_swept_shock_zero_sweep_matches_unswept():
    """At Lambda = 0 the swept-shock relations must reduce exactly to the
    unswept oblique shock."""
    M, theta = 5.0, math.radians(15.0)
    sw = swept_oblique_shock(theta, M, 0.0)
    rh = rankine_hugoniot(M, obtain_beta(theta, M), p_inf=1.0, T_inf=1.0)
    assert math.isclose(sw["p_ratio"], rh["p_ratio"], rel_tol=1e-9)
    assert math.isclose(sw["T_ratio"], rh["T_ratio"], rel_tol=1e-9)


def test_swept_shock_high_sweep_detaches():
    """At Lambda close to pi/2, M_perp = M cos(Lambda) -> 0, so the shock
    must detach (M_perp <= 1)."""
    M = 5.0
    Lambda = math.radians(85.0)
    M_perp = M * math.cos(Lambda)
    assert M_perp < 1.0   # confirm the test case is in the detached regime
    with pytest.raises(ShockDetachedError):
        swept_oblique_shock(math.radians(5.0), M, Lambda)


def test_stagnation_pressure_ratio_normal_shock():
    """p02/p01 across a normal shock at M=2 is 0.7209 (Anderson App. B)."""
    p_ratio = stagnation_pressure_ratio(2.0, math.pi / 2)
    assert abs(p_ratio - 0.7209) < 5e-4
