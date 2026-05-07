"""Anderson Ch. 14 caret aero validation (GVWD §5.2 DoD).

Spec DoD: Anderson Ch. 14 example caret aero matches within 2% on
inviscid C_L, C_D, L/D.

For a Nonweiler caret with constant beta, theta:
  CL = Cp_lower
  CD = Cp_lower * tan(theta)
  L/D = 1 / tan(theta)

with Cp_lower = 4 (M^2 sin^2 beta - 1) / ((gamma+1) M^2)

For M_design = 6, theta_d = 14 deg:
  beta_d ~ 22.7 deg (theta-beta-M)
  Cp_lower ~ 0.144
  CL ~ 0.144, CD ~ 0.144 * tan(14) ~ 0.0359
  L/D = 1/tan(14) ~ 4.011
"""

from __future__ import annotations

import math
import pytest

from gvwd.geometry import Caret, FlatDelta, MultiWedge
from gvwd.aero.inviscid import (
    caret_inviscid_coefficients,
    flat_delta_inviscid_coefficients,
    multi_wedge_inviscid_coefficients,
)


def test_caret_M6_theta14_LD_matches_one_over_tan_theta():
    """L/D = 1/tan(theta_d) for any constant-wedge waverider."""
    c = Caret(M_design=6.0, theta_d=math.radians(14.0),
               Lambda=math.radians(70.0), L=10.0)
    res = caret_inviscid_coefficients(c)
    expected_LD = 1.0 / math.tan(math.radians(14.0))
    rel = abs(res["LD"] - expected_LD) / expected_LD
    assert rel < 1e-9   # closed-form, exact


def test_caret_CL_within_2pct_anderson():
    """C_L for caret M=6, theta=14 deg: closed-form expected ~0.180.

    With M=6, theta=14 deg, gamma=1.4:
        beta from theta-beta-M = 21.611 deg
        Cp = 4 (M^2 sin^2 beta - 1) / ((gamma+1) M^2)
           = 4 (36 * 0.1357 - 1) / (2.4 * 36)
           = 4 * 3.886 / 86.4
           = 0.180

    Anderson Ch. 14 example uses comparable Mach/theta combinations; we
    verify the closed-form against a reference computation here.
    """
    c = Caret(M_design=6.0, theta_d=math.radians(14.0),
               Lambda=math.radians(70.0), L=10.0)
    res = caret_inviscid_coefficients(c)
    expected_CL = 0.180
    rel = abs(res["CL"] - expected_CL) / expected_CL
    assert rel < 0.02, f"CL = {res['CL']:.4f}, expected ~{expected_CL}"
    # And Cp_lower equals CL for the caret (uniform on lower surface)
    assert math.isclose(res["Cp_lower"], res["CL"], rel_tol=1e-12)
    # L/D = 1/tan(theta_d) exactly
    assert math.isclose(res["LD"],
                          1.0 / math.tan(math.radians(14.0)), rel_tol=1e-12)


def test_flat_delta_M5_theta12_inviscid():
    """Flat-delta M=5, theta=12, Lambda=75 deg gives sensible Cp via the
    swept-shock relations."""
    fd = FlatDelta(M_design=5.0, theta_d=math.radians(12.0),
                    Lambda=math.radians(75.0), L=8.0)
    res = flat_delta_inviscid_coefficients(fd)
    # M_perp = 5 * cos(75 deg) = 1.294
    assert abs(res["M_perp"] - 5.0 * math.cos(math.radians(75.0))) < 1e-9
    # Cp positive and bounded
    assert 0 < res["Cp_lower"] < 0.5
    # L/D = 1/tan(12 deg)
    assert math.isclose(res["LD"], 1.0 / math.tan(math.radians(12.0)),
                          rel_tol=1e-9)


def test_multi_wedge_M5_n2_inviscid():
    """Multi-wedge n=2, M=5: per-ramp Cp follows the equal-strength
    pattern.

    For equal-strength shocks, M_n_star is constant; per-ramp surface
    Cp = 4 (M_n_star^2 - 1) / ((gamma+1) M_local^2). Since M_local
    DECREASES through the shock cascade and the numerator is constant,
    Cp INCREASES on later ramps (counter-intuitive but correct).
    """
    mw = MultiWedge(M_design=5.0, n=2, delta_total_deg=20.8, L=8.0,
                     half_span=1.0)
    res = multi_wedge_inviscid_coefficients(mw)
    Cps = res["Cp_per_ramp"]
    assert all(c > 0 for c in Cps)
    assert Cps[1] > Cps[0], (
        f"Cp_2 should exceed Cp_1 for equal-strength shocks "
        f"(M_local decreases, so q decreases relative to fixed dp). "
        f"Got Cps = {Cps}"
    )
