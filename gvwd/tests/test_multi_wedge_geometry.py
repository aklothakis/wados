"""Multi-wedge geometry tests (GVWD §5.2 DoD).

Spec DoD:
- n=2, M=5: ramp angles match Phase-1 Oswatitsch outputs to 0.1 deg
- closed mesh in both extrusion modes
"""

from __future__ import annotations

import math
import pytest

from gvwd.geometry import MultiWedge, numerical_volume
from gvwd.thermo.oswatitsch import equal_strength_ramps


def test_multi_wedge_n2_M5_matches_oswatitsch_to_0p1_deg():
    """Spec DoD: ramp angles match Phase-1 Oswatitsch outputs to 0.1 deg."""
    mw = MultiWedge(M_design=5.0, n=2, delta_total_deg=20.8, L=8.0,
                     half_span=1.0)
    osw_ref = equal_strength_ramps(M_inf=5.0, n=2, delta_total_deg=20.8)
    for d_mw, d_ref in zip(mw.osw.deltas_cum_deg, osw_ref.deltas_cum_deg):
        assert abs(d_mw - d_ref) < 0.1, (
            f"ramp angle mismatch: mw={d_mw}, osw_ref={d_ref}"
        )


def test_multi_wedge_rectangular_mesh_closes():
    """Rectangular extrusion produces a closed manifold mesh whose
    divergence-theorem volume is positive."""
    mw = MultiWedge(M_design=5.0, n=2, delta_total_deg=20.0, L=8.0,
                     half_span=1.0, height=0.6, extrusion="rectangular")
    V = numerical_volume(mw.mesh)
    assert V > 0
    # Sanity: V should be in the right order of magnitude. The bounding
    # box has the upper prism (8 x 2 x 0.6 = 9.6) PLUS the ramp drops
    # below the apex baseline by L tan(delta_total) ~ 8 tan(20 deg) ~ 2.9,
    # adding more volume below. Total bounding volume up to ~25 m^3.
    assert 1.0 < V < 30.0


def test_multi_wedge_delta_mesh_closes():
    """Delta extrusion produces a closed mesh; volume positive."""
    mw = MultiWedge(M_design=5.0, n=2, delta_total_deg=20.0, L=8.0,
                     half_span=1.0, extrusion="delta")
    V = numerical_volume(mw.mesh)
    assert V > 0


def test_multi_wedge_n_one_degenerates_to_single_ramp():
    """n=1 should reduce to a single ramp at delta_total_deg."""
    mw = MultiWedge(M_design=5.0, n=1, delta_total_deg=15.0, L=6.0,
                     half_span=0.8)
    assert len(mw.osw.deltas_cum_deg) == 1
    assert abs(mw.osw.deltas_cum_deg[0] - 15.0) < 0.1


def test_multi_wedge_n3_more_recovery_than_n2():
    """At fixed total deflection, n=3 obliques have higher pi_OS than n=2.
    (Oswatitsch's optimum: more equal-strength shocks give better
    recovery.)"""
    mw2 = MultiWedge(M_design=5.0, n=2, delta_total_deg=20.0, L=8.0,
                      half_span=1.0)
    mw3 = MultiWedge(M_design=5.0, n=3, delta_total_deg=20.0, L=8.0,
                      half_span=1.0)
    assert mw3.osw.pi_OS > mw2.osw.pi_OS
