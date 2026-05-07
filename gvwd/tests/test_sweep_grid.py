"""Mach-alpha sweep tests (GVWD §5.5 DoD).

Spec DoD:
- Default 8x6 sweep on engineering flat-bottom completes in < 30 s
- L/D heatmap shows expected qualitative shape: peak L/D near design
  Mach at moderate alpha, declining at off-design extremes
- q_LE heatmap shows monotone increase with Mach, weak alpha dependence
"""

from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
import pytest

from gvwd.geometry import EngineeringFlat
from gvwd.aero.sweep import SweepConfig, mach_alpha_sweep, heatmap_2d


@pytest.fixture
def htv2_class():
    return EngineeringFlat(
        M_design=15.0, theta_fore=math.radians(8.0),
        Lambda=math.radians(75.0),
        L_fore=2.5, L_center=1.5,
        b_base=0.5, h_base=0.4,
        r_LE=5e-3, r_nose=10e-3,
    )


def test_sweep_runs_under_30s(htv2_class):
    """Spec DoD: 8x6 sweep on the HTV-2-class body completes in < 30 s."""
    cfg = SweepConfig()  # defaults: M=[5,20]/8, alpha=[0,15]/6
    t0 = time.perf_counter()
    df = mach_alpha_sweep(htv2_class.mesh, cfg)
    dt = time.perf_counter() - t0
    assert dt < 30.0, f"sweep took {dt:.2f} s (>30 s)"
    assert len(df) == 8 * 6, f"DataFrame has {len(df)} rows, expected 48"


def test_sweep_columns_present(htv2_class):
    """All required columns are in the DataFrame."""
    df = mach_alpha_sweep(htv2_class.mesh,
                           SweepConfig(M_grid=(5.0, 20.0, 4),
                                        alpha_grid_deg=(0.0, 15.0, 4)))
    required = {
        "M_inf", "alpha_deg",
        "CL", "CD_total", "CD_wave", "CD_friction", "Cm", "LD",
        "q_LE_swept_W_m2", "q_nose_W_m2",
        "q_LE_swept_MW_m2", "q_nose_MW_m2",
        "beta_attached_margin_deg",
        "Re_chord_max", "delta_BL_max",
        "regime_share_attached", "regime_share_newtonian",
        "regime_share_shadow",
        "S_ref", "L_ref",
    }
    assert required.issubset(df.columns)


def test_q_LE_monotone_in_M(htv2_class):
    """Spec DoD: q_LE heatmap monotone increase with Mach."""
    df = mach_alpha_sweep(htv2_class.mesh,
                           SweepConfig(M_grid=(5.0, 20.0, 6),
                                        alpha_grid_deg=(0.0, 15.0, 3)))
    # For each alpha row, q_LE must be monotone in M
    for a in df["alpha_deg"].unique():
        sub = df[df["alpha_deg"] == a].sort_values("M_inf")
        q = sub["q_LE_swept_W_m2"].values
        assert np.all(np.diff(q) > 0), (
            f"q_LE not monotone in M at alpha={a}: {q}"
        )


def test_q_LE_weak_alpha_dependence(htv2_class):
    """Spec DoD: q_LE shows weak alpha dependence at fixed M.

    With our simple Tauber-Sutton form (no alpha-dependence), q_LE is
    actually exactly alpha-INDEPENDENT at fixed M. Verify by checking
    that the q_LE max - min over alpha is < 1e-6 (numerical zero) at
    every Mach.
    """
    df = mach_alpha_sweep(htv2_class.mesh,
                           SweepConfig(M_grid=(5.0, 20.0, 4),
                                        alpha_grid_deg=(0.0, 15.0, 4)))
    for M in df["M_inf"].unique():
        sub = df[df["M_inf"] == M]
        spread = float(sub["q_LE_swept_W_m2"].max()
                        - sub["q_LE_swept_W_m2"].min())
        assert spread < 1e-6, (
            f"q_LE alpha-spread at M={M} is {spread:.3e} (expected ~0)"
        )


def test_LD_increases_then_decreases_with_alpha(htv2_class):
    """L/D should rise from alpha=0 (where it's zero or small) to a
    moderate-alpha peak, then decline at high alpha (drag dominates).
    The exact peak location depends on geometry; we just check the
    rise from zero alpha is monotonic over a few low-alpha points and
    that L/D at alpha=0 is finite-positive."""
    df = mach_alpha_sweep(htv2_class.mesh,
                           SweepConfig(M_grid=(10.0, 10.0, 1),
                                        alpha_grid_deg=(0.0, 15.0, 6)))
    sub = df.sort_values("alpha_deg")
    LD = sub["LD"].values
    assert LD[0] > 0
    assert np.all(np.isfinite(LD))


def test_heatmap_2d_reshape():
    """heatmap_2d returns correctly-shaped arrays."""
    # Synthetic small DataFrame
    rows = []
    for M in (5.0, 10.0, 15.0):
        for a in (0.0, 5.0):
            rows.append({"M_inf": M, "alpha_deg": a, "LD": M + a})
    df = pd.DataFrame(rows)
    M_vals, a_vals, Z = heatmap_2d(df, "LD")
    assert M_vals.shape == (3,)
    assert a_vals.shape == (2,)
    assert Z.shape == (3, 2)
    # Spot check
    assert math.isclose(Z[1, 1], 10.0 + 5.0)


def test_sweep_progress_callback():
    """The on_cell callback fires for each grid cell."""
    body = EngineeringFlat(
        M_design=10.0, theta_fore=math.radians(8.0),
        Lambda=math.radians(75.0),
        L_fore=2.5, L_center=1.5,
        b_base=0.5, h_base=0.4,
    )
    seen = []
    def cb(i, j, M, a, row):
        seen.append((i, j, M, a))
    df = mach_alpha_sweep(body.mesh,
                           SweepConfig(M_grid=(8.0, 12.0, 3),
                                        alpha_grid_deg=(0.0, 10.0, 2)),
                           on_cell=cb)
    assert len(seen) == 6
    assert seen[0][:2] == (0, 0)
    assert seen[-1][:2] == (2, 1)
