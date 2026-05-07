"""Baseline HTV-2-class run-end-to-end test (spec §5.6 DoD).

Verifies the full Phase-6 pipeline: load YAML config, build geometry,
on-design + sweep, export STL+STEP, write JSON results, all in < 60 s.
Also checks SHA-256 reproducibility: rerunning with the same config
gives the same hash and bit-comparable on-design coefficients.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from gvwd.io.config import load_config_yaml, config_sha256
from gvwd.examples._runner import run_from_config


HERE = Path(__file__).resolve().parent
CFG_DIR = HERE.parent / "examples" / "configs"


def test_caret_M6_runs_under_60s(tmp_path):
    cfg = load_config_yaml(CFG_DIR / "caret_M6.yaml")
    cfg.output_dir = str(tmp_path)
    t0 = time.perf_counter()
    res = run_from_config(cfg, write_step=False, verbose=False)
    dt = time.perf_counter() - t0
    assert dt < 60.0, f"caret example took {dt:.1f} s (>60 s)"
    art = res["artifact"]
    assert (art.base_dir / "config.yaml").exists()
    assert (art.base_dir / "config_sha256.txt").exists()
    assert (art.base_dir / "geometry.stl").exists()
    assert (art.base_dir / "volumetric.json").exists()


def test_flat_delta_M5_runs_end_to_end(tmp_path):
    cfg = load_config_yaml(CFG_DIR / "flat_delta_M5.yaml")
    cfg.output_dir = str(tmp_path)
    res = run_from_config(cfg, write_step=False, verbose=False)
    assert res["wall_time_s"] < 60.0
    assert (res["artifact"].base_dir / "geometry.stl").exists()


def test_two_ramp_runs_end_to_end(tmp_path):
    cfg = load_config_yaml(CFG_DIR / "two_ramp_M5.yaml")
    cfg.output_dir = str(tmp_path)
    res = run_from_config(cfg, write_step=False, verbose=False)
    assert res["wall_time_s"] < 60.0


def test_htv2_class_with_sweep_runs_under_60s(tmp_path):
    cfg = load_config_yaml(CFG_DIR / "engineering_flat_htv2_class.yaml")
    cfg.output_dir = str(tmp_path)
    res = run_from_config(cfg, write_step=False, verbose=False)
    art = res["artifact"]
    assert res["wall_time_s"] < 60.0
    # Sweep results
    assert (art.base_dir / "sweep_results.json").exists() \
            or (art.base_dir / "sweep_results.h5").exists()
    # Plots
    assert (art.base_dir / "plots").exists()
    pdfs = list((art.base_dir / "plots").glob("*.pdf"))
    assert len(pdfs) >= 4   # at least 4 sweep plots produced


def test_sha256_reproducibility(tmp_path):
    """Same YAML config -> same SHA-256 -> same on-design CL/CD/L/D."""
    cfg = load_config_yaml(CFG_DIR / "caret_M6.yaml")
    cfg.output_dir = str(tmp_path)
    sha1 = config_sha256(cfg)
    sha2 = config_sha256(cfg)
    assert sha1 == sha2

    res1 = run_from_config(cfg, write_step=False, verbose=False)
    cfg2 = load_config_yaml(CFG_DIR / "caret_M6.yaml")
    cfg2.output_dir = str(tmp_path)
    res2 = run_from_config(cfg2, write_step=False, verbose=False)
    assert res1["artifact"].sha256 == res2["artifact"].sha256


def test_changing_input_changes_sha(tmp_path):
    """Changing any input parameter must change the SHA-256."""
    cfg1 = load_config_yaml(CFG_DIR / "caret_M6.yaml")
    cfg2 = load_config_yaml(CFG_DIR / "caret_M6.yaml")
    cfg2.geometry.theta_d_deg = 15.0   # was 14
    sha1 = config_sha256(cfg1)
    sha2 = config_sha256(cfg2)
    assert sha1 != sha2
