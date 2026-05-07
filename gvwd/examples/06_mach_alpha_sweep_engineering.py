#!/usr/bin/env python
"""GVWD example 06: Larger-grid Mach-alpha sweep on the engineering
flat-bottom geometry, showcasing the full plot suite.

Builds the HTV-2-class geometry programmatically (not via YAML) and runs
a finer 12 x 8 sweep grid for paper-quality heatmaps.
"""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from gvwd.io.config import (
    GVWDRunConfig, EngineeringFlatConfig, FinsConfig, SweepRunConfig,
)
from gvwd.examples._runner import run_from_config


if __name__ == "__main__":
    cfg = GVWDRunConfig(
        geometry=EngineeringFlatConfig(
            M_design=15.0, theta_fore_deg=8.0, Lambda_deg=75.0,
            L_fore=2.5, L_center=1.5, b_base=0.5, h_base=0.4,
            r_LE_mm=5.0, r_nose_mm=10.0,
        ),
        fins=FinsConfig(n_fins=0),
        sweep=SweepRunConfig(
            enabled=True,
            M_grid=(5.0, 20.0, 12),
            alpha_grid_deg=(0.0, 15.0, 8),
            altitude_km=30.0, T_w=1500.0, Re_x_tr=1.0e6,
        ),
        output_dir="results",
        tag="sweep_finegrid",
    )
    run_from_config(cfg, write_step=False)
