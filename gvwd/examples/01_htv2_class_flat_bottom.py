#!/usr/bin/env python
"""GVWD example 01: HTV-2-class flat-bottom glide vehicle.

Loads ``configs/engineering_flat_htv2_class.yaml``, builds the geometry,
runs on-design aero + a default 8x6 Mach-alpha sweep, exports STL and
STEP, and writes the full plot suite.

Usage:
    python -m gvwd.examples.01_htv2_class_flat_bottom
"""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from gvwd.examples._runner import run_from_config

if __name__ == "__main__":
    cfg_path = HERE / "configs" / "engineering_flat_htv2_class.yaml"
    run_from_config(cfg_path, write_step=True)
