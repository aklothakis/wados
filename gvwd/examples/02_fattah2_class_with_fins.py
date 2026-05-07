#!/usr/bin/env python
"""GVWD example 02: Fattah-2-class flat-bottom + 4 fins.

Smaller body (L_total = 1.8 m) with 4 fins at 45 deg dihedral
(X-tail). Demonstrates the body+fins merged-mesh pipeline.
"""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from gvwd.examples._runner import run_from_config

if __name__ == "__main__":
    cfg_path = HERE / "configs" / "engineering_flat_fattah2_class.yaml"
    run_from_config(cfg_path, write_step=True)
