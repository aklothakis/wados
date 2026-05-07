#!/usr/bin/env python
"""GVWD example 03: Nonweiler caret reference (M=6)."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from gvwd.examples._runner import run_from_config

if __name__ == "__main__":
    run_from_config(HERE / "configs" / "caret_M6.yaml", write_step=True)
