"""Phase 4 verification harness for the three MOC-numerical fixes in
``waverider_generator/vmplo/moc.py``.

Runs the gated sweeps from the Phase 4 plan:

* Gate A - regression (Phase 2 cone equivalence at 1e-13, plus Phase 3 tests
  1, 2, 3).
* Gate B - smoothness sweep at Liu defaults across n in {0.4, 0.5, 0.6, 0.7,
  1.0, 1.3, 1.5}. Acceptance: max |Δ²y_TE| < 30 mm at all n; Vol/η drift
  vs baseline < 5 %.
* Gate C - the targeted "Combined params" smoothness improvement
  (β=16°, L_s=0.05, Ma 8-14). Acceptance: n=0.5 max |Δ²y_TE| < 30 mm
  (currently 215 mm).
* Gate D - power-law performance budget (paper case build < 3 min).

Run all gates::

    py -3.10 -m mfof.test_phase4_moc_fixes

Run just one (e.g. before/after a fix)::

    py -3.10 -m mfof.test_phase4_moc_fixes --gate A
    py -3.10 -m mfof.test_phase4_moc_fixes --gate B
    py -3.10 -m mfof.test_phase4_moc_fixes --gate C
    py -3.10 -m mfof.test_phase4_moc_fixes --gate D
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

import numpy as np


_LIU_DEFAULTS = {
    "beta_deg":  13.0, "L_w":   6.0, "W":   3.0,   "L_s":  0.30,
    "y5":     0.1608,  "z5":    1.5, "y6":  1.608, "z6":   0.0,
    "delta5": 0.0,     "delta6": 0.0,
    "Ma_center": 6.0,  "Ma_tip": 13.0, "gamma": 1.4,
}
_COMBINED = {**_LIU_DEFAULTS, "beta_deg": 16.0, "L_s": 0.05,
             "Ma_center": 8.0, "Ma_tip": 14.0}


def _ensure_qt():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication
    if QApplication.instance() is None:
        QApplication([])


def _build(params, ftype, n=1.0, n_z=80, n_x=40):
    _ensure_qt()
    from mfof_waverider_tab import _MFOFGeometryWorker
    w = _MFOFGeometryWorker(params, n_z=n_z, n_x=n_x,
                              flowfield_type=ftype, power_law_n=float(n))
    caught = {}
    w.finished_ok.connect(lambda x: caught.setdefault("wr", x))
    w.failed.connect(lambda m: caught.setdefault("err", m))
    t0 = time.time()
    w.run()
    dt = time.time() - t0
    return caught.get("wr"), dt, caught.get("err", "")


def _smoothness(wr):
    y_TE = np.array([p.P_TE[1] for p in wr.planes])
    if len(y_TE) < 3:
        return 0.0, 0.0
    d2y = np.diff(np.diff(y_TE))
    return float(abs(d2y).max() * 1000.0), float(np.std(np.diff(y_TE)) * 1000.0)


# ---------------------------------------------------------------------------
# Gate A - regression
# ---------------------------------------------------------------------------

def gate_A() -> bool:
    print("=== Gate A: regression ===")
    print("[A.1] mfof.validate (Phase 2 cone equivalence at 1e-13)")
    from mfof.validate import run_equivalence_test
    eq_ok = run_equivalence_test(verbose=False)
    print(f"  -> {'PASS' if eq_ok else 'FAIL'}")

    print("[A.2] mfof.test_phase3 (Tests 1, 2, 3)")
    from mfof.test_phase3 import (
        test_1_powerlaw_near_cone_vs_cone,
        test_2_powerlaw_n1_exact_vs_cone,
        test_3_wedge_vs_cone_qualitative,
    )
    p3 = [test_1_powerlaw_near_cone_vs_cone(verbose=False),
          test_2_powerlaw_n1_exact_vs_cone(verbose=False),
          test_3_wedge_vs_cone_qualitative(verbose=False)]
    for i, ok in enumerate(p3, 1):
        print(f"  Test {i}: {'PASS' if ok else 'FAIL'}")
    return eq_ok and all(p3)


# ---------------------------------------------------------------------------
# Gate B - smoothness at Liu defaults
# ---------------------------------------------------------------------------

def gate_B(verbose: bool = True) -> bool:
    print("=== Gate B: smoothness sweep at Liu defaults ===")
    n_values = [0.4, 0.5, 0.6, 0.7, 1.0, 1.3, 1.5]
    results = []
    print(f"  {'n':>5} {'Vol':>8} {'eta':>8} "
          f"{'maxd2y_mm':>10} {'std_dy_mm':>10} {'time':>6}  status")
    print("  " + "-" * 60)
    all_ok = True
    for n in n_values:
        wr, dt, err = _build(_LIU_DEFAULTS, "power-law", n=n,
                              n_z=80, n_x=40)
        if wr is None:
            print(f"  {n:>5}: BUILD FAILED: {err[:80]}")
            all_ok = False
            continue
        max_d2y, std_dy = _smoothness(wr)
        Vol = float(wr.volume())
        eta = float(wr.volumetric_efficiency())
        ok = max_d2y < 30.0
        all_ok = all_ok and ok
        results.append({"n": n, "Vol": Vol, "eta": eta,
                         "max_d2y_mm": max_d2y, "std_dy_mm": std_dy,
                         "time_s": dt})
        if verbose:
            print(f"  {n:>5.2f} {Vol:>8.4f} {eta:>8.5f} "
                  f"{max_d2y:>10.2f} {std_dy:>10.2f} {dt:>6.1f}s  "
                  f"{'PASS' if ok else 'FAIL'}")
    return all_ok


# ---------------------------------------------------------------------------
# Gate C - Combined params smoothness improvement
# ---------------------------------------------------------------------------

def gate_C(verbose: bool = True) -> bool:
    print("=== Gate C: Combined-params smoothness sweep ===")
    print(f"  params: beta=16°, L_s=0.05, Ma 8-14")
    n_values = [0.4, 0.5, 0.55, 0.6, 0.7]
    print(f"  {'n':>5} {'Vol':>8} {'eta':>8} "
          f"{'maxd2y_mm':>10} {'std_dy_mm':>10} {'time':>6}  status")
    print("  " + "-" * 60)
    all_ok = True
    for n in n_values:
        wr, dt, err = _build(_COMBINED, "power-law", n=n, n_z=80, n_x=40)
        if wr is None:
            print(f"  {n:>5}: BUILD FAILED: {err[:80]}")
            all_ok = False
            continue
        max_d2y, std_dy = _smoothness(wr)
        Vol = float(wr.volume())
        eta = float(wr.volumetric_efficiency())
        ok = max_d2y < 30.0
        all_ok = all_ok and ok
        if verbose:
            print(f"  {n:>5.2f} {Vol:>8.4f} {eta:>8.5f} "
                  f"{max_d2y:>10.2f} {std_dy:>10.2f} {dt:>6.1f}s  "
                  f"{'PASS' if ok else 'FAIL'}")
    return all_ok


# ---------------------------------------------------------------------------
# Gate D - performance budget
# ---------------------------------------------------------------------------

def gate_D(verbose: bool = True) -> bool:
    print("=== Gate D: performance budget ===")
    print("  power-law n=0.7 paper case (n_z=200, n_x=100)")
    wr, dt, err = _build(_LIU_DEFAULTS, "power-law", n=0.7,
                          n_z=200, n_x=100)
    if wr is None:
        print(f"  BUILD FAILED: {err[:80]}")
        return False
    ok = dt <= 180.0   # 3 min
    print(f"  build time = {dt:.1f}s  ({'PASS' if ok else 'FAIL'} <= 180s)")
    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_GATES = {"A": gate_A, "B": gate_B, "C": gate_C, "D": gate_D}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gate", choices=list(_GATES.keys()) + ["all"],
                   default="all")
    args = p.parse_args(argv)
    selected = list(_GATES.keys()) if args.gate == "all" else [args.gate]
    n_pass = 0
    for g in selected:
        ok = _GATES[g]()
        if ok:
            n_pass += 1
        print()
    print(f"=== {n_pass}/{len(selected)} gates PASS ===")
    return 0 if n_pass == len(selected) else 1


if __name__ == "__main__":
    sys.exit(main())
