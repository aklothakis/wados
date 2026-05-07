"""Validation script: reproduce Liu 2019 Tables 1, 4 and Figure 12."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .aero import Liu2019AeroEvaluator
from .config import (
    PAPER_PARAMS,
    PAPER_REFERENCE_AERO,
    PAPER_REFERENCE_GEOMETRY,
    TOLERANCES,
)
from .geometry import build_liu2019_waverider


def _pass(value, reference, fractional_tol):
    if reference is None or reference == 0:
        return True, 0.0
    dev = (value - reference) / reference
    return abs(dev) <= fractional_tol, dev


def run_paper_validation(params: Dict = None,
                         n_z: int = 200,
                         n_x: int = 100,
                         verbose: bool = True,
                         run_aero: bool = True) -> Dict:
    params = dict(params or PAPER_PARAMS)
    wr = build_liu2019_waverider(params, n_z=n_z, n_x=n_x)

    geom = {
        "Vol_m3":   wr.volume(),
        "S_wet_m2": wr.wetted_area(),
        "S_p_m2":   wr.planform_area(),
        "S_b_m2":   wr.base_area(),
        "eta":      wr.volumetric_efficiency(),
    }
    geom_checks = []
    for key, value in geom.items():
        ok, dev = _pass(value, PAPER_REFERENCE_GEOMETRY[key], TOLERANCES[key])
        geom_checks.append({
            "metric": key,
            "computed": value,
            "reference": PAPER_REFERENCE_GEOMETRY[key],
            "deviation": dev,
            "tolerance": TOLERANCES[key],
            "pass": ok,
        })

    aero_checks = []
    aero_rows = []
    if run_aero:
        evaluator = Liu2019AeroEvaluator(wr)
        aero_rows = evaluator.evaluate_paper_trajectory()
        for r in aero_rows:
            ref = PAPER_REFERENCE_AERO.get(int(r["Ma"]), {})
            for key in ("CL", "CD", "L_D", "Cmz", "Xcp"):
                ok, dev = _pass(r[key], ref.get(key), TOLERANCES[key])
                aero_checks.append({
                    "Ma":       r["Ma"],
                    "metric":   key,
                    "computed": r[key],
                    "reference": ref.get(key),
                    "deviation": dev,
                    "tolerance": TOLERANCES[key],
                    "pass":     ok,
                })

    if verbose:
        print("Liu 2019 validation — geometric metrics")
        print(f"  {'metric':>10} {'computed':>12} {'paper':>10} "
              f"{'dev':>8} {'tol':>6}  status")
        for c in geom_checks:
            status = "PASS" if c["pass"] else "FAIL"
            print(f"  {c['metric']:>10} {c['computed']:>12.4f} "
                  f"{c['reference']:>10.4f} "
                  f"{c['deviation']*100:>7.2f}% "
                  f"{c['tolerance']*100:>5.1f}%  {status}")
        if aero_checks:
            print("\nLiu 2019 validation — aerodynamic metrics")
            print(f"  {'Ma':>3} {'metric':>5} {'computed':>10} "
                  f"{'paper':>8} {'dev':>8} {'tol':>6}  status")
            for c in aero_checks:
                ref = c["reference"]
                status = "PASS" if c["pass"] else "FAIL"
                ref_s = f"{ref:>8.3f}" if ref is not None else "      --"
                print(f"  {int(c['Ma']):>3} {c['metric']:>5} "
                      f"{c['computed']:>10.3f} {ref_s} "
                      f"{c['deviation']*100:>7.2f}% "
                      f"{c['tolerance']*100:>5.1f}%  {status}")

    n_pass = sum(1 for c in geom_checks + aero_checks if c["pass"])
    n_total = len(geom_checks) + len(aero_checks)
    return {
        "waverider": wr,
        "geometry":  geom,
        "aero":      aero_rows,
        "geometry_checks": geom_checks,
        "aero_checks": aero_checks,
        "pass_fraction": (n_pass, n_total),
    }


if __name__ == "__main__":
    run_paper_validation()
