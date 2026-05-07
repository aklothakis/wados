"""PySAGAS-based aerodynamic evaluator for VMPLO.

Spec reference: VMPLO_implementation_prompt.md ``vmplo/aero.py``.

Optional: guarded behind ``PYSAGAS_AVAILABLE``.  Not used by the
Waverider tab; stubbed so :mod:`optimizer` can import cleanly and
future ``vmplo_optimization_tab`` integration can wire it up.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

try:
    from pysagas.cfd import OPM   # noqa: F401
    from pysagas.flow import FlowState   # noqa: F401
    from pysagas.geometry.parsers import MeshIO   # noqa: F401
    PYSAGAS_AVAILABLE = True
except Exception:
    PYSAGAS_AVAILABLE = False


class VMPLOAeroEvaluator:
    """Evaluate CL, CD, L/D, CM via PySAGAS Oblique Panel Method.

    Parameters
    ----------
    waverider : VMPLOWaverider
    gamma : float, optional
    """

    def __init__(self, waverider, gamma: float = 1.4):
        if not PYSAGAS_AVAILABLE:
            raise ImportError(
                "PySAGAS is not installed; VMPLOAeroEvaluator requires it.")
        self.waverider = waverider
        self.gamma = float(gamma)

    # ------------------------------------------------------------------ #
    #  Core evaluation                                                    #
    # ------------------------------------------------------------------ #

    def evaluate(self, Ma: float, alpha_deg: float = 0.0) -> dict:
        """Run OPM at ``Ma`` and ``alpha_deg``; return CL/CD/L_D/CM."""
        from pysagas.cfd import OPM
        from pysagas.flow import FlowState
        from pysagas.geometry.parsers import MeshIO

        # Export a temporary STL
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
            stl_path = f.name
        try:
            self.waverider.export_stl(stl_path, mirror=True)
            flow = FlowState(mach=float(Ma), aoa=float(alpha_deg))
            mesh = MeshIO.read_stl(stl_path)
            opm = OPM(mesh=mesh, flow=flow, gamma=self.gamma)
            result = opm.solve()
        finally:
            if os.path.exists(stl_path):
                os.unlink(stl_path)

        # PySAGAS returns force/moment coefficients; different versions
        # use slightly different attribute names.  Extract defensively.
        def _get(*names, default=0.0):
            for n in names:
                if hasattr(result, n):
                    return float(getattr(result, n))
            return default

        CL = _get("CL", "cl")
        CD = _get("CD", "cd")
        CM = _get("Cm", "CM", "cm", default=0.0)
        LoD = CL / CD if abs(CD) > 1e-9 else 0.0
        return {"CL": CL, "CD": CD, "L_D": LoD, "CM": CM}

    def evaluate_multi_mach(self, Ma_list, alpha_deg: float = 0.0,
                            weights=None) -> dict:
        Ma_list = list(Ma_list)
        if weights is None:
            weights = [1.0 / len(Ma_list)] * len(Ma_list)
        per: list[dict] = [self.evaluate(Ma, alpha_deg) for Ma in Ma_list]
        ld = [p["L_D"] for p in per]
        cl = [p["CL"] for p in per]
        cd = [p["CD"] for p in per]
        return {
            "L_D_weighted": float(sum(w * x for w, x in zip(weights, ld))),
            "L_D_per_mach": ld,
            "CL_per_mach":  cl,
            "CD_per_mach":  cd,
        }

    def objectives(self, Ma_list, weights=None) -> tuple[float, float]:
        """(eta, L/D_weighted) — the two VMPLO optimisation objectives."""
        eta = float(self.waverider.volumetric_efficiency("corda"))
        multi = self.evaluate_multi_mach(Ma_list, weights=weights)
        return eta, multi["L_D_weighted"]
