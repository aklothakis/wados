"""NSGA-II optimiser wrapper for VMPLO.

Spec reference: VMPLO_implementation_prompt.md ``vmplo/optimizer.py``.

Scaffolded per spec.  Not exercised by the Waverider tab; the separate
``vmplo_optimization_tab.py`` is expected to drive this once the
Waverider tab has converged.  Requires pymoo.

The multi-objective problem maximises:
  * eta  — Corda volumetric efficiency (V^{2/3} / S_planform)
  * L/D  — weighted average across ``Ma_eval``

which in minimisation form is ``f1 = -eta``, ``f2 = -L/D_weighted``.
"""

from __future__ import annotations

import numpy as np

from waverider_generator.vmplo.design_space import (
    DEFAULT_PARAMS, BOUNDS, default_design_vector,
    unpack_design_vector, check_feasibility,
)
from waverider_generator.vmplo.bspline import BSpline1D
from waverider_generator.vmplo.osculating import OsculatingAssembly
from waverider_generator.vmplo.geometry import VMPLOWaverider

try:
    from pymoo.core.problem import Problem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.optimize import minimize
    PYMOO_AVAILABLE = True
except Exception:
    Problem = object   # sentinel so the class body below parses
    PYMOO_AVAILABLE = False


class VMPLOOptimizer:
    """NSGA-II driver over the VMPLO design vector."""

    def __init__(self, params: dict | None = None):
        if not PYMOO_AVAILABLE:
            raise ImportError(
                "pymoo is not installed; VMPLOOptimizer requires it.")
        self.params = params if params is not None else DEFAULT_PARAMS

    # ------------------------------------------------------------------ #
    #  Assembly and objective                                             #
    # ------------------------------------------------------------------ #

    def _build_waverider(self, x: np.ndarray) -> VMPLOWaverider:
        p = self.params
        beta, Ma_c, n_c, ICC_c = unpack_design_vector(
            x, p["n_knots_Ma"], p["n_knots_n"], p["n_knots_ICC"])
        Ma_sp = BSpline1D(0.0, p["W"], p["n_knots_Ma"]
                          ).from_coefficients(Ma_c)
        n_sp = BSpline1D(0.0, p["W"], p["n_knots_n"]
                         ).from_coefficients(n_c)
        icc_sp = BSpline1D(0.0, p["W"], p["n_knots_ICC"]
                           ).from_coefficients(ICC_c)
        assembly = OsculatingAssembly(
            Ma_spline=Ma_sp, n_spline=n_sp,
            ICC_spline=icc_sp, US_spline=None,
            beta_design=beta,
            L=p["L"], W=p["W"], H=p["H"], x_LE=p["x_LE"],
            gamma=p["gamma"],
        )
        return VMPLOWaverider(assembly, n_planes=25, n_streamwise=40)

    def objective_function(self, x: np.ndarray) -> tuple[float, float, bool]:
        feasible = all(ok for _, _, ok in check_feasibility(x, self.params))
        if not feasible:
            return 1e3, 1e3, False
        try:
            wv = self._build_waverider(x)
            eta = wv.volumetric_efficiency("corda")
        except Exception:
            return 1e3, 1e3, False

        # Without aero.py we'd need PySAGAS; default to 0.0 for L/D so
        # NSGA-II can still run on geometry alone.
        try:
            from waverider_generator.vmplo.aero import (
                VMPLOAeroEvaluator, PYSAGAS_AVAILABLE,
            )
            if PYSAGAS_AVAILABLE:
                ev = VMPLOAeroEvaluator(wv, gamma=self.params["gamma"])
                _, LoD = ev.objectives(
                    self.params["Ma_eval"], self.params["Ma_weights"])
            else:
                LoD = 0.0
        except Exception:
            LoD = 0.0

        return -float(eta), -float(LoD), True

    # ------------------------------------------------------------------ #
    #  pymoo driver                                                       #
    # ------------------------------------------------------------------ #

    def run_nsga2(self, n_gen: int = 100, pop_size: int = 50,
                  seed: int = 42, callback=None) -> "ParetoResult":
        opt = self

        class _VMPLOProblem(Problem):
            def __init__(self):
                # Use design-vector bounds: one (beta) + 3 spline coeffs
                x0 = default_design_vector(opt.params)
                n_var = x0.size
                xl = np.empty(n_var)
                xu = np.empty(n_var)
                p = opt.params
                n_ma = p["n_knots_Ma"] + 4
                n_n = p["n_knots_n"] + 4
                n_icc = p["n_knots_ICC"] + 4
                # beta
                xl[0], xu[0] = BOUNDS["beta_design"]
                i = 1
                xl[i:i + n_ma], xu[i:i + n_ma] = BOUNDS["Ma"]; i += n_ma
                xl[i:i + n_n], xu[i:i + n_n] = BOUNDS["n"]; i += n_n
                xl[i:i + n_icc] = BOUNDS["y_ICC_norm"][0] * p["H"]
                xu[i:i + n_icc] = BOUNDS["y_ICC_norm"][1] * p["H"]
                super().__init__(n_var=n_var, n_obj=2, n_constr=0,
                                 xl=xl, xu=xu)

            def _evaluate(self, X, out, *args, **kwargs):
                fs = np.empty((X.shape[0], 2))
                for k in range(X.shape[0]):
                    f1, f2, _ = opt.objective_function(X[k])
                    fs[k] = (f1, f2)
                out["F"] = fs

        problem = _VMPLOProblem()
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(eta=15, prob=0.9),
            mutation=PM(eta=20),
            eliminate_duplicates=True,
        )
        res = minimize(problem, algorithm, ("n_gen", n_gen),
                       seed=seed, verbose=False)
        # Optional: invoke callback at the end; per-generation callbacks
        # require a custom pymoo Callback subclass — omitted for now.
        if callable(callback):
            try:
                callback(n_gen,
                         -res.F[:, 0] if res.F is not None else np.array([]),
                         -res.F[:, 1] if res.F is not None else np.array([]))
            except Exception:
                pass
        return ParetoResult(res)


class ParetoResult:
    def __init__(self, pymoo_result):
        self._r = pymoo_result

    def pareto_front(self):
        F = self._r.F
        if F is None:
            return np.array([]), np.array([])
        return -F[:, 0], -F[:, 1]

    def best_volume(self):
        eta, _ = self.pareto_front()
        if eta.size == 0:
            return None
        i = int(np.argmax(eta))
        return self._r.X[i]

    def best_ld(self):
        _, ld = self.pareto_front()
        if ld.size == 0:
            return None
        i = int(np.argmax(ld))
        return self._r.X[i]

    def knee_point(self):
        eta, ld = self.pareto_front()
        if eta.size == 0:
            return None
        en = (eta - eta.min()) / max(eta.ptp(), 1e-12)
        ln = (ld - ld.min()) / max(ld.ptp(), 1e-12)
        dist = np.sqrt((1 - en) ** 2 + (1 - ld) ** 2)
        return self._r.X[int(np.argmin(dist))]

    def to_dataframe(self):
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for to_dataframe().")
        eta, ld = self.pareto_front()
        if eta.size == 0:
            return pd.DataFrame()
        df = pd.DataFrame({"eta": eta, "L_D": ld})
        for k in range(self._r.X.shape[1]):
            df[f"x{k}"] = self._r.X[:, k]
        return df
