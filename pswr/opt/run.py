"""NSGA-II driver for the PSWR-1 problem (Phase 4 pilot)."""

from __future__ import annotations

import math
import time
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.optimize import minimize
    from pymoo.core.callback import Callback
except ImportError as e:
    raise ImportError("pymoo required for PSWR-1 Phase 4") from e

from .problem import PSWRProblem


@dataclass
class ParetoResult:
    X: np.ndarray            # design vectors (n_pareto, 4)  [radians]
    F: np.ndarray            # objectives (n_pareto, 3)
    G: np.ndarray            # constraints (n_pareto, 4)
    n_eval: int
    n_feasible: int
    wall_time_s: float
    history_F: list          # per-generation full-population F arrays
    pop_size: int
    n_gen: int
    seed: int


class _ProgressCallback(Callback):
    """Records per-generation population objectives + a user progress hook."""

    def __init__(self, on_gen=None):
        super().__init__()
        self.history_F: list = []
        self.history_n_feas: list = []
        self.on_gen = on_gen

    def notify(self, algorithm):
        F = np.copy(algorithm.pop.get("F"))
        G = algorithm.pop.get("G")
        n_feas = int(np.sum(np.all(G <= 0, axis=1))) if G is not None else 0
        self.history_F.append(F)
        self.history_n_feas.append(n_feas)
        if self.on_gen is not None:
            self.on_gen(algorithm.n_gen, n_feas, F)


def run_nsga2_pilot(problem: PSWRProblem, *,
                    pop_size: int = 20,
                    n_gen: int = 20,
                    seed: int = 20260503,
                    sbx_eta: float = 15.0,
                    pm_eta: float = 20.0,
                    on_gen=None,
                    verbose: bool = False) -> ParetoResult:
    """Run an NSGA-II optimization on the supplied PSWRProblem.

    on_gen(gen, n_feasible, pop_F) is invoked after each generation if given,
    enabling GUI progress bars.
    """
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(eta=sbx_eta),
        mutation=PM(eta=pm_eta),
        eliminate_duplicates=True,
    )
    cb = _ProgressCallback(on_gen=on_gen)

    t0 = time.perf_counter()
    res = minimize(
        problem, algorithm,
        ("n_gen", n_gen),
        seed=seed, verbose=verbose,
        callback=cb, save_history=False,
    )
    dt = time.perf_counter() - t0

    if res.X is None:
        # No non-dominated solutions found at all
        return ParetoResult(
            X=np.zeros((0, 4)), F=np.zeros((0, 3)), G=np.zeros((0, 4)),
            n_eval=problem.eval_count,
            n_feasible=problem.feasible_count,
            wall_time_s=dt, history_F=cb.history_F,
            pop_size=pop_size, n_gen=n_gen, seed=seed,
        )

    X = np.atleast_2d(res.X)
    F = np.atleast_2d(res.F)
    G = res.G if res.G is not None else np.zeros((X.shape[0], 4))
    G = np.atleast_2d(G)

    return ParetoResult(
        X=X, F=F, G=G,
        n_eval=problem.eval_count,
        n_feasible=problem.feasible_count,
        wall_time_s=dt, history_F=cb.history_F,
        pop_size=pop_size, n_gen=n_gen, seed=seed,
    )


def save_pareto(result: ParetoResult, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(result, f)


def load_pareto(path: str | Path) -> ParetoResult:
    with open(path, "rb") as f:
        return pickle.load(f)
