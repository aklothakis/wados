"""Optimization submodule (PSWR-1 Phase 4).

NSGA-II multi-objective coupling of geometry + Saha + Drude + Born RCS.
"""

from .objectives import evaluate_design, fay_riddell_swept
from .problem import PSWRProblem, PSWRConfig
from .run import run_nsga2_pilot, ParetoResult

__all__ = [
    "evaluate_design", "fay_riddell_swept",
    "PSWRProblem", "PSWRConfig",
    "run_nsga2_pilot", "ParetoResult",
]
