"""pymoo Problem subclass wrapping :func:`evaluate_design` (PSWR-1 §5.7).

Design vector x = (beta0, beta1, beta2, Lambda) in radians.
Objectives F = (-L/D, max sigma_b [dBsm], -eta_V).
Constraints G = (g1 detachment, g2 mach-angle, g3 Fay-Riddell, g4 Born).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

try:
    from pymoo.core.problem import ElementwiseProblem
except ImportError as e:
    raise ImportError(
        "pymoo is required for PSWR-1 Phase 4. Install via `pip install pymoo`."
    ) from e

from .objectives import evaluate_design, _BIG


# ----------------------------------------------------------------------
#  Configuration container
# ----------------------------------------------------------------------

@dataclass
class PSWRConfig:
    """All Phase-4 evaluation parameters in one place.

    Defaults match PSWR-1 §8 (M_inf=15 since the M=6 baseline does not have
    plasma — see Phase 2 report).
    """
    M_inf: float = 15.0
    body_length: float = 10.0
    T_w: float = 1500.0
    p_inf: float = 1197.03      # Pa, US Std 30 km
    T_inf: float = 226.65       # K
    f0_Hz: float = 10.0e9       # X-band
    R_LE: float = 1.0e-3        # 1 mm
    # NOTE: spec §8 default is 50 MW/m^2, but for M_inf=15 at 30 km with a
    # 1 mm LE the Fay-Riddell prediction is ~4e11 W/m^2 — the spec gate is
    # achievable only for moderate-Mach (~6-8) re-entry-style trajectories.
    # For the M=15 plasma-relevant pilot we relax to 1e13 W/m^2 so g3 does
    # not dominate; tighten in Phase 5 with realistic R_LE / M_inf coupling.
    q_LE_max: float = 1.0e13
    bistatic_angles_deg: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0.0, 0.0), (90.0, 0.0), (180.0, 0.0)]
    )
    k_i_hat: Tuple[float, float, float] = (-1.0, 0.0, 0.0)

    # Resolution
    n_span_geom: int = 41
    n_chord_geom: int = 20
    n_span_grid: int = 21
    n_chord_grid: int = 30
    n_normal: int = 10

    # Design variable bounds (degrees, converted to radians on use)
    beta_lo_deg: float = 8.0
    beta_hi_deg: float = 35.0
    lambda_lo_deg: float = 55.0
    lambda_hi_deg: float = 80.0
    # Flat-nose fraction X1 (5th design variable). Range [0, 0.4] — 0 is sharp
    # apex, 0.4 is a fairly aggressive centerline blunt strip.
    flat_lo: float = 0.0
    flat_hi: float = 0.4

    # Tolerances
    margin_detach_deg: float = 1.0
    margin_mach_deg: float = 1.0
    gamma: float = 1.4


# ----------------------------------------------------------------------
#  pymoo Problem
# ----------------------------------------------------------------------

class PSWRProblem(ElementwiseProblem):
    """4-var, 3-obj, 4-constr PSWR-1 evaluation problem.

    With ``capture_archive=True`` every evaluation (including dominated and
    infeasible designs) is appended to ``archive_X``/``archive_F``/``archive_G``
    for post-hoc diagnostic plots.
    """

    def __init__(self, cfg: PSWRConfig | None = None, *,
                 capture_archive: bool = False):
        self.cfg = cfg or PSWRConfig()
        c = self.cfg
        # 5 design variables: (beta0, beta1, beta2, Lambda, X1_flat_fraction)
        xl = np.array([
            math.radians(c.beta_lo_deg),
            math.radians(c.beta_lo_deg),
            math.radians(c.beta_lo_deg),
            math.radians(c.lambda_lo_deg),
            float(c.flat_lo),
        ])
        xu = np.array([
            math.radians(c.beta_hi_deg),
            math.radians(c.beta_hi_deg),
            math.radians(c.beta_hi_deg),
            math.radians(c.lambda_hi_deg),
            float(c.flat_hi),
        ])
        super().__init__(n_var=5, n_obj=3, n_ieq_constr=4, xl=xl, xu=xu)
        # Bookkeeping
        self.eval_count = 0
        self.feasible_count = 0
        self.best_LD_so_far = -math.inf
        self.history_raw: list = []   # optional debug trace
        # Full-archive capture (every evaluation, dominated + infeasible)
        self.capture_archive = capture_archive
        self.archive_X: list = []
        self.archive_F: list = []
        self.archive_G: list = []
        # Diagnostic raw outputs (e.g. delta_BL_max, max_re_chi) — only
        # populated when capture_archive is True
        self.archive_diag: list = []

    def _evaluate(self, x, out, *args, **kwargs):
        c = self.cfg
        try:
            res = evaluate_design(
                x,
                M_inf=c.M_inf, body_length=c.body_length, T_w=c.T_w,
                p_inf=c.p_inf, T_inf=c.T_inf,
                f0_Hz=c.f0_Hz, R_LE=c.R_LE, q_LE_max=c.q_LE_max,
                bistatic_angles_deg=c.bistatic_angles_deg,
                k_i_hat=c.k_i_hat,
                n_span_geom=c.n_span_geom, n_chord_geom=c.n_chord_geom,
                n_span_grid=c.n_span_grid, n_chord_grid=c.n_chord_grid,
                n_normal=c.n_normal,
                margin_detach_rad=math.radians(c.margin_detach_deg),
                margin_mach_rad=math.radians(c.margin_mach_deg),
                gamma=c.gamma,
            )
            out["F"] = res.F
            out["G"] = res.G
            self.eval_count += 1
            if res.feasible:
                self.feasible_count += 1
                if -res.F[0] > self.best_LD_so_far:
                    self.best_LD_so_far = -res.F[0]
            if self.capture_archive:
                self.archive_X.append(np.array(x, copy=True))
                self.archive_F.append(np.array(res.F, copy=True))
                self.archive_G.append(np.array(res.G, copy=True))
                # Subset of res.raw safe to keep
                self.archive_diag.append({
                    k: res.raw.get(k) for k in
                    ("max_re_chi", "max_sigma_dBsm", "sigmas_dBsm",
                     "q_FR_MW_m2", "LD", "eta_V")
                })
        except Exception as e:
            out["F"] = np.array([_BIG, _BIG, _BIG])
            out["G"] = np.array([_BIG, _BIG, _BIG, _BIG])
            self.eval_count += 1
            if self.capture_archive:
                self.archive_X.append(np.array(x, copy=True))
                self.archive_F.append(np.array([_BIG, _BIG, _BIG]))
                self.archive_G.append(np.array([_BIG, _BIG, _BIG, _BIG]))
                self.archive_diag.append({"error": str(e)})
