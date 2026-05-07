"""Oswatitsch equal-strength multi-shock ramp solver (GVWD §4.1, §5.1 DoD).

Oswatitsch's principle (Oswatitsch 1944 / NACA TM-1140 1947): for an
external compression inlet using ``n`` oblique shocks plus a terminal
normal shock, total-pressure recovery is maximised when each oblique
shock has the SAME normal-Mach number (``M_n*``).

Given (M_inf, n, gamma), find:
  - the equal M_n*
  - the cumulative deflection angles delta_1, delta_2, ..., delta_n
    (cumulative = body-fixed surface angle at each ramp end)
  - the per-shock pressure ratios and the cumulative total-pressure
    recovery pi_OS = prod(p02/p01)_i over the n oblique shocks

This module solves for delta given a target terminal Mach (default
M_terminal = 1.0, i.e. flow is just sonic at the end of the n obliques,
ready for the terminal normal shock). Calling the solver with a desired
total deflection delta_total instead is also supported.

Reference: Heiser & Pratt §6.3.4; Anderson "Modern Compressible Flow"
Ch. 9; Oswatitsch 1944.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

from scipy.optimize import brentq

from .oblique_shock import (
    obtain_beta, theta_from_beta_M, mach_angle, stagnation_pressure_ratio,
    rankine_hugoniot,
)


@dataclass
class OswatitschResult:
    """Output of the equal-strength solver.

    Attributes
    ----------
    M_n_star : float
        Common normal-Mach component of all n oblique shocks.
    M_inf : float
        Freestream Mach number.
    n : int
        Number of oblique shocks (excluding the terminal normal shock).
    deltas_cum_deg : list[float]
        Cumulative ramp angles (body-fixed surface angles) in degrees,
        ramp_1, ramp_1+ramp_2, ...
    deltas_inc_deg : list[float]
        Per-ramp incremental deflections in degrees.
    betas_deg : list[float]
        Per-shock wave angles (in the local frame) in degrees.
    machs_after : list[float]
        Mach number after each shock (M_inf, M_after_1, ..., M_after_n).
    p02_p01_per_shock : list[float]
        Total-pressure recovery across each oblique shock.
    pi_OS : float
        Cumulative recovery across all n oblique shocks.
    pi_OS_with_normal : float
        Recovery if a terminal normal shock at M_after_n is included.
    """
    M_n_star: float
    M_inf: float
    n: int
    deltas_cum_deg: List[float]
    deltas_inc_deg: List[float]
    betas_deg: List[float]
    machs_after: List[float]
    p02_p01_per_shock: List[float]
    pi_OS: float
    pi_OS_with_normal: float


def _trace_n_shocks(M_n_star: float, M_inf: float, n: int,
                    gamma: float) -> dict:
    """Trace ``n`` oblique shocks of equal normal-Mach M_n_star starting
    from freestream M_inf. Returns intermediate Mach numbers, betas,
    deflections, and per-shock recoveries.
    """
    if M_n_star <= 1.0:
        raise ValueError(f"M_n_star must be > 1; got {M_n_star}")
    M = M_inf
    betas: List[float] = []
    deltas_inc: List[float] = []
    M_after: List[float] = [M_inf]
    recovery: List[float] = []
    for _ in range(n):
        # Equal-strength condition: this shock has M_n_star
        sin_b = M_n_star / M
        if sin_b >= 1.0:
            raise ValueError(
                f"M_n_star ({M_n_star:.4f}) > local M ({M:.4f}); "
                f"cannot maintain equal strength with this many ramps"
            )
        beta = math.asin(sin_b)
        delta_inc = theta_from_beta_M(beta, M, gamma)
        if delta_inc <= 0.0:
            raise ValueError(
                f"non-positive incremental deflection at shock {len(betas)+1}"
            )
        # Post-shock Mach
        rh = rankine_hugoniot(M, beta, p_inf=1.0, T_inf=1.0, gamma=gamma)
        M_post = rh["M2"]
        # Total-pressure recovery
        p02_p01 = stagnation_pressure_ratio(M, beta, gamma)

        betas.append(beta)
        deltas_inc.append(delta_inc)
        M_after.append(M_post)
        recovery.append(p02_p01)
        M = M_post
    return {
        "betas": betas,
        "deltas_inc": deltas_inc,
        "M_after": M_after,
        "recovery": recovery,
    }


def equal_strength_ramps(M_inf: float, n: int, *,
                          gamma: float = 1.4,
                          M_terminal: float = 1.0,
                          delta_total_deg: Optional[float] = None,
                          ) -> OswatitschResult:
    """Solve for the equal-strength multi-ramp configuration.

    Two solve modes:
      - default: pick M_n_star such that the Mach AFTER the n-th oblique
        shock equals ``M_terminal`` (default 1.0, ready for terminal normal).
      - if ``delta_total_deg`` is supplied: pick M_n_star such that the
        CUMULATIVE deflection after n ramps equals delta_total_deg. This
        is the inlet-design closure when the user specifies the total
        turning angle rather than the terminal Mach.

    Parameters
    ----------
    M_inf : float
        Freestream Mach number.
    n : int
        Number of equal-strength oblique shocks (>= 1).
    gamma : float, default 1.4
    M_terminal : float, default 1.0
        Target Mach after the n-th oblique shock (used when
        ``delta_total_deg`` is None).
    delta_total_deg : float or None
        Optional total-deflection target. If given, ``M_terminal`` is
        ignored and M_n_star is solved to match this total turning.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1; got {n}")
    mu = mach_angle(M_inf)
    # M_n_star is bounded below by 1 (otherwise no shock) and above by
    # M_inf * sin(beta_max(M_inf)) — past this point we'd be on the strong
    # branch (or detached) and theta_from_beta_M is non-monotonic in M_n.
    from .oblique_shock import detachment_beta
    bdet = detachment_beta(M_inf, gamma)
    lo = 1.0 + 1e-6
    hi = M_inf * math.sin(bdet) - 1e-6

    if delta_total_deg is not None:
        target = math.radians(delta_total_deg)

        def residual(Mn: float) -> float:
            try:
                tr = _trace_n_shocks(Mn, M_inf, n, gamma)
            except ValueError:
                return 1e3
            return sum(tr["deltas_inc"]) - target
    else:
        target_M = float(M_terminal)

        def residual(Mn: float) -> float:
            try:
                tr = _trace_n_shocks(Mn, M_inf, n, gamma)
            except ValueError:
                return -1e3
            return tr["M_after"][-1] - target_M

    f_lo = residual(lo); f_hi = residual(hi)
    if f_lo * f_hi > 0:
        raise ValueError(
            f"no solution found in M_n* in ({lo}, {hi}); "
            f"residual({lo})={f_lo}, residual({hi})={f_hi}"
        )
    M_n_star = float(brentq(residual, lo, hi, xtol=1e-9))

    tr = _trace_n_shocks(M_n_star, M_inf, n, gamma)
    deltas_inc_deg = [math.degrees(d) for d in tr["deltas_inc"]]
    deltas_cum_deg = []
    s = 0.0
    for d in deltas_inc_deg:
        s += d
        deltas_cum_deg.append(s)
    pi_oblique = 1.0
    for r in tr["recovery"]:
        pi_oblique *= r

    # Optional terminal normal-shock recovery at M_after[-1]
    M_term = tr["M_after"][-1]
    pi_normal = stagnation_pressure_ratio(M_term, math.pi / 2, gamma)
    return OswatitschResult(
        M_n_star=M_n_star,
        M_inf=M_inf,
        n=n,
        deltas_cum_deg=deltas_cum_deg,
        deltas_inc_deg=deltas_inc_deg,
        betas_deg=[math.degrees(b) for b in tr["betas"]],
        machs_after=tr["M_after"],
        p02_p01_per_shock=list(tr["recovery"]),
        pi_OS=pi_oblique,
        pi_OS_with_normal=pi_oblique * pi_normal,
    )
