"""Tangent-wedge off-design pressure model (GVWD §4.3).

For each panel, treat it as a local 2-D wedge of half-angle theta_local
relative to the freestream. If theta_local <= theta_max(M_inf), apply the
attached oblique-shock Cp; otherwise fall back to modified Newtonian.

This is the Starkey & Lewis 2000 prescription for off-design analysis on
panel-method evaluators. The two regimes do NOT generally meet smoothly
at theta = theta_max (the spec's "no discontinuity > 1%" claim is only
asymptotically true at high Mach); we expose the discontinuity rather
than blend it artificially. The discontinuity is reported by the
diagnostic helper :func:`tangent_wedge_discontinuity`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np

from .oblique_shock import (
    obtain_beta, theta_max, ShockDetachedError, cp_attached_wedge,
)
from .newtonian import modified_newtonian_cp


ArrayLike = Union[float, np.ndarray]


def tangent_wedge_cp(M_inf: float, theta_local: float,
                      gamma: float = 1.4) -> Tuple[float, str]:
    """Cp on a panel at local incidence ``theta_local`` [rad].

    Returns ``(Cp, regime)`` where regime is ``'attached'``, ``'newtonian'``
    or ``'shadow'``.

    For theta_local <= 0 the panel is in the shadow region: Cp = 0,
    regime = 'shadow' (Newtonian shadow assumption).

    For 0 < theta_local <= theta_max(M_inf): attached oblique-shock Cp.
    For theta_local > theta_max(M_inf):       modified Newtonian Cp.
    """
    if theta_local <= 0.0:
        return 0.0, "shadow"
    th_max = theta_max(M_inf, gamma)
    if theta_local < th_max - 1e-9:
        try:
            beta = obtain_beta(theta_local, M_inf, gamma, weak=True)
            return cp_attached_wedge(M_inf, beta, gamma), "attached"
        except ShockDetachedError:
            pass  # fall through to Newtonian
    Cp_n = modified_newtonian_cp(M_inf, theta_local, gamma)
    return float(Cp_n), "newtonian"


def tangent_wedge_cp_array(M_inf: float, theta_local: np.ndarray,
                            gamma: float = 1.4) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised tangent-wedge Cp.

    Returns
    -------
    Cp : ndarray of same shape
    regime_code : ndarray of int (0=shadow, 1=attached, 2=newtonian)
    """
    th = np.asarray(theta_local, dtype=float)
    Cp = np.zeros_like(th)
    code = np.zeros_like(th, dtype=np.int8)  # 0 = shadow

    th_max = theta_max(M_inf, gamma)
    # Threshold theta_local at 1e-6 rad (~ 1.7e-3 deg) to avoid spurious
    # 'windward' classification of essentially-parallel panels caused by
    # floating-point noise in n . v_inf.
    windward = th > 1e-6
    high_inc = th > th_max - 1e-9

    # Newtonian for high-incidence and shadow already handled (default 0).
    mask_n = windward & high_inc
    if mask_n.any():
        Cp[mask_n] = modified_newtonian_cp(M_inf, th[mask_n], gamma)
        code[mask_n] = 2

    # Attached for moderate-incidence windward
    mask_a = windward & (~high_inc)
    if mask_a.any():
        # Loop — obtain_beta is scalar; vectorising scipy.brentq is messy
        Cp_a = np.empty(int(mask_a.sum()))
        idx_a = np.where(mask_a)[0]
        for k, i in enumerate(idx_a):
            try:
                beta = obtain_beta(float(th[i]), M_inf, gamma, weak=True)
                Cp_a[k] = cp_attached_wedge(M_inf, beta, gamma)
            except ShockDetachedError:
                # Fallback to Newtonian (corner case at theta_max)
                Cp_a[k] = modified_newtonian_cp(M_inf, float(th[i]), gamma)
        Cp[mask_a] = Cp_a
        code[mask_a] = 1

    return Cp, code


def tangent_wedge_discontinuity(M_inf: float, gamma: float = 1.4) -> dict:
    """Diagnostic: jump in Cp at theta = theta_max(M_inf).

    Returns dict with attached/newtonian Cp values just below and above
    theta_max, and the relative jump |Cp_jump| / Cp_attached.
    """
    th_max = theta_max(M_inf, gamma)
    th_below = th_max * 0.999
    th_above = th_max * 1.001
    beta = obtain_beta(th_below, M_inf, gamma, weak=True)
    Cp_attached = cp_attached_wedge(M_inf, beta, gamma)
    Cp_newt = float(modified_newtonian_cp(M_inf, th_above, gamma))
    return {
        "M": M_inf,
        "theta_max_deg": math.degrees(th_max),
        "Cp_attached_at_theta_max": Cp_attached,
        "Cp_newtonian_at_theta_max": Cp_newt,
        "jump": Cp_newt - Cp_attached,
        "rel_jump": (Cp_newt - Cp_attached) / max(abs(Cp_attached), 1e-30),
    }
