"""Oblique-shock relations for GVWD (spec §4.1).

Most of the textbook theta-beta-M / Rankine-Hugoniot machinery is shared
with PSWR-1; we re-export from :mod:`pswr.thermo.oblique_shock` to avoid
duplication. This module then adds the spec-named primitives:

- ``ShockDetachedError`` exception
- ``obtain_beta(theta, M1, gamma)``  — weak-shock inverse, raises on detachment
- ``theta_max(M1, gamma)``           — maximum attached-shock deflection
- ``swept_oblique_shock(...)``       — Emanuel 2015 swept-LE generalization
- ``cp_attached_wedge``              — Cp on a wedge surface (alias of pswr's
                                       cp_lower_wedge for clarity in GVWD context)
- ``stagnation_pressure_ratio``      — p02/p01 across an oblique shock
"""

from __future__ import annotations

import math
from typing import Dict

# Reuse PSWR-1 primitives (audit pass per spec §5.1 DoD)
from pswr.thermo.oblique_shock import (
    mach_angle,
    detachment_beta,
    theta_from_beta_M,
    rankine_hugoniot,
    cp_lower_wedge as cp_attached_wedge,
)


# ----------------------------------------------------------------------
#  Exceptions
# ----------------------------------------------------------------------

class ShockDetachedError(ValueError):
    """Raised when no real attached-shock solution exists for the
    requested (theta, M1) pair, i.e. the requested deflection exceeds the
    maximum attached-shock turning angle theta_max(M1, gamma).
    """


# ----------------------------------------------------------------------
#  Maximum attached-shock deflection theta_max(M1, gamma)
# ----------------------------------------------------------------------

def theta_max(M1: float, gamma: float = 1.4) -> float:
    """Maximum attached-shock deflection angle [rad] for freestream Mach M1.

    Computed as the value of ``theta`` at the detachment shock angle
    ``beta_detach``, i.e. the peak of the theta-beta-M curve.
    """
    bdet = detachment_beta(M1, gamma)
    return theta_from_beta_M(bdet, M1, gamma)


# ----------------------------------------------------------------------
#  Inverse theta-beta-M
# ----------------------------------------------------------------------

def obtain_beta(theta_rad: float, M1: float, gamma: float = 1.4,
                 weak: bool = True) -> float:
    """Inverse theta-beta-M: shock angle beta [rad] for the requested
    deflection theta and freestream Mach M1.

    Default is the weak-shock root (the physically observed solution for
    most external flows). Setting ``weak=False`` returns the strong-shock
    root.

    Raises
    ------
    ShockDetachedError
        if theta exceeds the maximum attached-shock deflection theta_max(M1).
    """
    if theta_rad <= 0.0:
        return mach_angle(M1)
    bdet = detachment_beta(M1, gamma)
    th_max = theta_from_beta_M(bdet, M1, gamma)
    if theta_rad > th_max:
        raise ShockDetachedError(
            f"theta={math.degrees(theta_rad):.3f} deg exceeds "
            f"theta_max={math.degrees(th_max):.3f} deg at M={M1:.3f} "
            f"(detached shock)"
        )
    # Brentq on the requested branch
    from scipy.optimize import brentq
    mu = mach_angle(M1)
    if weak:
        lo, hi = mu + 1e-10, bdet - 1e-10
    else:
        lo, hi = bdet + 1e-10, math.pi / 2 - 1e-10
    return float(brentq(
        lambda b: theta_from_beta_M(b, M1, gamma) - theta_rad,
        lo, hi, xtol=1e-12,
    ))


# ----------------------------------------------------------------------
#  Stagnation pressure ratio across an oblique shock
# ----------------------------------------------------------------------

def stagnation_pressure_ratio(M1: float, beta_rad: float,
                               gamma: float = 1.4) -> float:
    """Total-pressure recovery p02/p01 across an oblique shock.

    Standard Rayleigh-Pitot form using normal-Mach M_n1 = M1 sin(beta).
    """
    M_n1 = M1 * math.sin(beta_rad)
    if M_n1 <= 1.0:
        return 1.0   # no shock
    M_n1_sq = M_n1 * M_n1
    A = ((gamma + 1.0) * M_n1_sq) / ((gamma - 1.0) * M_n1_sq + 2.0)
    B = (gamma + 1.0) / (2.0 * gamma * M_n1_sq - (gamma - 1.0))
    return float(A ** (gamma / (gamma - 1.0)) * B ** (1.0 / (gamma - 1.0)))


# ----------------------------------------------------------------------
#  Swept-shock generalization (Emanuel 2015)
# ----------------------------------------------------------------------

def swept_oblique_shock(theta_rad: float, M1: float,
                         Lambda_rad: float, gamma: float = 1.4,
                         p_inf: float = 1.0,
                         T_inf: float = 1.0) -> Dict[str, float]:
    """Attached-shock relations on a swept LE panel (spec §4.1).

    Per Emanuel 2015 / Anderson §3.7, the surface pressure on a
    leading-edge-swept panel is governed by the freestream component
    perpendicular to the LE:

        M_perp = M1 * cos(Lambda)

    Standard theta-beta-M and Rankine-Hugoniot are then applied on M_perp
    (not M1). The component of velocity parallel to the LE is conserved
    and does not contribute to surface pressure.

    Returns
    -------
    dict with keys:
        beta    : shock angle in the LE-perpendicular plane [rad]
        M_perp  : perpendicular freestream Mach (M1 cos Lambda)
        p2/p1, rho2/rho1, T2/T1, M2_perp, theta_eff
        Cp_surf : surface pressure coefficient (referred to total q_inf)

    Raises
    ------
    ShockDetachedError
        if M_perp <= 1 (LE too swept for an attached shock at this Mach)
        or if theta exceeds theta_max(M_perp).
    """
    cos_L = math.cos(Lambda_rad)
    M_perp = M1 * cos_L
    if M_perp <= 1.0:
        raise ShockDetachedError(
            f"swept-shock LE: M_perp = M1 cos(Lambda) = {M_perp:.3f} <= 1; "
            f"M1={M1:.3f}, Lambda={math.degrees(Lambda_rad):.2f} deg"
        )
    beta = obtain_beta(theta_rad, M_perp, gamma)
    rh = rankine_hugoniot(M_perp, beta, p_inf, T_inf, gamma)
    # Cp is referenced to *full* freestream dynamic pressure 0.5 rho1 V1^2:
    # p2 - p1 = rho1 V1_perp^2 * (rh-1 form)... but the standard result is
    # Cp = 4 (M_perp^2 sin^2 beta - 1) / ((gamma+1) M1^2)  (note M1, not M_perp)
    s = math.sin(beta)
    Cp_surf = (4.0 * (M_perp * M_perp * s * s - 1.0)
                / ((gamma + 1.0) * M1 * M1))
    return {
        "beta": beta,
        "M_perp": M_perp,
        "p_ratio": rh["p_ratio"],
        "rho_ratio": rh["rho_ratio"],
        "T_ratio": rh["T_ratio"],
        "M2_perp": rh["M2"],
        "theta_eff": theta_rad,
        "Cp_surf": Cp_surf,
    }
