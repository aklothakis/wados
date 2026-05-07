"""Closed-form oblique-shock relations (PSWR-1 §5.2, §5.6).

All angles in radians unless suffixed ``_deg``. Perfect-gas (gamma=1.4 default).
"""

from __future__ import annotations

import math
import numpy as np
from scipy.optimize import brentq


# ----------------------------------------------------------------------
#  Mach-angle / detachment limits
# ----------------------------------------------------------------------

def mach_angle(M: float) -> float:
    """Mach angle mu = arcsin(1/M). Lower bound on a valid shock angle."""
    if M <= 1.0:
        raise ValueError(f"Mach angle undefined for M={M} (must be > 1)")
    return math.asin(1.0 / M)


def detachment_beta(M: float, gamma: float = 1.4) -> float:
    """Shock angle beta at which theta(beta;M) is maximised (detachment limit)."""
    mu = mach_angle(M)

    def neg_theta(beta: float) -> float:
        return -theta_from_beta_M(beta, M, gamma)

    # Search in (mu, pi/2) — theta is 0 at endpoints, peaks somewhere inside
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(neg_theta, bounds=(mu + 1e-6, math.pi/2 - 1e-6),
                          method="bounded",
                          options={"xatol": 1e-10})
    return float(res.x)


# ----------------------------------------------------------------------
#  Theta-Beta-M
# ----------------------------------------------------------------------

def theta_from_beta_M(beta: float, M: float, gamma: float = 1.4) -> float:
    """Wedge half-angle theta given shock angle beta and freestream M.

    tan theta = 2 cot beta * (M^2 sin^2 beta - 1) / (M^2 (gamma + cos 2 beta) + 2)
    """
    s = math.sin(beta)
    num = 2.0 / math.tan(beta) * (M*M * s*s - 1.0)
    den = M*M * (gamma + math.cos(2.0 * beta)) + 2.0
    if den <= 0.0:
        return 0.0
    return math.atan(num / den)


def theta_from_beta_M_array(beta: np.ndarray, M: float,
                            gamma: float = 1.4) -> np.ndarray:
    """Vectorised theta_from_beta_M for an array of beta values."""
    s = np.sin(beta)
    num = 2.0 / np.tan(beta) * (M*M * s*s - 1.0)
    den = M*M * (gamma + np.cos(2.0 * beta)) + 2.0
    return np.arctan(num / den)


def beta_from_theta_M(theta: float, M: float, gamma: float = 1.4,
                      weak: bool = True) -> float:
    """Invert theta-beta-M (weak-shock branch by default)."""
    if theta <= 0.0:
        return mach_angle(M)
    beta_max = detachment_beta(M, gamma)
    theta_max = theta_from_beta_M(beta_max, M, gamma)
    if theta > theta_max:
        raise ValueError(
            f"Detached shock: theta={math.degrees(theta):.3f}deg "
            f">= theta_max={math.degrees(theta_max):.3f}deg at M={M}")
    mu = mach_angle(M)
    if weak:
        lo, hi = mu + 1e-10, beta_max - 1e-10
    else:
        lo, hi = beta_max + 1e-10, math.pi/2 - 1e-10
    return brentq(lambda b: theta_from_beta_M(b, M, gamma) - theta, lo, hi,
                  xtol=1e-12)


# ----------------------------------------------------------------------
#  Rankine-Hugoniot post-shock state
# ----------------------------------------------------------------------

def rankine_hugoniot(M_inf: float, beta: float, p_inf: float, T_inf: float,
                     gamma: float = 1.4) -> dict:
    """Return post-shock state {p2, T2, rho2, M2, M_n1, M_n2, theta} for a 2D
    oblique shock at angle ``beta`` (rad) in M_inf, p_inf, T_inf freestream.

    Density ratio uses ideal gas with R_specific=287 J/(kg K) for rho_inf.
    """
    R = 287.0  # J/(kg K), air
    rho_inf = p_inf / (R * T_inf)

    M_n1 = M_inf * math.sin(beta)
    if M_n1 <= 1.0:
        raise ValueError(
            f"Subsonic normal Mach M_n1={M_n1:.4f}: shock will not form")

    p_ratio = 1.0 + (2.0 * gamma / (gamma + 1.0)) * (M_n1*M_n1 - 1.0)
    rho_ratio = (gamma + 1.0) * M_n1*M_n1 / ((gamma - 1.0) * M_n1*M_n1 + 2.0)
    T_ratio = p_ratio / rho_ratio

    M_n2_sq = ((gamma - 1.0) * M_n1*M_n1 + 2.0) / (2.0 * gamma * M_n1*M_n1
                                                    - (gamma - 1.0))
    M_n2 = math.sqrt(M_n2_sq)

    theta = theta_from_beta_M(beta, M_inf, gamma)
    M2 = M_n2 / math.sin(beta - theta)

    return {
        "p2": p_inf * p_ratio,
        "T2": T_inf * T_ratio,
        "rho2": rho_inf * rho_ratio,
        "M2": M2,
        "M_n1": M_n1,
        "M_n2": M_n2,
        "theta": theta,
        "p_ratio": p_ratio,
        "rho_ratio": rho_ratio,
        "T_ratio": T_ratio,
    }


# ----------------------------------------------------------------------
#  Inviscid lower-surface pressure coefficient
# ----------------------------------------------------------------------

def cp_lower_wedge(M_inf: float, beta: float, gamma: float = 1.4) -> float:
    """Pressure coefficient on the wedge lower surface.

    C_p = 4 (M^2 sin^2 beta - 1) / ((gamma + 1) M^2)

    PSWR-1 §5.6, eq. (Cp,low). Same as the Rayleigh / wedge result.
    """
    s = math.sin(beta)
    return 4.0 * (M_inf*M_inf * s*s - 1.0) / ((gamma + 1.0) * M_inf*M_inf)


# ----------------------------------------------------------------------
#  Saha-onset / temperature-targeted shock-angle helpers
# ----------------------------------------------------------------------

def beta_for_T_post(M_inf: float, T_inf: float, T_post_target: float,
                    gamma: float = 1.4) -> float:
    """Inverse Rankine-Hugoniot: find beta such that T_post = T_target.

    Returns the weak-shock beta in radians. Returns NaN if no real beta in
    (mu, beta_detach) satisfies the target (e.g. M_inf too low for the
    requested T_post).
    """
    mu = mach_angle(M_inf)
    bdet = detachment_beta(M_inf, gamma)

    def f(beta: float) -> float:
        rh = rankine_hugoniot(M_inf, beta, p_inf=1.0, T_inf=T_inf, gamma=gamma)
        return rh["T2"] - T_post_target

    a, b = mu + 1e-6, bdet - 1e-6
    fa, fb = f(a), f(b)
    if fa * fb > 0.0:
        return float("nan")
    return brentq(f, a, b, xtol=1e-9)


def saha_onset_beta(M_inf: float, T_inf: float, gamma: float = 1.4) -> float:
    """Shock angle at which Saha ionization becomes measurable (T_post = 2500 K)."""
    return beta_for_T_post(M_inf, T_inf, 2500.0, gamma)


def saha_strong_beta(M_inf: float, T_inf: float, gamma: float = 1.4) -> float:
    """Shock angle at strong-ionization regime (T_post = 3500 K)."""
    return beta_for_T_post(M_inf, T_inf, 3500.0, gamma)


def suggest_beta_knots(M_inf: float, T_inf: float, gamma: float = 1.4,
                       mode: str = "transition") -> tuple:
    """Return (beta0, beta1, beta2) in degrees suggested for a given regime.

    mode = ``'no_plasma'``  : all three knots ~5 deg below Saha onset
           ``'full_plasma'`` : all three at the strong-ionization point
           ``'transition'`` : centerline cool, tip in the transition window
                              (the only regime where geometric sheath shaping
                              is non-degenerate)
    """
    mu = math.degrees(mach_angle(M_inf))
    bdet = math.degrees(detachment_beta(M_inf, gamma))
    onset = math.degrees(saha_onset_beta(M_inf, T_inf, gamma))
    strong = math.degrees(saha_strong_beta(M_inf, T_inf, gamma))

    def _clip(b):
        return max(mu + 1.0, min(bdet - 1.0, b))

    # 'cruise' is plasma-independent — always returns 12/14/16 deg clamped
    # to the valid (mu, beta_detach) window.
    if mode == "cruise":
        return _clip(12.0), _clip(14.0), _clip(16.0)

    if math.isnan(onset):
        # No plasma reachable at this Mach for the plasma-relative modes;
        # fall back to a mid-range bracket between Mach angle and detachment.
        mid = 0.5 * (mu + bdet)
        return _clip(mid - 4), _clip(mid), _clip(mid + 4)

    if mode == "no_plasma":
        b = _clip(onset - 5.0)
        return b, b, b
    if mode == "full_plasma":
        b = _clip(strong if not math.isnan(strong) else onset + 4.0)
        return b, b, b
    # transition (default)
    b0 = _clip(onset - 5.0)
    b1 = _clip(onset)
    b2 = _clip(onset + 4.0)
    return b0, b1, b2
