"""Oblique-shock and compressible-flow relations for VMPLO.

Spec reference: VMPLO_implementation_prompt.md §5.

Everything here is pure gas-dynamics (no VMPLO-specific data): angle and
Mach conversions, Rudd-Lewis closed-form inversion of the theta-beta-Ma
relation, Newton-Raphson fallback, detachment limit, post-shock state
ratios.  gamma defaults to 1.4 throughout.
"""

from __future__ import annotations

import numpy as np


class DetachedShockError(ValueError):
    """Raised when the requested deflection exceeds ``theta_max(Ma)``."""


# ---------------------------------------------------------------------- #
#  Simple relations                                                       #
# ---------------------------------------------------------------------- #

def mach_angle(Ma: float) -> float:
    """Mach angle mu = arcsin(1/Ma) in degrees."""
    if Ma <= 1.0:
        raise ValueError(f"Subsonic or sonic Ma={Ma} has no Mach angle.")
    return float(np.degrees(np.arcsin(1.0 / Ma)))


def theta_from_beta_Ma(beta_deg: float, Ma: float,
                       gamma: float = 1.4) -> float:
    """Flow deflection angle ``theta`` (degrees) for a shock at ``beta_deg``
    and freestream ``Ma`` (weak or strong branch — formula is the same).
    """
    beta = np.radians(beta_deg)
    sinb = np.sin(beta)
    num = 2.0 / np.tan(beta) * (Ma**2 * sinb**2 - 1.0)
    den = Ma**2 * (gamma + np.cos(2.0 * beta)) + 2.0
    return float(np.degrees(np.arctan(num / den)))


# ---------------------------------------------------------------------- #
#  Detachment limit                                                       #
# ---------------------------------------------------------------------- #

def beta_detachment(Ma: float, gamma: float = 1.4) -> float:
    """Maximum attached-shock angle for given ``Ma`` (degrees).

    Analytic closed form from Anderson, *Modern Compressible Flow*.
    """
    g = gamma
    term = (g + 1.0) * (1.0 + (g - 1.0) / 2.0 * Ma**2
                         + (g + 1.0) / 16.0 * Ma**4)
    sin2 = (1.0 / (g * Ma**2)) * (
        (g + 1.0) * Ma**2 / 4.0 - 1.0 + np.sqrt(term))
    sin2 = np.clip(sin2, 0.0, 1.0)
    return float(np.degrees(np.arcsin(np.sqrt(sin2))))


def theta_max(Ma: float, gamma: float = 1.4) -> float:
    """Maximum flow deflection angle at shock detachment (degrees)."""
    b_det = beta_detachment(Ma, gamma)
    return theta_from_beta_Ma(b_det, Ma, gamma)


# ---------------------------------------------------------------------- #
#  Theta-beta-Ma inversion (weak shock branch)                            #
# ---------------------------------------------------------------------- #

def _beta_from_theta_Ma_closed(theta_deg: float, Ma: float,
                               gamma: float = 1.4) -> float:
    """Rudd-Lewis (1998) closed-form weak-branch inversion.

    Solves the Wellmann-Emanuel cubic in ``tan(beta)`` analytically via
    the Cardano trigonometric form (three real roots when
    ``theta <= theta_max``).  Returns the smallest root greater than
    ``tan(mu)`` — the weak-shock branch.
    """
    theta = np.radians(theta_deg)
    if theta <= 0.0:
        # Degenerate: Mach wave
        return mach_angle(Ma)

    # Cubic coefficients: a t^3 + b t^2 + c t + d = 0, t = tan(beta).
    a = (2.0 + (gamma - 1.0) * Ma**2) * np.tan(theta)
    b = -2.0 * (Ma**2 - 1.0)
    c = (2.0 + (gamma + 1.0) * Ma**2) * np.tan(theta)
    d = 2.0

    # Depress the cubic: substitute t = u - b/(3a), giving u^3 + p u + q = 0.
    p = (3.0 * a * c - b**2) / (3.0 * a**2)
    q = (2.0 * b**3 - 9.0 * a * b * c + 27.0 * a**2 * d) / (27.0 * a**3)

    discriminant = -(4.0 * p**3 + 27.0 * q**2)
    if discriminant < -1e-10:
        raise DetachedShockError(
            f"Detached shock: theta={theta_deg:.3f}° exceeds theta_max for Ma={Ma:.3f}.")

    # Three real roots via Cardano trigonometric form.
    m = 2.0 * np.sqrt(max(-p / 3.0, 0.0))
    if m < 1e-12:
        # Degenerate: p ~ 0 means cubic has a repeated root at u=0.
        return float(np.degrees(np.arctan(-b / (3.0 * a))))

    arg = 3.0 * q / (p * m) if abs(p) > 1e-12 else 0.0
    arg = np.clip(arg, -1.0, 1.0)
    roots_u = [m * np.cos((np.arccos(arg) - 2.0 * np.pi * k) / 3.0)
               for k in range(3)]
    roots_t = [u - b / (3.0 * a) for u in roots_u]

    tan_mu = np.tan(np.arcsin(1.0 / Ma))
    candidates = sorted(t for t in roots_t if t > tan_mu and np.isfinite(t))
    if not candidates:
        raise DetachedShockError(
            f"No weak-shock root for theta={theta_deg:.3f}°, Ma={Ma:.3f}.")
    return float(np.degrees(np.arctan(candidates[0])))


def _beta_from_theta_Ma_newton(theta_deg: float, Ma: float,
                               gamma: float = 1.4,
                               tol: float = 1e-10,
                               max_iter: int = 60) -> float:
    """Newton-Raphson inversion of theta-beta-Ma (weak branch).

    Initial guess is the midpoint between the Mach angle and the
    detachment angle, which always lies on the weak-shock branch.
    """
    theta = np.radians(theta_deg)
    b_mu = np.arcsin(1.0 / Ma)
    b_det = np.radians(beta_detachment(Ma, gamma))
    if theta > np.radians(theta_max(Ma, gamma)) + 1e-6:
        raise DetachedShockError(
            f"theta={theta_deg:.3f}° exceeds theta_max for Ma={Ma:.3f}.")
    beta = 0.5 * (b_mu + b_det)

    def f(b: float) -> float:
        sinb = np.sin(b)
        num = 2.0 / np.tan(b) * (Ma**2 * sinb**2 - 1.0)
        den = Ma**2 * (gamma + np.cos(2.0 * b)) + 2.0
        return np.tan(theta) - num / den

    for _ in range(max_iter):
        fb = f(beta)
        if abs(fb) < tol:
            break
        eps = 1e-7
        fp = (f(beta + eps) - f(beta - eps)) / (2.0 * eps)
        if abs(fp) < 1e-15:
            break
        step = fb / fp
        beta -= step
        beta = np.clip(beta, b_mu + 1e-8, b_det - 1e-8)
    return float(np.degrees(beta))


def beta_from_theta_Ma(theta_deg: float, Ma: float,
                       gamma: float = 1.4,
                       method: str = "auto") -> float:
    """Weak-branch beta from (theta, Ma).  Raises ``DetachedShockError``
    when the deflection exceeds ``theta_max(Ma)``.

    ``method``:
      * ``'closed'`` — Rudd-Lewis cubic only.
      * ``'newton'`` — Newton-Raphson only.
      * ``'auto'`` (default) — try closed form, fall back to Newton if
        that blows up within ~1° of detachment.
    """
    if method == "closed":
        return _beta_from_theta_Ma_closed(theta_deg, Ma, gamma)
    if method == "newton":
        return _beta_from_theta_Ma_newton(theta_deg, Ma, gamma)
    # auto
    try:
        b = _beta_from_theta_Ma_closed(theta_deg, Ma, gamma)
        # Sanity check via forward evaluation
        th_check = theta_from_beta_Ma(b, Ma, gamma)
        if abs(th_check - theta_deg) > 0.05:  # 3-sigma tolerance
            raise ValueError("closed-form result inconsistent")
        return b
    except Exception:
        return _beta_from_theta_Ma_newton(theta_deg, Ma, gamma)


# ---------------------------------------------------------------------- #
#  Post-shock state                                                       #
# ---------------------------------------------------------------------- #

def oblique_shock_ratios(Ma: float, beta_deg: float,
                         gamma: float = 1.4) -> dict:
    """Full post-shock state ratios across an oblique shock.

    Returns a dict with keys:
      * ``Ma2``       — post-shock Mach
      * ``p2_p1``     — static pressure ratio
      * ``rho2_rho1`` — density ratio
      * ``T2_T1``     — static temperature ratio
      * ``theta_deg`` — flow deflection angle (degrees)
      * ``T02_T01``   — stagnation temperature ratio (= 1 for adiabatic)
    """
    beta = np.radians(beta_deg)
    sinb = np.sin(beta)
    Ma_n1 = Ma * sinb
    if Ma_n1 <= 1.0:
        raise ValueError(
            f"Normal-component Ma={Ma_n1:.3f} <= 1: shock is not attached "
            f"(Ma={Ma}, beta={beta_deg}°).")
    g = gamma

    p2_p1 = 1.0 + (2.0 * g) / (g + 1.0) * (Ma_n1**2 - 1.0)
    rho2_rho1 = ((g + 1.0) * Ma_n1**2) / ((g - 1.0) * Ma_n1**2 + 2.0)
    T2_T1 = p2_p1 / rho2_rho1
    Ma_n2_sq = ((g - 1.0) * Ma_n1**2 + 2.0) / (2.0 * g * Ma_n1**2 - (g - 1.0))
    Ma_n2 = np.sqrt(max(Ma_n2_sq, 1e-12))

    theta_deg = theta_from_beta_Ma(beta_deg, Ma, g)
    theta = np.radians(theta_deg)
    Ma2 = Ma_n2 / np.sin(beta - theta)

    return {
        "Ma2":       float(Ma2),
        "p2_p1":     float(p2_p1),
        "rho2_rho1": float(rho2_rho1),
        "T2_T1":     float(T2_T1),
        "theta_deg": float(theta_deg),
        "T02_T01":   1.0,
    }
