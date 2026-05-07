"""Drude plasma permittivity with electron-neutral collisions (PSWR-1 §5.4).

    eps(r; omega_0) = 1 - omega_p^2(r) / (omega_0 (omega_0 + i nu_en(r)))
    chi(r)           = eps(r) - 1

with

    omega_p^2 = n_e e^2 / (eps_0 m_e)              (rad^2 / s^2)
    nu_en     = 5.4e-17 * n_neutral_SI * sqrt(T)   (Hz)

NOTE the unit fix relative to PSWR-1 spec §5.4. The spec quotes
``nu_en = 5.4e-11 * n_n * sqrt(T)`` taken from Park 1990 / NRL Plasma
Formulary; in those references ``n_n`` is in **cm^-3** (CGS). Converting
to SI (m^-3) gives the coefficient 5.4e-17. Without this fix, nu_en is
1e6x too large and the post-shock plasma is artificially driven into the
strong-collision limit where Re(chi) is suppressed by ~ 10 orders, RCS
collapses to ~10^-8 m^2, and the spec's RCS gate cannot be met.

The Born approximation in §5.5 needs |Re chi| < 0.3 to be reliable; the
helper :func:`born_validity` flags this.
"""

from __future__ import annotations

import numpy as np


# CODATA 2018
EPSILON_0 = 8.8541878128e-12   # F/m
ELEMENTARY_CHARGE = 1.602176634e-19   # C
ELECTRON_MASS = 9.1093837015e-31   # kg


def drude_permittivity(n_e, n_neutral, T_K, omega_0):
    """Complex relative permittivity of a partially ionized plasma.

    Parameters
    ----------
    n_e, n_neutral : array_like   [m^-3]
    T_K            : array_like   [K]
    omega_0        : float        [rad/s]
    """
    n_e = np.asarray(n_e, dtype=float)
    n_n = np.asarray(n_neutral, dtype=float)
    T = np.asarray(T_K, dtype=float)

    omega_p2 = n_e * ELEMENTARY_CHARGE * ELEMENTARY_CHARGE \
        / (EPSILON_0 * ELECTRON_MASS)
    nu_en = 5.4e-17 * n_n * np.sqrt(np.maximum(T, 0.0))

    # eps - 1 = -omega_p^2 / (omega (omega + i nu)). Compute chi directly to
    # avoid catastrophic cancellation when omega_p << omega.
    denom = omega_0 * (omega_0 + 1j * nu_en)
    chi = -omega_p2 / denom
    return 1.0 + chi


def susceptibility(n_e, n_neutral, T_K, omega_0):
    """chi = eps - 1 (complex array)."""
    n_e = np.asarray(n_e, dtype=float)
    n_n = np.asarray(n_neutral, dtype=float)
    T = np.asarray(T_K, dtype=float)
    omega_p2 = n_e * ELEMENTARY_CHARGE * ELEMENTARY_CHARGE \
        / (EPSILON_0 * ELECTRON_MASS)
    nu_en = 5.4e-17 * n_n * np.sqrt(np.maximum(T, 0.0))
    denom = omega_0 * (omega_0 + 1j * nu_en)
    return -omega_p2 / denom


def critical_density(omega_0):
    """n_critical = omega_0^2 eps_0 m_e / e^2  [m^-3]."""
    return omega_0 * omega_0 * EPSILON_0 * ELECTRON_MASS \
        / (ELEMENTARY_CHARGE * ELEMENTARY_CHARGE)


def born_validity(chi, threshold: float = 0.3):
    """Return (is_valid, max_abs_re_chi). Born is suspect if |Re chi| > threshold."""
    re = np.abs(np.real(np.asarray(chi)))
    max_re = float(re.max()) if re.size else 0.0
    return (max_re < threshold), max_re
