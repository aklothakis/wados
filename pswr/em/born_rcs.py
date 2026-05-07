"""Born-approximation bistatic RCS (PSWR-1 §5.5).

Born amplitude for a low-contrast scatterer:

    f(k_i_hat, k_s_hat) = (k_0^2 / 4 pi)  ∫_V chi(r) exp(i q . r) d^3 r

with scattering wavevector  q = k_0 (k_i_hat - k_s_hat).

Bistatic RCS:

    sigma_b = 4 pi |f|^2

Conventions:
    k_i_hat is the unit vector in the *propagation direction* of the incident
    wave (e.g. k_i_hat = -x_hat for a co-flying threat radar at the nose).
    k_s_hat is the propagation direction of the scattered wave; for
    monostatic backscatter k_s_hat = -k_i_hat.
"""

from __future__ import annotations

import math
import numpy as np

C_LIGHT = 2.99792458e8


# ----------------------------------------------------------------------
#  Core Born integral
# ----------------------------------------------------------------------

def born_amplitude(k_i_hat, k_s_hat, omega_0: float,
                   chi: np.ndarray, points: np.ndarray,
                   volumes: np.ndarray) -> complex:
    """Vectorised single-sum Born scattering amplitude.

    chi      : complex array, shape S
    points   : real array, shape S + (3,)
    volumes  : real array, shape S
    """
    k0 = omega_0 / C_LIGHT
    k_i = k0 * np.asarray(k_i_hat, dtype=float)
    k_s = k0 * np.asarray(k_s_hat, dtype=float)
    q = k_i - k_s
    # phase = q . r  (broadcast over leading dims)
    phase = (points[..., 0] * q[0]
             + points[..., 1] * q[1]
             + points[..., 2] * q[2])
    integrand = chi * volumes * np.exp(1j * phase)
    f = (k0 * k0 / (4.0 * math.pi)) * np.sum(integrand)
    return complex(f)


def bistatic_rcs(k_i_hat, k_s_hat, omega_0: float,
                 chi: np.ndarray, points: np.ndarray,
                 volumes: np.ndarray) -> float:
    """Bistatic radar cross section sigma_b [m^2]."""
    f = born_amplitude(k_i_hat, k_s_hat, omega_0, chi, points, volumes)
    return float(4.0 * math.pi * (f.real * f.real + f.imag * f.imag))


def monostatic_rcs(k_i_hat, omega_0: float,
                   chi: np.ndarray, points: np.ndarray,
                   volumes: np.ndarray) -> float:
    k_i = np.asarray(k_i_hat, dtype=float)
    return bistatic_rcs(k_i, -k_i, omega_0, chi, points, volumes)


def rcs_dBsm(sigma_m2: float) -> float:
    """Convert linear RCS [m^2] to dBsm."""
    return 10.0 * math.log10(max(sigma_m2, 1e-300))


# ----------------------------------------------------------------------
#  Direction parametrisation
# ----------------------------------------------------------------------

def bistatic_direction_from_angles(k_i_hat, theta_s_rad: float,
                                   phi_s_rad: float = 0.0) -> np.ndarray:
    """Return k_s_hat for prescribed bistatic angle (theta_s, phi_s).

    theta_s is measured from the back-to-source direction (-k_i_hat):
        theta_s = 0      -> k_s = -k_i_hat   (monostatic backscatter)
        theta_s = pi     -> k_s = +k_i_hat   (forward scatter)
        theta_s = pi/2   -> k_s perpendicular (side scatter)
    phi_s is the azimuth around -k_i_hat (default 0 puts the scattered
    direction in the plane spanned by -k_i_hat and the global y axis).
    """
    k_i = np.asarray(k_i_hat, dtype=float)
    n = np.linalg.norm(k_i)
    if n == 0:
        raise ValueError("k_i_hat must be non-zero")
    k_i = k_i / n
    z_hat = -k_i  # back-to-source axis

    # Build orthonormal basis (z_hat, e_x, e_y)
    # e_x: pick world y if z_hat not parallel to it, else world x
    world_y = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(z_hat, world_y)) > 0.99:
        world_y = np.array([1.0, 0.0, 0.0])
    e_x = world_y - np.dot(world_y, z_hat) * z_hat
    e_x = e_x / np.linalg.norm(e_x)
    e_y = np.cross(z_hat, e_x)

    sin_t = math.sin(theta_s_rad); cos_t = math.cos(theta_s_rad)
    cos_p = math.cos(phi_s_rad); sin_p = math.sin(phi_s_rad)
    k_s = cos_t * z_hat + sin_t * cos_p * e_x + sin_t * sin_p * e_y
    return k_s / np.linalg.norm(k_s)


# ----------------------------------------------------------------------
#  Analytic / validation helpers
# ----------------------------------------------------------------------

def sphere_form_factor(qa: float) -> float:
    """F(qa) = 3 (sin(qa) - qa cos(qa)) / (qa)^3,  F(0) = 1."""
    if qa < 1e-6:
        return 1.0 - 0.1 * qa * qa  # Taylor expansion
    return 3.0 * (math.sin(qa) - qa * math.cos(qa)) / (qa ** 3)


def rayleigh_uniform_analytic(chi: complex, V: float, omega_0: float) -> float:
    """Rayleigh-limit RCS of a uniform-chi region (sphere/cube same):

        sigma_R = (k_0^4 / 4 pi) |chi|^2 V^2

    Valid when k_0 (V^{1/3}) << 1.
    """
    k0 = omega_0 / C_LIGHT
    return (k0 ** 4) * V * V * (abs(chi) ** 2) / (4.0 * math.pi)


def cube_validation(chi: complex = -0.01 + 0.0j, side: float = 1.0,
                    f0_Hz: float = 1.0e6, n: int = 20) -> dict:
    """1 m^3 uniform-chi cube test from PSWR-1 §5.5 DoD gate.

    Returns dict with sigma_num, sigma_analytic, error_pct, ka.
    Default frequency f0 = 1 MHz keeps k_0 a = 0.021 << 1.
    """
    omega_0 = 2.0 * math.pi * f0_Hz
    k0 = omega_0 / C_LIGHT

    edges = np.linspace(-0.5 * side, 0.5 * side, n + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    X, Y, Z = np.meshgrid(centres, centres, centres, indexing="ij")
    points = np.stack([X, Y, Z], axis=-1)
    dV = (side / n) ** 3
    volumes = np.full((n, n, n), dV)
    chi_grid = np.full((n, n, n), chi, dtype=complex)

    k_i_hat = np.array([1.0, 0.0, 0.0])
    sigma_num = monostatic_rcs(k_i_hat, omega_0, chi_grid, points, volumes)
    sigma_ana = rayleigh_uniform_analytic(chi, side ** 3, omega_0)
    err = abs(sigma_num - sigma_ana) / max(sigma_ana, 1e-300) * 100.0
    return {
        "sigma_num": sigma_num,
        "sigma_analytic": sigma_ana,
        "error_pct": err,
        "ka": k0 * 0.5 * side,
        "k0": k0,
        "n_cells": n ** 3,
    }


# ----------------------------------------------------------------------
#  Sheath-grid wrapper
# ----------------------------------------------------------------------

def rcs_from_sheath(grid, omega_0: float,
                    k_i_hat=(-1.0, 0.0, 0.0),
                    k_s_hat=(1.0, 0.0, 0.0)) -> dict:
    """Compute Born sigma_b for a :class:`SheathGrid`.

    Returns dict with sigma_b, max_abs_re_chi, born_valid, n_cells, run_time_s.
    """
    import time
    from ..plasma.permittivity import susceptibility, born_validity

    chi = susceptibility(grid.n_e, grid.n_neutral, grid.T, omega_0)
    points = np.stack([grid.X, grid.Y, grid.Z], axis=-1)
    volumes = grid.cell_volume
    valid, max_re = born_validity(chi)

    t0 = time.perf_counter()
    sigma = bistatic_rcs(k_i_hat, k_s_hat, omega_0, chi, points, volumes)
    dt = time.perf_counter() - t0

    return {
        "sigma_b": sigma,
        "sigma_b_dBsm": rcs_dBsm(sigma),
        "max_abs_re_chi": max_re,
        "born_valid": valid,
        "n_cells": int(np.prod(chi.shape)),
        "run_time_s": dt,
    }
