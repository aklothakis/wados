"""Plasma-sheath n_e(r) on a structured 3-D grid (PSWR-1 §5.3-5.4).

For the variable-wedge family the post-shock thermodynamic state behind the
local 2-D shock depends only on the spanwise coordinate y. We solve Saha
once per spanwise station, then extrude n_e wall-normally with a top-hat
profile of thickness :math:`3 \\delta_{BL}(x, y)` per spec §5.3 first attack.

The resulting grid is consumed by Phase 3 (permittivity + Born RCS).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..geometry.variable_wedge import VariableWedgeWaverider
from ..thermo.oblique_shock import rankine_hugoniot
from ..thermo.saha import solve_saha_lte, SahaResult
from ..aero.viscous import per_station_state


# Physical constants (CODATA 2018)
EPSILON_0 = 8.8541878128e-12   # F/m
ELEMENTARY_CHARGE = 1.602176634e-19  # C
ELECTRON_MASS = 9.1093837015e-31  # kg
K_B = 1.380649e-23


# ----------------------------------------------------------------------
#  Plasma physics helpers
# ----------------------------------------------------------------------

def plasma_frequency(n_e: float | np.ndarray) -> float | np.ndarray:
    """omega_p = sqrt(n_e e^2 / (eps_0 m_e))   [rad/s]."""
    n = np.asarray(n_e)
    return np.sqrt(np.maximum(n, 0.0) * ELEMENTARY_CHARGE * ELEMENTARY_CHARGE
                   / (EPSILON_0 * ELECTRON_MASS))


def electron_collision_frequency(n_neutral: float | np.ndarray,
                                  T_K: float | np.ndarray) -> float | np.ndarray:
    """Spitzer/Park collision frequency: nu_en = 5.4e-11 n_n sqrt(T)  [Hz].

    PSWR-1 spec §5.4.
    """
    n_n = np.asarray(n_neutral)
    T = np.asarray(T_K)
    return 5.4e-11 * n_n * np.sqrt(np.maximum(T, 0.0))


# ----------------------------------------------------------------------
#  Sheath grid
# ----------------------------------------------------------------------

@dataclass
class SheathGrid:
    """Structured wall-normal-extruded grid wrapping the lower surface.

    All arrays have shape (n_chord, n_span, n_normal). Coordinates are in
    PSWR-1 frame (x stream, y span, z up).
    """
    X: np.ndarray
    Y: np.ndarray
    Z: np.ndarray
    n_e: np.ndarray
    T: np.ndarray
    p: np.ndarray
    n_neutral: np.ndarray   # total neutral particle density (excludes e-, NO+)
    delta_BL: np.ndarray    # (n_chord, n_span) — sheath thickness used
    saha_per_station: list = field(default_factory=list)  # list of SahaResult per spanwise station

    @property
    def shape(self):
        return self.X.shape

    @property
    def cell_volume(self) -> np.ndarray:
        """Approx cell volume = dx * dy * d_zeta (uniform structured)."""
        # Use forward differences with edge-padding
        def _spacing(arr, axis):
            d = np.diff(arr, axis=axis)
            d_pad = np.concatenate([d, np.take(d, [-1], axis=axis)], axis=axis)
            return np.abs(d_pad)
        dx = _spacing(self.X, 0)
        dy = _spacing(self.Y, 1)
        dz = _spacing(self.Z, 2)
        return dx * dy * dz

    @property
    def n_e_max(self) -> float:
        return float(self.n_e.max())

    @property
    def n_e_min_nonzero(self) -> float:
        nz = self.n_e[self.n_e > 0]
        return float(nz.min()) if nz.size else 0.0


def build_sheath_grid(wr: VariableWedgeWaverider, *,
                      T_w: float = 1500.0,
                      Re_x_tr: float = 1e6,
                      sheath_factor: float = 3.0,
                      n_chord: int | None = None,
                      n_span: int | None = None,
                      n_normal: int = 20,
                      profile: str = "top_hat") -> SheathGrid:
    """Build a structured (n_chord, n_span, n_normal) grid of n_e(r).

    Resolution defaults: streamwise = wr.n_chord, spanwise = wr.n_span,
    n_normal=20 (per spec §5.4 default).

    ``profile`` may be ``'top_hat'`` (uniform n_e for zeta in [0, delta_BL])
    or ``'crocco'`` (cos^2 ramp from wall to delta_BL — Phase 5 stretch goal).
    """
    n_chord_g = int(n_chord) if n_chord else wr.n_chord
    n_span_g = int(n_span) if n_span else wr.n_span
    n_normal = int(n_normal)

    # Spanwise: re-sample on a uniform η grid
    eta = np.linspace(-1.0, 1.0, n_span_g)
    y_grid = eta * wr.y_tip

    # Per-station post-shock state
    state = per_station_state(wr, T_w=T_w, Re_x_tr=Re_x_tr)

    # Saha at each spanwise station (using post-shock T2, p2 from RH)
    sahas = []
    n_e_y = np.zeros(n_span_g)
    n_neutral_y = np.zeros(n_span_g)
    T_y = np.zeros(n_span_g)
    p_y = np.zeros(n_span_g)
    # Use the waverider's own beta(y) helper so the flat-nose mapping is
    # respected (eta = 0 in the flat region, otherwise linear in y).
    beta_at_y = wr.beta_at_y(y_grid)
    for j, yj in enumerate(y_grid):
        beta_j = float(beta_at_y[j])
        rh = rankine_hugoniot(wr.M_inf, beta_j, wr.p_inf, wr.T_inf, wr.gamma)
        T2 = rh["T2"]; p2 = rh["p2"]
        T_y[j] = T2
        p_y[j] = p2
        try:
            r = solve_saha_lte(T2, p2)
            ne = r.n_e if (r.converged and math.isfinite(r.n_e)) else 0.0
            nn = (r.n["N2"] + r.n["O2"] + r.n["N"] + r.n["O"] + r.n["NO"]) \
                 if r.converged else (p2 / (K_B * T2))
            n_e_y[j] = ne
            n_neutral_y[j] = nn if math.isfinite(nn) else (p2 / (K_B * T2))
        except Exception:
            r = None
            n_e_y[j] = 0.0
            n_neutral_y[j] = p2 / (K_B * T2)
        sahas.append(r)

    # Streamwise grid: parametric chord position s in [0, chord(y)]
    s_frac = np.linspace(0.0, 1.0, n_chord_g)

    # Allocate
    X = np.zeros((n_chord_g, n_span_g, n_normal))
    Y = np.zeros_like(X)
    Z = np.zeros_like(X)
    n_e = np.zeros_like(X)
    T = np.zeros_like(X)
    p = np.zeros_like(X)
    n_n = np.zeros_like(X)
    delta_BL = np.zeros((n_chord_g, n_span_g))

    # Need beta and theta interpolated at the y_grid stations (state is on wr.y_grid)
    # Re-do per the resampled eta grid, respecting flat-nose mapping:
    beta_eta = wr.beta_at_y(y_grid)
    from ..thermo.oblique_shock import theta_from_beta_M
    theta_eta = np.array([theta_from_beta_M(b, wr.M_inf, wr.gamma)
                          for b in beta_eta])

    # x_LE, z_LE on resampled grid (uses helpers so flat-nose region is correct)
    x_LE = wr.x_LE_at_y(y_grid)
    z_LE = wr.z_LE_at_y(y_grid)
    chord_y = wr.body_length - x_LE

    # Re_x*, mu*, rho* — recompute on resampled grid
    a_inf = math.sqrt(wr.gamma * 287.05 * wr.T_inf)
    u_inf = wr.M_inf * a_inf

    from ..aero.viscous import (eckert_reference_T, sutherland_viscosity,
                                  boundary_layer_thickness)
    T_e = T_y; p_e = p_y
    rho_e = p_e / (287.05 * T_e)
    M_e = np.array([rankine_hugoniot(wr.M_inf, float(b), wr.p_inf, wr.T_inf,
                                      wr.gamma)["M2"] for b in beta_eta])
    a_e = np.sqrt(wr.gamma * 287.05 * T_e)
    u_e = M_e * a_e

    T_star = eckert_reference_T(T_e, M_e, T_w)
    mu_star = sutherland_viscosity(T_star)
    rho_star = p_e / (287.05 * T_star)

    for i, sf in enumerate(s_frac):
        # Lower-surface point at chord-fraction sf, spanwise station j
        for j in range(n_span_g):
            chord_j = chord_y[j]
            x_chord = sf * chord_j
            cos_th = math.cos(theta_eta[j])
            sin_th = math.sin(theta_eta[j])

            x_wall = x_LE[j] + x_chord * cos_th
            y_wall = y_grid[j]
            z_wall = z_LE[j] - x_chord * sin_th

            # Re_x* at this chord position
            Re_x = rho_star[j] * u_e[j] * max(x_chord, 1e-9) / max(mu_star[j], 1e-30)
            d_BL = boundary_layer_thickness(max(x_chord, 1e-9), Re_x)
            d_BL = float(d_BL)
            d_grid = sheath_factor * d_BL
            delta_BL[i, j] = d_BL

            # Wall-normal direction in PSWR-1 frame: outward (lower surface
            # faces downward in z, but the post-shock fluid wedge sits above
            # the wedge surface — i.e. between wedge and shock). The n_e is
            # nonzero in that wedge layer. n_hat points TO the post-shock
            # fluid; for the lower surface n_hat = (sin(theta), 0, cos(theta))
            # i.e. forward and upward.
            n_hat = np.array([math.sin(theta_eta[j]), 0.0, math.cos(theta_eta[j])])

            zetas = np.linspace(0.0, max(d_grid, 1e-9), n_normal)
            for k, zeta in enumerate(zetas):
                pt = np.array([x_wall, y_wall, z_wall]) + zeta * n_hat
                X[i, j, k] = pt[0]
                Y[i, j, k] = pt[1]
                Z[i, j, k] = pt[2]
                # n_e profile
                if profile == "top_hat":
                    f_zeta = 1.0 if zeta <= d_BL else 0.0
                elif profile == "crocco":
                    if zeta <= d_BL:
                        eta_n = zeta / max(d_BL, 1e-30)
                        f_zeta = math.cos(0.5 * math.pi * eta_n) ** 2
                    else:
                        f_zeta = 0.0
                else:
                    raise ValueError(f"unknown profile '{profile}'")
                n_e[i, j, k] = n_e_y[j] * f_zeta
                n_n[i, j, k] = n_neutral_y[j] * f_zeta + (
                    p_e[j] / (K_B * T_e[j]) * (0.0 if zeta <= d_BL else 1.0)
                ) * 0.0   # outside BL we leave it zero (won't be queried)
                T[i, j, k] = T_e[j]
                p[i, j, k] = p_e[j]

    return SheathGrid(
        X=X, Y=Y, Z=Z, n_e=n_e, T=T, p=p, n_neutral=n_n,
        delta_BL=delta_BL, saha_per_station=sahas,
    )
