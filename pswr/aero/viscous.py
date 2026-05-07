"""Compressible boundary-layer skin friction (PSWR-1 §5.6).

Eckert reference temperature method for laminar/turbulent flat-plate
compressible boundary layers, plus a 1/7-power-law thickness estimate.
Per-spanwise-station application to the variable-wedge waverider.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np

from ..geometry.variable_wedge import VariableWedgeWaverider
from ..geometry.volume import planform_area
from ..thermo.oblique_shock import rankine_hugoniot

_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")

# Sutherland constants for air
_MU_REF_AIR = 1.716e-5      # Pa s at T_ref
_T_REF_AIR = 273.15         # K
_S_AIR = 110.4              # K
_R_AIR = 287.05             # J/(kg K)
_GAMMA_AIR = 1.4


# ----------------------------------------------------------------------
#  Thermophysical helpers
# ----------------------------------------------------------------------

def sutherland_viscosity(T_K: float | np.ndarray) -> float | np.ndarray:
    """Sutherland viscosity for air [Pa s]."""
    T = np.asarray(T_K)
    return _MU_REF_AIR * (T / _T_REF_AIR) ** 1.5 * (_T_REF_AIR + _S_AIR) / (T + _S_AIR)


def eckert_reference_T(T_e: float | np.ndarray, M_e: float | np.ndarray,
                       T_w: float) -> float | np.ndarray:
    """Eckert reference temperature T* (PSWR-1 §5.6)."""
    return T_e * (0.5 + 0.039 * np.asarray(M_e) ** 2 + 0.5 * (T_w / np.asarray(T_e)))


def cf_laminar(Re_x: float | np.ndarray) -> float | np.ndarray:
    return 0.664 / np.sqrt(np.asarray(Re_x))


def cf_turbulent(Re_x: float | np.ndarray) -> float | np.ndarray:
    return 0.0592 / np.asarray(Re_x) ** 0.2


def cf_blended(Re_x: np.ndarray, Re_x_tr: float = 1e6) -> np.ndarray:
    """Laminar below transition Re, turbulent above."""
    Re = np.asarray(Re_x)
    out = np.where(Re < Re_x_tr, cf_laminar(Re), cf_turbulent(Re))
    return out


def boundary_layer_thickness(x_chord: float | np.ndarray,
                             Re_x_star: float | np.ndarray) -> float | np.ndarray:
    """Compressible turbulent BL displacement thickness (Pohlhausen 1/7 power)."""
    Re = np.asarray(Re_x_star)
    # Avoid divide-by-zero at LE
    Re = np.where(Re > 1.0, Re, 1.0)
    return 0.37 * np.asarray(x_chord) / Re ** 0.2


# ----------------------------------------------------------------------
#  Per-station and integrated viscous quantities
# ----------------------------------------------------------------------

def per_station_state(wr: VariableWedgeWaverider, T_w: float = 1500.0,
                      Re_x_tr: float = 1e6) -> Dict[str, np.ndarray]:
    """Compute spanwise post-shock + Eckert reference state at the base plane.

    Returns a dict of arrays of length n_span:
        T_e, p_e, rho_e, M_e   : edge (post-shock) state
        T_star, mu_star, rho_star : Eckert reference state
        u_e                    : edge velocity = M_e * sqrt(gamma R T_e)
        chord                  : x_b - x_LE(y)
        Re_chord_star          : Reynolds at base using starred properties
        delta_BL_base          : BL thickness at base
        Cf_chord               : skin-friction averaged over chord (uses analytic
                                 integration of cf_laminar+cf_turbulent)
    """
    n = len(wr.y_grid)
    T_e = np.empty(n); p_e = np.empty(n); rho_e = np.empty(n); M_e = np.empty(n)
    for j in range(n):
        rh = rankine_hugoniot(wr.M_inf, float(wr.beta_y[j]),
                              wr.p_inf, wr.T_inf, wr.gamma)
        T_e[j] = rh["T2"]; p_e[j] = rh["p2"]
        rho_e[j] = rh["rho2"]; M_e[j] = rh["M2"]

    T_star = eckert_reference_T(T_e, M_e, T_w)
    mu_star = sutherland_viscosity(T_star)
    rho_star = p_e / (_R_AIR * T_star)
    a_e = np.sqrt(_GAMMA_AIR * _R_AIR * T_e)
    u_e = M_e * a_e

    chord = wr.body_length - wr.leading_edge[:, 0]
    chord_safe = np.where(chord > 1e-9, chord, 1e-9)

    Re_chord = rho_star * u_e * chord_safe / mu_star
    Re_chord = np.where(chord > 1e-9, Re_chord, 0.0)

    # Analytic integral of Cf along chord:
    # If Re_x = (rho_e u_e / mu_e) x = K x, then for laminar:
    #   int_0^L 0.664/sqrt(K x) dx = 0.664 * 2 sqrt(L/K) = 1.328 sqrt(L/K)
    #   chord-averaged Cf_lam = 1.328 / sqrt(Re_chord)
    # For turbulent: int 0.0592 (Kx)^-0.2 dx = 0.0592 * (Kx)^0.8 / (0.8 K) at L
    #   = 0.074 / Re_chord^0.2 (chord-averaged)
    Cf_lam_avg = np.where(Re_chord > 0, 1.328 / np.sqrt(np.where(Re_chord>0, Re_chord, 1.0)), 0.0)
    Cf_turb_avg = np.where(Re_chord > 0, 0.074 / np.where(Re_chord>0, Re_chord, 1.0)**0.2, 0.0)

    # Choose laminar if Re_chord < Re_x_tr, else turbulent (mixed BL ignored
    # for simplicity; Phase 5 stretch goal could refine to a transition mix).
    Cf_chord = np.where(Re_chord < Re_x_tr, Cf_lam_avg, Cf_turb_avg)

    delta_BL_base = boundary_layer_thickness(chord_safe, Re_chord)
    delta_BL_base = np.where(chord > 1e-9, delta_BL_base, 0.0)

    return dict(T_e=T_e, p_e=p_e, rho_e=rho_e, M_e=M_e,
                T_star=T_star, mu_star=mu_star, rho_star=rho_star,
                u_e=u_e, chord=chord,
                Re_chord_star=Re_chord, delta_BL_base=delta_BL_base,
                Cf_chord=Cf_chord)


def viscous_drag_coefficient(wr: VariableWedgeWaverider, T_w: float = 1500.0,
                             Re_x_tr: float = 1e6) -> Dict[str, float]:
    """C_D,fric = (2/q_inf S_ref) ∫∫ (½ rho_e u_e^2 C_f) dA.

    For the variable-wedge family, integrating over chord at each station:
        ∫_0^L (½ rho_e u_e^2 C_f) dx = ½ rho_e u_e^2 C_f_avg * L
    The slant-area conversion factor ``1/cos(theta)`` is applied per station.
    """
    state = per_station_state(wr, T_w=T_w, Re_x_tr=Re_x_tr)
    y = wr.y_grid
    rho_e = state["rho_e"]; u_e = state["u_e"]
    chord = state["chord"]; Cf = state["Cf_chord"]
    cos_th = np.cos(wr.theta_y)

    # q_inf based on freestream
    a_inf = math.sqrt(_GAMMA_AIR * _R_AIR * wr.T_inf)
    u_inf = wr.M_inf * a_inf
    rho_inf = wr.p_inf / (_R_AIR * wr.T_inf)
    q_inf = 0.5 * rho_inf * u_inf * u_inf

    # Per-station chord-averaged friction force per unit span:
    # dF_fric/dy = ½ rho_e u_e^2 * C_f_avg * (chord / cos(theta))
    dF_dy = 0.5 * rho_e * u_e * u_e * Cf * np.where(cos_th > 1e-6, chord / cos_th, chord)
    F_fric = float(_trapz(dF_dy, y))

    S_ref = planform_area(wr)
    if S_ref <= 0.0:
        return dict(CD_friction=0.0, F_friction=F_fric, q_inf=q_inf)
    CD_fric = F_fric / (q_inf * S_ref)
    return dict(CD_friction=CD_fric, F_friction=F_fric, q_inf=q_inf,
                S_ref=S_ref, state=state)
