"""Distributed surface heat flux via Reynolds analogy (GVWD §4.10).

For each panel along a streamline, with local skin-friction coefficient
C_f(s):

    q(s) [W/m^2] = (C_f(s) / 2) * rho_e u_e (h_aw - h_w) * Pr^(-2/3)

Adiabatic-wall enthalpy: h_aw = h_e + r * u_e^2 / 2, with recovery
factor r ~ Pr^(1/2) laminar, Pr^(1/3) turbulent.

Inputs are panel-level edge state values (rho_e, u_e, Pr, h_e, h_w),
typically obtained from the panel-method post-shock state and the user-
supplied wall temperature.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def adiabatic_wall_enthalpy(h_e: float, u_e: float, Pr: float = 0.715,
                              turbulent: bool = False) -> float:
    """h_aw = h_e + r u_e^2 / 2 with r = Pr^(1/2) laminar or Pr^(1/3) turbulent."""
    r = Pr ** (1.0 / 3.0) if turbulent else Pr ** 0.5
    return h_e + r * u_e * u_e / 2.0


def distributed_surface_heat_flux(
    Cf_local: np.ndarray,
    rho_e: np.ndarray,
    u_e: np.ndarray,
    h_aw: np.ndarray,
    h_w: float,
    Pr: float = 0.715,
) -> np.ndarray:
    """Distributed wall heat flux via Reynolds analogy [W/m^2].

    Inputs are per-panel arrays. Returns same-shape array of q_dot.
    """
    return (np.asarray(Cf_local) / 2.0
             * np.asarray(rho_e) * np.asarray(u_e) * (np.asarray(h_aw) - h_w)
             * Pr ** (-2.0 / 3.0))


def panel_heating_summary(rho_e: np.ndarray, u_e: np.ndarray,
                            Cf: np.ndarray, T_e: np.ndarray, T_w: float,
                            cp_air: float = 1004.5, Pr: float = 0.715,
                            turbulent: bool = True) -> dict:
    """Convenience: compute per-panel q_dot given edge state.

    Returns dict with keys ``q_dot_panel`` (array), ``q_max``, ``q_mean``
    (both scalars in W/m^2).
    """
    h_e = cp_air * np.asarray(T_e)
    h_w = cp_air * T_w
    h_aw = h_e + (Pr ** (1.0/3.0) if turbulent else Pr ** 0.5) * u_e * u_e / 2.0
    q = distributed_surface_heat_flux(Cf, rho_e, u_e, h_aw, h_w, Pr=Pr)
    return {
        "q_dot_panel": q,
        "q_max": float(np.max(q)) if q.size else 0.0,
        "q_mean": float(np.mean(q)) if q.size else 0.0,
        "h_aw_max": float(np.max(h_aw)) if h_aw.size else 0.0,
        "h_w": h_w,
    }
