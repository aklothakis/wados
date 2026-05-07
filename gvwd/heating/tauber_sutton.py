"""Tauber-Sutton 1991 convective + radiative heating correlations
(GVWD §4.10).

Convective:
    q_conv = C_1 * sqrt(rho_inf / R_N) * V_inf^3.15      [W/m^2]
    C_1 = 1.83e-4

Radiative (significant only for M > 12):
    q_rad = C_2 * R_N^a * rho_inf^b * V_inf^c            [W/m^2]
    The Tauber-Sutton 1991 correlation gives a velocity-dependent
    coefficient C_2(V); we use the table form for V in [9, 16] km/s.
"""

from __future__ import annotations

import math


# Convective: same as fay_riddell coefficient
C_TS_CONV = 1.83e-4


def tauber_sutton_convective(rho_inf: float, V_inf: float,
                              R_N: float) -> float:
    """Tauber-Sutton convective stagnation-point heat flux [W/m^2]."""
    if R_N <= 0:
        raise ValueError("R_N must be > 0")
    return C_TS_CONV * math.sqrt(rho_inf / R_N) * V_inf ** 3.15


# Radiative: Tauber-Sutton 1991 fit with velocity-dependent coefficient
# (see Tauber & Sutton J. Spacecraft & Rockets 28(1), 40-42, 1991, Eq. 16)
# The radiative-heating correlation for Earth atmosphere:
#   q_rad = C_2 * R_N^a * rho^b * f(V)
# with a=1.0, b=1.22 for Earth, and f(V) tabulated as a piecewise-poly.
# For first attack we use a simple linear-in-V form valid 9-16 km/s.

def _f_velocity(V_km_s: float) -> float:
    """Velocity-dependent factor f(V) for Earth radiative heating.

    Tabulated at V = 9, 10, 11, ..., 16 km/s; linear interpolation
    elsewhere. f(V) values from Tauber-Sutton 1991 Table II (W/cm^2/atm).
    """
    table_V = [9, 10, 11, 12, 13, 14, 15, 16]
    table_f = [1.5, 4.3, 9.7, 19.5, 35.0, 55.0, 81.0, 115.0]
    if V_km_s <= table_V[0]:
        return table_f[0]
    if V_km_s >= table_V[-1]:
        return table_f[-1]
    for i in range(len(table_V) - 1):
        if table_V[i] <= V_km_s <= table_V[i+1]:
            t = (V_km_s - table_V[i]) / (table_V[i+1] - table_V[i])
            return table_f[i] + t * (table_f[i+1] - table_f[i])
    return 0.0


def tauber_sutton_radiative(rho_inf: float, V_inf: float,
                             R_N: float) -> float:
    """Tauber-Sutton radiative stagnation-point heat flux [W/m^2].

    Significant only for V > ~9 km/s (M > ~25 at ground-level T).
    Returns 0 for V_inf < 9000 m/s (correlation lower bound).

    q_rad [W/m^2] = R_N^1.0 * rho_inf^1.22 * f(V) * c
    where f(V) is in W/cm^2/atm and the unit conversion is included.
    """
    V_km = V_inf / 1000.0
    if V_km < 9.0:
        return 0.0
    f_V = _f_velocity(V_km)   # W/cm^2/atm
    rho_atm = rho_inf / 1.225   # density in atm-equivalent
    q_W_cm2 = R_N ** 1.0 * rho_atm ** 1.22 * f_V
    return q_W_cm2 * 1.0e4   # W/cm^2 -> W/m^2
