"""Modified Newtonian impact pressure (GVWD §4.2).

For high-alpha regimes where attached oblique-shock theory breaks down,
use modified Newtonian:

    Cp(theta_local) = Cp_max * sin^2(theta_local)   (theta_local > 0)
    Cp(theta_local) = 0                              (shadow region)

with Cp_max from the Lees / Anderson form (Anderson 1989 Eq. 3.41 / spec §4.2):

    Cp_max = (2/(gamma M^2)) * { [(gamma+1)^2 M^2
                                  / (4 gamma M^2 - 2 (gamma-1))]^(gamma/(gamma-1))
                                * (1 - gamma + 2 gamma M^2) / (gamma+1)
                               - 1 }

For gamma=1.4, M=10 -> Cp_max ~ 1.832 (matches spec DoD).
For M -> infinity, Cp_max -> 1.839; classical Newtonian uses Cp_max = 2.
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np

ArrayLike = Union[float, np.ndarray]


def cp_max_modified_newtonian(M_inf: float, gamma: float = 1.4) -> float:
    """Cp_max from Lees / Anderson modified-Newtonian formula.

    Anderson "Hypersonic and High-Temperature Gas Dynamics" 3rd ed. Eq. 3.41.
    """
    if M_inf <= 1.0:
        raise ValueError(f"M_inf must be supersonic; got {M_inf}")
    M2 = M_inf * M_inf
    num1 = (gamma + 1.0) ** 2 * M2
    den1 = 4.0 * gamma * M2 - 2.0 * (gamma - 1.0)
    bracket = ((num1 / den1) ** (gamma / (gamma - 1.0))
                * (1.0 - gamma + 2.0 * gamma * M2) / (gamma + 1.0)
                - 1.0)
    return float((2.0 / (gamma * M2)) * bracket)


def modified_newtonian_cp(M_inf: float, theta_local: ArrayLike,
                           gamma: float = 1.4) -> ArrayLike:
    """Cp on an inclined surface, modified Newtonian impact theory.

    Parameters
    ----------
    M_inf : float
        Freestream Mach number.
    theta_local : float or ndarray
        Local incidence angle of the surface to the freestream [rad]:
        positive = surface faces upstream (windward), negative = lee side.
        For a flat-plate at angle of attack alpha, theta_local = alpha.
    gamma : float, default 1.4

    Returns
    -------
    Cp : same shape as ``theta_local``. Zero on the lee side.
    """
    Cp_max = cp_max_modified_newtonian(M_inf, gamma)
    th = np.asarray(theta_local, dtype=float)
    s = np.sin(th)
    # Windward (s > 0): Cp = Cp_max * sin^2(theta).  Lee (s <= 0): Cp = 0.
    Cp = np.where(s > 0.0, Cp_max * s * s, 0.0)
    if np.isscalar(theta_local):
        return float(Cp)
    return Cp


def classical_newtonian_cp(theta_local: ArrayLike) -> ArrayLike:
    """Classical (M -> infinity, gamma -> 1) Newtonian Cp = 2 sin^2(theta).

    Provided for comparison and as a high-Mach check.
    """
    th = np.asarray(theta_local, dtype=float)
    s = np.sin(th)
    Cp = np.where(s > 0.0, 2.0 * s * s, 0.0)
    if np.isscalar(theta_local):
        return float(Cp)
    return Cp
