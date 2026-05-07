"""Species partition functions and equilibrium constants (PSWR-1 §5.3).

Loads species data from ``pswr/data/species_thermo.json`` (NASA / Park 1990
form). Partition functions are the simple high-T closed-form expressions:

    Q_int_atom    = g_el                    (constants for first-attack)
    Q_int_diatom  = (T / sigma theta_r) (1 / (1 - exp(-theta_v/T))) g_el

Translational partition function per unit volume:

    q_tr(T, m) = (2 pi m k_B T / h^2)^(3/2)   [m^-3]

Total per-volume partition function Q(T) = q_tr * Q_int has units m^-3 and
appears directly in the law-of-mass-action

    K_n = prod_i Q_i^nu_i  exp(-Delta E / k_B T)

with stoichiometric coefficients ``nu_i > 0`` for products, < 0 for reactants.
The units of K_n are m^(-3 sum nu_i).
"""

from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from typing import Dict, List, Tuple

# Physical constants (CODATA 2018)
K_B = 1.380649e-23      # J/K
H_PLANCK = 6.62607015e-34  # J s
AMU = 1.66053906660e-27  # kg
EV = 1.602176634e-19    # J


_DATA_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "data",
                          "species_thermo.json")


def _load_data() -> dict:
    with open(_DATA_PATH, "r") as f:
        return json.load(f)


_DATA = _load_data()
SPECIES: Dict[str, dict] = _DATA["species"]
REACTIONS: Dict[str, dict] = _DATA["reactions"]
X_N: float = _DATA["air_atom_fractions"]["X_N"]
X_O: float = _DATA["air_atom_fractions"]["X_O"]


# ----------------------------------------------------------------------
#  Partition functions
# ----------------------------------------------------------------------

def q_translational(T_K: float, mass_u: float) -> float:
    """Translational q per unit volume [m^-3]."""
    m_kg = mass_u * AMU
    return (2.0 * math.pi * m_kg * K_B * T_K / (H_PLANCK * H_PLANCK)) ** 1.5


def q_rotational(T_K: float, theta_r_K: float, sigma: int) -> float:
    """Linear-molecule rotational partition function (rigid-rotor, high-T)."""
    return T_K / (sigma * theta_r_K)


def q_vibrational(T_K: float, theta_v_K: float) -> float:
    """Harmonic-oscillator vibrational partition function (ground = 1)."""
    return 1.0 / (1.0 - math.exp(-theta_v_K / T_K))


def q_internal(species: str, T_K: float) -> float:
    """Q_int = Q_rot * Q_vib * g_el (or just g_el for atoms)."""
    s = SPECIES[species]
    g_el = float(s["g_electronic"])
    if s["is_atom"]:
        return g_el
    Qr = q_rotational(T_K, s["theta_rot_K"], s["sigma_sym"])
    Qv = q_vibrational(T_K, s["theta_vib_K"])
    return Qr * Qv * g_el


def q_total_per_volume(species: str, T_K: float) -> float:
    """Q(T)/V = q_tr(T, m) * Q_int(T) [m^-3]."""
    s = SPECIES[species]
    return q_translational(T_K, s["mass_u"]) * q_internal(species, T_K)


def log_q_total_per_volume(species: str, T_K: float) -> float:
    """log of total Q per volume — useful when q_total is huge or tiny."""
    s = SPECIES[species]
    m_kg = s["mass_u"] * AMU
    log_qtr = 1.5 * math.log(2.0 * math.pi * m_kg * K_B * T_K
                             / (H_PLANCK * H_PLANCK))
    if s["is_atom"]:
        log_qint = math.log(s["g_electronic"])
    else:
        Qr = q_rotational(T_K, s["theta_rot_K"], s["sigma_sym"])
        Qv = q_vibrational(T_K, s["theta_vib_K"])
        log_qint = math.log(Qr * Qv * s["g_electronic"])
    return log_qtr + log_qint


# ----------------------------------------------------------------------
#  Equilibrium constants
# ----------------------------------------------------------------------

def reaction_K(name: str, T_K: float) -> float:
    """Equilibrium constant for a named reaction in the data file.

    K_n = prod Q_i^nu_i exp(-Delta E / k_B T), units m^(-3 sum nu).
    """
    rxn = REACTIONS[name]
    log_K = -rxn["delta_E_eV"] * EV / (K_B * T_K)
    for sp, nu in rxn["products"]:
        log_K += nu * log_q_total_per_volume(sp, T_K)
    for sp, nu in rxn["reactants"]:
        log_K -= nu * log_q_total_per_volume(sp, T_K)
    return math.exp(log_K)


def all_reaction_K(T_K: float) -> Dict[str, float]:
    return {name: reaction_K(name, T_K) for name in REACTIONS}
