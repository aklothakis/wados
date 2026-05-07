"""Thermodynamic / shock submodule (PSWR-1).

Phase 1 ships only :mod:`oblique_shock`. Saha and species are added in Phase 2.
"""

from .oblique_shock import (
    mach_angle,
    detachment_beta,
    theta_from_beta_M,
    beta_from_theta_M,
    rankine_hugoniot,
    cp_lower_wedge,
    beta_for_T_post,
    saha_onset_beta,
    saha_strong_beta,
    suggest_beta_knots,
)
from .species import (
    q_total_per_volume,
    q_internal,
    reaction_K,
    all_reaction_K,
    SPECIES,
    REACTIONS,
)
from .saha import (
    SahaResult,
    solve_saha_lte,
)

__all__ = [
    "mach_angle", "detachment_beta",
    "theta_from_beta_M", "beta_from_theta_M",
    "rankine_hugoniot", "cp_lower_wedge",
    "beta_for_T_post", "saha_onset_beta", "saha_strong_beta",
    "suggest_beta_knots",
    "q_total_per_volume", "q_internal",
    "reaction_K", "all_reaction_K",
    "SPECIES", "REACTIONS",
    "SahaResult", "solve_saha_lte",
]
