"""Thermodynamic / shock-wave submodule (GVWD §4.1-4.3).

Phase 1 ships:
  - ``oblique_shock``  : theta-beta-M, Rankine-Hugoniot, swept-shock
                         (Emanuel 2015), theta_max, ShockDetachedError.
  - ``newtonian``      : modified Newtonian Cp.
  - ``tangent_wedge``  : tangent-wedge with Newtonian fallback.
  - ``oswatitsch``     : equal-strength multi-shock ramp solver.

Where PSWR-1 already provides equivalent utilities, this module re-exports
them rather than duplicating code.
"""

from .oblique_shock import (
    ShockDetachedError,
    mach_angle,
    theta_max,
    theta_from_beta_M,
    obtain_beta,
    rankine_hugoniot,
    cp_attached_wedge,
    swept_oblique_shock,
    stagnation_pressure_ratio,
)
from .newtonian import (
    cp_max_modified_newtonian,
    modified_newtonian_cp,
)
from .tangent_wedge import (
    tangent_wedge_cp,
)
from .oswatitsch import (
    OswatitschResult,
    equal_strength_ramps,
)

__all__ = [
    # oblique_shock
    "ShockDetachedError", "mach_angle", "theta_max",
    "theta_from_beta_M", "obtain_beta",
    "rankine_hugoniot", "cp_attached_wedge",
    "swept_oblique_shock", "stagnation_pressure_ratio",
    # newtonian
    "cp_max_modified_newtonian", "modified_newtonian_cp",
    # tangent_wedge
    "tangent_wedge_cp",
    # oswatitsch
    "OswatitschResult", "equal_strength_ramps",
]
