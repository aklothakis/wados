"""Electromagnetic submodule (PSWR-1).

Phase 3: Born-approximation bistatic RCS for low-contrast plasma sheaths.
"""

from .born_rcs import (
    born_amplitude,
    bistatic_rcs,
    monostatic_rcs,
    rcs_from_sheath,
    cube_validation,
    sphere_form_factor,
    rayleigh_uniform_analytic,
    bistatic_direction_from_angles,
    rcs_dBsm,
)

__all__ = [
    "born_amplitude", "bistatic_rcs", "monostatic_rcs",
    "rcs_from_sheath", "cube_validation",
    "sphere_form_factor", "rayleigh_uniform_analytic",
    "bistatic_direction_from_angles", "rcs_dBsm",
]
