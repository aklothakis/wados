"""Aerodynamic submodule (PSWR-1).

Phase 1 ships only :mod:`inviscid`. Eckert / van Driest II viscous module
arrives in Phase 2.
"""

from .inviscid import (
    inviscid_coefficients,
    cl_cd_caret_analytic,
)
from .viscous import (
    sutherland_viscosity,
    eckert_reference_T,
    cf_laminar,
    cf_turbulent,
    cf_blended,
    boundary_layer_thickness,
    per_station_state,
    viscous_drag_coefficient,
)

__all__ = [
    "inviscid_coefficients", "cl_cd_caret_analytic",
    "sutherland_viscosity", "eckert_reference_T",
    "cf_laminar", "cf_turbulent", "cf_blended",
    "boundary_layer_thickness", "per_station_state",
    "viscous_drag_coefficient",
]
