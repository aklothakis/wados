"""Aerothermal heating submodule (GVWD §4.10).

Phase 4 ships:
  - ``fay_riddell``       : LE / nose stagnation-point convective heat flux
  - ``tauber_sutton``     : Tauber-Sutton correlation (cross-check + radiative)
  - ``eckert_distributed``: distributed surface heat flux via Reynolds analogy
"""

from .fay_riddell import (
    stagnation_point_heat_flux,
    swept_LE_heat_flux,
    nose_heat_flux,
)
from .tauber_sutton import (
    tauber_sutton_convective,
    tauber_sutton_radiative,
)
from .eckert_distributed import distributed_surface_heat_flux

__all__ = [
    "stagnation_point_heat_flux",
    "swept_LE_heat_flux",
    "nose_heat_flux",
    "tauber_sutton_convective",
    "tauber_sutton_radiative",
    "distributed_surface_heat_flux",
]
