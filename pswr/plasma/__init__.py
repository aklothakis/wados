"""Plasma submodule (PSWR-1).

Phase 2 ships :mod:`sheath` for n_e(r) on a structured grid.
Phase 3 will add :mod:`permittivity` (Drude epsilon) and :mod:`em.born_rcs`.
"""

from .sheath import (
    SheathGrid,
    build_sheath_grid,
    plasma_frequency,
    electron_collision_frequency,
)
from .permittivity import (
    drude_permittivity,
    susceptibility,
    critical_density,
    born_validity,
)

__all__ = [
    "SheathGrid",
    "build_sheath_grid",
    "plasma_frequency",
    "electron_collision_frequency",
    "drude_permittivity",
    "susceptibility",
    "critical_density",
    "born_validity",
]
