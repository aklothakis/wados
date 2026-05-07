"""Geometry submodule (GVWD §4.6, §5.2).

Phase 2 ships three reference modes (caret, flat-delta, multi-wedge), a
common ``Mesh`` dataclass, and analytic/numerical volume utilities.
Phase 3 adds the engineering glide-vehicle generators.
"""

from .mesh import Mesh, mesh_volume_signed
from .caret import Caret
from .flat_delta import FlatDelta
from .multi_wedge import MultiWedge
from .engineering_flat import EngineeringFlat
from .engineering_shallow_v import EngineeringShallowV
from .fins import (
    FinParams, generate_fins, diamond_LE_TE_half_angles, merge_meshes,
)
from .volume import (
    caret_analytic_volume,
    flat_delta_analytic_volume,
    numerical_volume,
    eta_V,
    planform_area_from_mesh,
)

__all__ = [
    "Mesh", "mesh_volume_signed",
    "Caret", "FlatDelta", "MultiWedge",
    "EngineeringFlat", "EngineeringShallowV",
    "FinParams", "generate_fins", "diamond_LE_TE_half_angles", "merge_meshes",
    "caret_analytic_volume", "flat_delta_analytic_volume",
    "numerical_volume", "eta_V", "planform_area_from_mesh",
]
