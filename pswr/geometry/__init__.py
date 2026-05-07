"""Geometry submodule (PSWR-1 Phase 1)."""

from .variable_wedge import (
    VariableWedgeWaverider,
    BetaSpline,
    to_gui_frame,
)
from .streamlines import trace_lower_streamline, lower_surface_grid
from .volume import volume_efficiency, planform_area, body_volume

__all__ = [
    "VariableWedgeWaverider",
    "BetaSpline",
    "to_gui_frame",
    "trace_lower_streamline",
    "lower_surface_grid",
    "volume_efficiency",
    "planform_area",
    "body_volume",
]
