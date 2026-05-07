"""Aerodynamic submodule (GVWD §4.8).

Phase 2 ships only :mod:`inviscid` (closed-form coefficients for the
three textbook reference modes). Phase 4 adds the panel-method evaluator
and viscous corrections.
"""

from .inviscid import (
    caret_inviscid_coefficients,
    flat_delta_inviscid_coefficients,
    multi_wedge_inviscid_coefficients,
)
from .panel_method import (
    PanelAeroResult,
    panel_aero_coefficients,
    freestream_direction,
    lift_direction,
)
from .viscous import (
    ViscousResult,
    panel_viscous_drag,
)
from .coefficients import aero_coefficients_full
from .sweep import SweepConfig, mach_alpha_sweep, heatmap_2d

__all__ = [
    "caret_inviscid_coefficients",
    "flat_delta_inviscid_coefficients",
    "multi_wedge_inviscid_coefficients",
    "PanelAeroResult", "panel_aero_coefficients",
    "freestream_direction", "lift_direction",
    "ViscousResult", "panel_viscous_drag",
    "aero_coefficients_full",
    "SweepConfig", "mach_alpha_sweep", "heatmap_2d",
]
