"""Plotting submodule (PSWR-1 Phase 5).

All figures use a paper-quality serif style with three size variants:
``style="paper"`` (default, single-column), ``"slide"`` (16:9 talk),
``"draft"`` (low-DPI for fast iteration). See :func:`apply_style`.
"""

from .style import apply_style, default_caret_label
from .pareto import (
    plot_pareto_3d,
    plot_pareto_projections,
    plot_pareto_full,
)
from .geometry_3d import plot_design_geometry
from .sheath import plot_sheath_contour, plot_rcs_polar

__all__ = [
    "apply_style", "default_caret_label",
    "plot_pareto_3d", "plot_pareto_projections", "plot_pareto_full",
    "plot_design_geometry",
    "plot_sheath_contour", "plot_rcs_polar",
]
