"""Plotting submodule (GVWD Phase 5).

Publication-quality matplotlib plots for the Mach-alpha sweep,
Cp distribution, polar diagram, and shock-detachment diagnostic.

Style presets reuse the PSWR-1 viz convention via :func:`apply_style`.
"""

from .plotting import (
    apply_style,
    plot_LD_heatmap,
    plot_q_LE_heatmap,
    plot_LD_vs_M_at_alpha,
    plot_q_LE_vs_M,
    plot_polar_CL_vs_CD,
    plot_cp_centerline,
    plot_shock_detachment_diagnostic,
    plot_full_sweep_suite,
)

__all__ = [
    "apply_style",
    "plot_LD_heatmap", "plot_q_LE_heatmap",
    "plot_LD_vs_M_at_alpha", "plot_q_LE_vs_M",
    "plot_polar_CL_vs_CD",
    "plot_cp_centerline",
    "plot_shock_detachment_diagnostic",
    "plot_full_sweep_suite",
]
