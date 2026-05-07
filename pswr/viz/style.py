"""Publication-figure rcParams for matplotlib (PSWR-1 §10)."""

from __future__ import annotations

import matplotlib as mpl


_STYLE_PRESETS = {
    "paper": {
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "font.family": "serif",
        "savefig.dpi": 600,
        "figure.dpi": 120,
        "lines.linewidth": 1.2,
        "axes.grid": True,
        "grid.alpha": 0.25,
    },
    "slide": {
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "font.family": "sans-serif",
        "savefig.dpi": 300,
        "figure.dpi": 150,
        "lines.linewidth": 2.0,
        "axes.grid": True,
        "grid.alpha": 0.3,
    },
    "draft": {
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "savefig.dpi": 100,
        "figure.dpi": 100,
        "lines.linewidth": 1.0,
        "axes.grid": True,
        "grid.alpha": 0.25,
    },
}


def apply_style(style: str = "paper") -> None:
    if style not in _STYLE_PRESETS:
        raise ValueError(f"unknown style {style!r}; must be one of "
                         f"{list(_STYLE_PRESETS)}")
    mpl.rcParams.update(_STYLE_PRESETS[style])


def default_caret_label(M_inf: float, beta_deg: float, Lambda_deg: float) -> str:
    return (f"caret baseline (M={M_inf:.1f}, "
            f"beta={beta_deg:.1f} deg, Lambda={Lambda_deg:.0f} deg)")
