"""3-D geometry rendering for selected Pareto designs (PSWR-1 §10)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from .style import apply_style
from ..geometry.variable_wedge import VariableWedgeWaverider, to_gui_frame


def plot_design_geometry(wr: VariableWedgeWaverider, *,
                         ax=None, half_only: bool = False,
                         title: str = "Variable-wedge waverider",
                         style: str = "paper") -> Figure:
    apply_style(style)
    if ax is None:
        fig = plt.figure(figsize=(6.5, 5.5))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    # Lower surface streamlines (post-shock side)
    for s in wr.lower_surface_streams:
        if s.shape[0] < 2: continue
        gs = to_gui_frame(s)
        ax.plot(gs[:, 2], gs[:, 0], gs[:, 1], color="#cb4b16",
                alpha=0.5, linewidth=0.6)
    # Upper surface lines
    for j in range(wr.upper_surface.shape[0]):
        seg = to_gui_frame(wr.upper_surface[j])
        ax.plot(seg[:, 2], seg[:, 0], seg[:, 1], color="#268bd2",
                alpha=0.45, linewidth=0.6)
    le = to_gui_frame(wr.leading_edge)
    ax.plot(le[:, 2], le[:, 0], le[:, 1], "k-", linewidth=2.2)

    ax.set_xlabel("Z (span) [m]")
    ax.set_ylabel("X (streamwise) [m]")
    ax.set_zlabel("Y (vertical) [m]")
    b0, b1, b2 = wr.beta_knots_deg
    ax.set_title(f"{title}\n"
                 r"M$_\infty$" + f"={wr.M_inf:.2f}, "
                 r"$\beta$=" + f"({b0:.1f}, {b1:.1f}, {b2:.1f}) deg, "
                 r"$\Lambda$=" + f"{wr.Lambda_deg:.0f} deg")
    # Equal axis
    try:
        limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
        center = np.mean(limits, axis=1)
        radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
        ax.set_xlim3d([center[0] - radius, center[0] + radius])
        ax.set_ylim3d([center[1] - radius, center[1] + radius])
        ax.set_zlim3d([center[2] - radius, center[2] + radius])
    except Exception:
        pass
    fig.tight_layout()
    return fig


def render_three_designs(wr_best_LD: VariableWedgeWaverider,
                         wr_best_RCS: VariableWedgeWaverider,
                         wr_compromise: VariableWedgeWaverider,
                         out_dir: Path, *, style: str = "paper") -> None:
    """Save three PNG/PDF figures: best L/D, best RCS, best compromise."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for wr, tag in ((wr_best_LD, "best_LD"),
                    (wr_best_RCS, "best_RCS"),
                    (wr_compromise, "best_compromise")):
        fig = plot_design_geometry(wr, title=f"Geometry — {tag.replace('_', ' ')}",
                                    style=style)
        fig.savefig(out_dir / f"geometry_{tag}.png", dpi=200)
        fig.savefig(out_dir / f"geometry_{tag}.pdf")
        plt.close(fig)
