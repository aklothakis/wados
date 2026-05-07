"""Pareto-front plots for the 3-objective PSWR-1 problem (§10)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .style import apply_style


def _split_F(F: np.ndarray):
    return -F[:, 0], F[:, 1], -F[:, 2]   # L/D, sigma_dBsm, eta_V


def plot_pareto_3d(F: np.ndarray, *, ax=None,
                   caret_F: Optional[np.ndarray] = None,
                   highlight_F: Optional[np.ndarray] = None,
                   title: str = "Pareto front (3-objective)",
                   style: str = "paper") -> Figure:
    apply_style(style)
    if ax is None:
        fig = plt.figure(figsize=(7.0, 5.5))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()
    LD, sig, eta = _split_F(F)
    ax.scatter(LD, sig, eta, c="#cb4b16", s=22, alpha=0.85,
               edgecolor='k', linewidth=0.3, label="Pareto solutions")
    if caret_F is not None and caret_F.size > 0:
        cLD, csig, ceta = _split_F(np.atleast_2d(caret_F))
        ax.scatter(cLD, csig, ceta, c="#268bd2", marker="X", s=110,
                   edgecolor='k', linewidth=0.6, label="caret baseline")
    if highlight_F is not None and highlight_F.size > 0:
        hLD, hsig, heta = _split_F(np.atleast_2d(highlight_F))
        ax.scatter(hLD, hsig, heta, c="#2aa198", marker="*", s=180,
                   edgecolor='k', linewidth=0.5, label="best compromise")
    ax.set_xlabel("L/D"); ax.set_ylabel(r"max $\sigma_b$ [dBsm]")
    ax.set_zlabel(r"$\eta_V$")
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_pareto_projections(F: np.ndarray, *,
                            caret_F: Optional[np.ndarray] = None,
                            highlight_F: Optional[np.ndarray] = None,
                            title: str = "Pareto projections",
                            style: str = "paper") -> Figure:
    apply_style(style)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))
    LD, sig, eta = _split_F(F)
    pairs = [
        (LD, sig, axes[0], "L/D", r"$\sigma_b$ [dBsm]"),
        (LD, eta, axes[1], "L/D", r"$\eta_V$"),
        (sig, eta, axes[2], r"$\sigma_b$ [dBsm]", r"$\eta_V$"),
    ]
    for x, y, ax, xl, yl in pairs:
        ax.scatter(x, y, c="#cb4b16", s=14, alpha=0.85, edgecolor='k', linewidth=0.2)
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(True, alpha=0.3)
    if caret_F is not None and caret_F.size > 0:
        cLD, csig, ceta = _split_F(np.atleast_2d(caret_F))
        axes[0].scatter(cLD, csig, marker="X", s=80, c="#268bd2", edgecolor='k', linewidth=0.5, zorder=5)
        axes[1].scatter(cLD, ceta, marker="X", s=80, c="#268bd2", edgecolor='k', linewidth=0.5, zorder=5)
        axes[2].scatter(csig, ceta, marker="X", s=80, c="#268bd2", edgecolor='k', linewidth=0.5, zorder=5)
    if highlight_F is not None and highlight_F.size > 0:
        hLD, hsig, heta = _split_F(np.atleast_2d(highlight_F))
        axes[0].scatter(hLD, hsig, marker="*", s=120, c="#2aa198", edgecolor='k', linewidth=0.4, zorder=6)
        axes[1].scatter(hLD, heta, marker="*", s=120, c="#2aa198", edgecolor='k', linewidth=0.4, zorder=6)
        axes[2].scatter(hsig, heta, marker="*", s=120, c="#2aa198", edgecolor='k', linewidth=0.4, zorder=6)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_pareto_full(F: np.ndarray, *,
                     caret_F: Optional[np.ndarray] = None,
                     highlight_F: Optional[np.ndarray] = None,
                     out_dir: Optional[Path] = None,
                     prefix: str = "pareto",
                     caption: str = "",
                     style: str = "paper") -> Tuple[Figure, Figure]:
    """Render both 3-D and projections; save to ``out_dir`` if given."""
    fig3d = plot_pareto_3d(F, caret_F=caret_F, highlight_F=highlight_F,
                            title=f"Pareto 3-D {caption}".strip(),
                            style=style)
    fig2d = plot_pareto_projections(F, caret_F=caret_F, highlight_F=highlight_F,
                                     title=f"Pareto projections {caption}".strip(),
                                     style=style)
    if out_dir is not None:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        fig3d.savefig(out_dir / f"{prefix}_3d.pdf")
        fig3d.savefig(out_dir / f"{prefix}_3d.png", dpi=200)
        fig2d.savefig(out_dir / f"{prefix}_LD_vs_RCS.pdf")
        fig2d.savefig(out_dir / f"{prefix}_projections.pdf")
    return fig3d, fig2d
