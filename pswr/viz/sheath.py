"""Sheath n_e contour and sigma_b polar plots (PSWR-1 §10)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .style import apply_style


def plot_sheath_contour(grid, *, j_slice: Optional[int] = None,
                        ax=None, title: str = r"$n_e$ midspan slice",
                        style: str = "paper") -> Figure:
    apply_style(style)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 3.0))
    else:
        fig = ax.get_figure()
    n_chord, n_span, n_norm = grid.shape
    if j_slice is None:
        j_slice = n_span // 2
    s = grid.X[:, j_slice, 0] - grid.X[0, j_slice, 0]
    zeta = np.linalg.norm(np.stack([
        grid.X[0, j_slice, :] - grid.X[0, j_slice, 0],
        grid.Y[0, j_slice, :] - grid.Y[0, j_slice, 0],
        grid.Z[0, j_slice, :] - grid.Z[0, j_slice, 0],
    ], axis=0), axis=0)
    S, Zeta = np.meshgrid(s, zeta, indexing="ij")
    ne = grid.n_e[:, j_slice, :]
    ne_log = np.where(ne > 0, np.log10(np.maximum(ne, 1e-30)), np.nan)
    if np.all(np.isnan(ne_log)):
        ax.text(0.5, 0.5, r"$n_e \approx 0$ in this case",
                transform=ax.transAxes, ha="center", va="center",
                color="gray", fontsize=10)
    else:
        cs = ax.contourf(S, Zeta, ne_log, levels=14, cmap="magma")
        cb = fig.colorbar(cs, ax=ax,
                           label=r"$\log_{10} n_e$  [m$^{-3}$]")
    ax.set_xlabel(r"streamwise $s = x - x_{LE}$  [m]")
    ax.set_ylabel(r"wall-normal $\zeta$  [m]")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_rcs_polar(theta_deg: np.ndarray, sigma_dBsm: np.ndarray, *,
                   caret_dBsm: Optional[np.ndarray] = None,
                   title: str = r"Bistatic $\sigma_b(\theta_s)$",
                   style: str = "paper") -> Figure:
    apply_style(style)
    fig = plt.figure(figsize=(5.0, 4.5))
    ax = fig.add_subplot(111, projection="polar")
    theta = np.deg2rad(np.asarray(theta_deg))
    ax.plot(theta, np.asarray(sigma_dBsm),
            color="#cb4b16", linewidth=1.5, label="design")
    if caret_dBsm is not None:
        ax.plot(theta, np.asarray(caret_dBsm),
                color="#268bd2", linewidth=1.2, linestyle="--", label="caret")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(title, pad=15)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18),
              ncol=2, frameon=False)
    fig.tight_layout()
    return fig
