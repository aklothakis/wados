"""Publication-quality plots for the GVWD Mach-alpha sweep (§5.5).

All plot functions return the matplotlib Figure so the caller can save
to PDF/PNG or further customize. ``apply_style`` sets paper-quality
serif rcParams; reuse the PSWR-1 style preset where available.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd

# Reuse PSWR-1 style preset where available
try:
    from pswr.viz.style import apply_style as _pswr_apply_style
    _HAS_PSWR_STYLE = True
except Exception:
    _HAS_PSWR_STYLE = False


from gvwd.aero.sweep import heatmap_2d


def apply_style(style: str = "paper") -> None:
    """Set matplotlib rcParams for paper / slide / draft figures."""
    if _HAS_PSWR_STYLE:
        _pswr_apply_style(style)
        return
    # Fallback minimal preset
    presets = {
        "paper": {"font.family": "serif", "axes.titlesize": 11,
                   "axes.labelsize": 10, "savefig.dpi": 600,
                   "axes.grid": True, "grid.alpha": 0.25,
                   "lines.linewidth": 1.2},
        "slide": {"font.family": "sans-serif", "axes.titlesize": 16,
                   "axes.labelsize": 14, "savefig.dpi": 300,
                   "lines.linewidth": 2.0},
        "draft": {"savefig.dpi": 100, "lines.linewidth": 1.0},
    }
    plt.rcParams.update(presets.get(style, presets["paper"]))


# ----------------------------------------------------------------------
#  Heatmaps (M, alpha)
# ----------------------------------------------------------------------

def plot_LD_heatmap(df: pd.DataFrame, *, ax=None,
                    title: Optional[str] = None,
                    style: str = "paper") -> Figure:
    """L/D heatmap over (M, alpha) with contour lines."""
    apply_style(style)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.0, 4.5))
    else:
        fig = ax.get_figure()
    M_vals, a_vals, Z = heatmap_2d(df, "LD")
    A, MM = np.meshgrid(a_vals, M_vals)
    cs = ax.contourf(A, MM, Z, levels=14, cmap="viridis")
    cl = ax.contour(A, MM, Z, levels=8, colors="white",
                     linewidths=0.5, alpha=0.6)
    ax.clabel(cl, fmt="%.1f", fontsize=8)
    fig.colorbar(cs, ax=ax, label="L/D")
    ax.set_xlabel(r"$\alpha$ [deg]"); ax.set_ylabel(r"$M_\infty$")
    ax.set_title(title or "L/D vs Mach and angle of attack")
    fig.tight_layout()
    return fig


def plot_q_LE_heatmap(df: pd.DataFrame, *, ax=None,
                      use_MW: bool = True,
                      title: Optional[str] = None,
                      style: str = "paper") -> Figure:
    """Swept LE heat flux heatmap over (M, alpha)."""
    apply_style(style)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.0, 4.5))
    else:
        fig = ax.get_figure()
    col = "q_LE_swept_MW_m2" if use_MW else "q_LE_swept_W_m2"
    M_vals, a_vals, Z = heatmap_2d(df, col)
    A, MM = np.meshgrid(a_vals, M_vals)
    cs = ax.contourf(A, MM, Z, levels=14, cmap="magma")
    cl = ax.contour(A, MM, Z, levels=8, colors="white",
                     linewidths=0.5, alpha=0.6)
    ax.clabel(cl, fmt="%.0f", fontsize=8)
    fig.colorbar(cs, ax=ax,
                  label=r"$\dot{q}_{LE}$ [MW/m$^2$]" if use_MW
                        else r"$\dot{q}_{LE}$ [W/m$^2$]")
    ax.set_xlabel(r"$\alpha$ [deg]"); ax.set_ylabel(r"$M_\infty$")
    ax.set_title(title or "LE stagnation heat flux (Tauber-Sutton swept)")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
#  1-D slices
# ----------------------------------------------------------------------

def plot_LD_vs_M_at_alpha(df: pd.DataFrame, *, ax=None,
                            title: Optional[str] = None,
                            style: str = "paper") -> Figure:
    """L/D vs M, one curve per alpha."""
    apply_style(style)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
    else:
        fig = ax.get_figure()
    a_vals = sorted(df["alpha_deg"].unique())
    cmap = plt.get_cmap("viridis")
    for k, a in enumerate(a_vals):
        sub = df[df["alpha_deg"] == a].sort_values("M_inf")
        ax.plot(sub["M_inf"], sub["LD"],
                marker="o", markersize=3, linewidth=1.4,
                color=cmap(k / max(len(a_vals)-1, 1)),
                label=fr"$\alpha$={a:.1f}$^\circ$")
    ax.set_xlabel(r"$M_\infty$"); ax.set_ylabel("L/D")
    ax.set_title(title or "L/D vs Mach at each angle of attack")
    ax.legend(fontsize=8, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_q_LE_vs_M(df: pd.DataFrame, *, ax=None, use_MW: bool = True,
                    title: Optional[str] = None,
                    style: str = "paper") -> Figure:
    """Heat flux vs Mach (alpha-dependence is weak; show one curve at
    the smallest alpha plus envelope band)."""
    apply_style(style)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
    else:
        fig = ax.get_figure()
    col = "q_LE_swept_MW_m2" if use_MW else "q_LE_swept_W_m2"
    M_vals = np.array(sorted(df["M_inf"].unique()))
    q_min = np.array([df[df["M_inf"] == M][col].min() for M in M_vals])
    q_max = np.array([df[df["M_inf"] == M][col].max() for M in M_vals])
    q_mean = np.array([df[df["M_inf"] == M][col].mean() for M in M_vals])
    ax.fill_between(M_vals, q_min, q_max, color="#cb4b16", alpha=0.25,
                     label="envelope across alpha")
    ax.plot(M_vals, q_mean, "o-", color="#cb4b16", linewidth=1.5,
             label="mean across alpha")
    ax.set_xlabel(r"$M_\infty$")
    ax.set_ylabel(r"$\dot{q}_{LE}$ [MW/m$^2$]" if use_MW
                  else r"$\dot{q}_{LE}$ [W/m$^2$]")
    ax.set_title(title or "Swept LE heat flux vs Mach")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_polar_CL_vs_CD(df: pd.DataFrame, *, ax=None,
                         title: Optional[str] = None,
                         style: str = "paper") -> Figure:
    """Drag polar: CL vs CD_total, one curve per Mach."""
    apply_style(style)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
    else:
        fig = ax.get_figure()
    M_vals = sorted(df["M_inf"].unique())
    cmap = plt.get_cmap("plasma")
    for k, M in enumerate(M_vals):
        sub = df[df["M_inf"] == M].sort_values("alpha_deg")
        ax.plot(sub["CD_total"], sub["CL"],
                marker="o", markersize=3, linewidth=1.2,
                color=cmap(k / max(len(M_vals)-1, 1)),
                label=fr"$M_\infty$={M:.1f}")
    ax.set_xlabel(r"$C_D$"); ax.set_ylabel(r"$C_L$")
    ax.set_title(title or "Drag polar (CL vs CD across Mach)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best", ncol=2)
    fig.tight_layout()
    return fig


def plot_shock_detachment_diagnostic(df: pd.DataFrame, *, ax=None,
                                       style: str = "paper") -> Figure:
    """Heatmap of (theta_max - max theta_local) margin in degrees:
    POSITIVE = all forebody panels are below detachment, NEGATIVE = at
    least one panel exceeds theta_max and falls back to Newtonian."""
    apply_style(style)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.0, 4.5))
    else:
        fig = ax.get_figure()
    M_vals, a_vals, Z = heatmap_2d(df, "beta_attached_margin_deg")
    A, MM = np.meshgrid(a_vals, M_vals)
    vmax = np.nanmax(np.abs(Z))
    cs = ax.contourf(A, MM, Z, levels=14, cmap="RdBu",
                      vmin=-vmax, vmax=vmax)
    ax.contour(A, MM, Z, levels=[0], colors="black", linewidths=1.2)
    fig.colorbar(cs, ax=ax,
                  label=r"$\theta_{max} - \theta_{local,max}$ [deg]")
    ax.set_xlabel(r"$\alpha$ [deg]"); ax.set_ylabel(r"$M_\infty$")
    ax.set_title("Shock-detachment margin (red = detached)")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
#  Cp centerline (single point)
# ----------------------------------------------------------------------

def plot_cp_centerline(mesh, M_inf: float, alpha_rad: float, *,
                        ax=None, gamma: float = 1.4,
                        title: Optional[str] = None,
                        style: str = "paper") -> Figure:
    """Cp distribution along the centerline (y=0 strip) of the mesh.

    For each face whose centroid has |y| < tol, plot Cp vs streamwise x.
    Useful for seeing the lower-surface pressure distribution at a
    chosen flight point.
    """
    from gvwd.aero.panel_method import panel_aero_coefficients
    apply_style(style)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.0, 3.5))
    else:
        fig = ax.get_figure()
    res = panel_aero_coefficients(mesh, M_inf, alpha_rad, gamma=gamma)
    centroids = mesh.face_centroids()
    # Strip near y = 0
    half_y = float(np.percentile(np.abs(centroids[:, 1]), 30))
    on_centerline = np.abs(centroids[:, 1]) < max(half_y, 1e-3)
    # Split into upper / lower by face normal
    n = mesh.face_normals()
    upper = (n[:, 2] > 0.1) & on_centerline
    lower = (n[:, 2] < -0.1) & on_centerline
    if np.any(lower):
        ax.scatter(centroids[lower, 0], res.Cp[lower],
                    color="#cb4b16", s=20, label="lower (windward)")
    if np.any(upper):
        ax.scatter(centroids[upper, 0], res.Cp[upper],
                    color="#268bd2", s=20, label="upper")
    ax.set_xlabel("x [m]"); ax.set_ylabel(r"$C_p$")
    ax.set_title(title or
                  f"Cp centerline @ M={M_inf:.1f}, "
                  fr"$\alpha$={math.degrees(alpha_rad):.1f}$^\circ$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
#  Combined suite
# ----------------------------------------------------------------------

def plot_full_sweep_suite(df: pd.DataFrame, *, out_dir: Optional[Path] = None,
                            prefix: str = "sweep",
                            style: str = "paper") -> dict:
    """Render all sweep plots in one go. If ``out_dir`` is given, save
    each as PDF + PNG.

    Returns a dict ``{name: Figure}``.
    """
    figs = {
        "LD_heatmap": plot_LD_heatmap(df, style=style),
        "q_LE_heatmap": plot_q_LE_heatmap(df, style=style),
        "LD_vs_M": plot_LD_vs_M_at_alpha(df, style=style),
        "q_LE_vs_M": plot_q_LE_vs_M(df, style=style),
        "polar_CL_vs_CD": plot_polar_CL_vs_CD(df, style=style),
        "shock_detachment": plot_shock_detachment_diagnostic(df, style=style),
    }
    if out_dir is not None:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in figs.items():
            fig.savefig(out_dir / f"{prefix}_{name}.pdf")
            fig.savefig(out_dir / f"{prefix}_{name}.png", dpi=200)
    return figs
