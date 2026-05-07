"""Visualization functions for the variable-Mach power-law waverider.

Provides plots for spanwise distributions, base-plane comparisons,
compression surface depth maps, and volume-vs-exponent parameter studies.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_spanwise_distributions(Ma_dist=None, n_dist=None, beta_dist=None,
                                half_span=None, n_eval=200):
    """Plot Ma(z), n(z), beta(z) distributions side by side.

    Parameters
    ----------
    Ma_dist, n_dist, beta_dist : SpanwiseDistribution or None
        Distributions to plot.  Panels for None distributions are omitted.
    half_span : float
        Half-span (required if any distribution is provided).
    n_eval : int
        Number of evaluation points.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    dists = []
    labels = []
    units = []
    if Ma_dist is not None:
        dists.append(Ma_dist); labels.append("Mach number"); units.append("Ma")
    if n_dist is not None:
        dists.append(n_dist); labels.append("Power-law exponent"); units.append("n")
    if beta_dist is not None:
        dists.append(beta_dist); labels.append("Shock angle"); units.append(r"$\beta$ [deg]")

    if not dists:
        raise ValueError("At least one distribution must be provided.")
    if half_span is None:
        half_span = dists[0].half_span

    n_panels = len(dists)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4), squeeze=False)
    axes = axes[0]
    z_eval = np.linspace(0, half_span, n_eval)

    for ax, dist, label, unit in zip(axes, dists, labels, units):
        vals = dist(z_eval)
        ax.plot(z_eval, vals, 'b-', linewidth=2)
        # Plot control points
        z_cp = np.linspace(0, half_span, dist.n_cp)
        ax.plot(z_cp, dist.control_points, 'ro', markersize=6, label="Control points")
        ax.set_xlabel("z [m]")
        ax.set_ylabel(unit)
        ax.set_title(label)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_base_plane_comparison(wr_standard, wr_extended, title="Base plane comparison"):
    """Overlay base-plane cross-sections of a standard and extended waverider.

    Parameters
    ----------
    wr_standard : object
        Standard OC waverider (has upper_surface_streams, lower_surface_streams).
    wr_extended : object
        Extended variable-Mach waverider.
    title : str
        Figure title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Extract base-plane cross-section (last point of each stream)
    for wr, label, ls in [(wr_standard, "Standard OC", '--'),
                           (wr_extended, "Variable-Mach", '-')]:
        z_upper, y_upper = [], []
        z_lower, y_lower = [], []
        for stream in wr.upper_surface_streams:
            if stream.shape[0] >= 2:
                z_upper.append(stream[-1, 2])
                y_upper.append(stream[-1, 1])
        for stream in wr.lower_surface_streams:
            if stream.shape[0] >= 2:
                z_lower.append(stream[-1, 2])
                y_lower.append(stream[-1, 1])

        ax.plot(z_upper, y_upper, f'b{ls}', linewidth=1.5, label=f"{label} upper")
        ax.plot(z_lower, y_lower, f'r{ls}', linewidth=1.5, label=f"{label} lower")

    ax.set_xlabel("z [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_compression_surface_depth(wr_extended, title="Compression surface depth"):
    """Color-coded plot of compression surface depth across the span.

    Depth = y_upper - y_lower at matching spanwise stations at the base plane.

    Parameters
    ----------
    wr_extended : object
        Waverider with upper_surface_streams and lower_surface_streams.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    n_streams = min(len(wr_extended.upper_surface_streams),
                    len(wr_extended.lower_surface_streams))

    z_vals = []
    depth_vals = []

    for i in range(n_streams):
        us = wr_extended.upper_surface_streams[i]
        ls = wr_extended.lower_surface_streams[i]
        if us.shape[0] < 2 or ls.shape[0] < 2:
            continue
        z_vals.append(us[-1, 2])
        depth_vals.append(us[-1, 1] - ls[-1, 1])

    fig, ax = plt.subplots(figsize=(8, 4))
    sc = ax.scatter(z_vals, depth_vals, c=depth_vals, cmap='viridis', s=60, edgecolors='k')
    ax.plot(z_vals, depth_vals, 'k-', alpha=0.4)
    plt.colorbar(sc, ax=ax, label="Depth [m]")
    ax.set_xlabel("z [m]")
    ax.set_ylabel("Depth (y_upper - y_lower) [m]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_volume_vs_exponent(M_inf, beta, height, width, dp, n_values,
                            n_planes=15, n_streamwise=40):
    """Parameter study: volume vs power-law exponent.

    Generates waveriders for different constant n values and plots volume.

    Parameters
    ----------
    M_inf : float
        Freestream Mach number.
    beta : float
        Shock angle [degrees].
    height, width : float
        Geometry parameters.
    dp : list
        [X1, X2, X3, X4] design parameters.
    n_values : list[float]
        Power-law exponents to test (e.g. [0.6, 0.7, 0.8, 0.9, 1.0]).
    n_planes, n_streamwise : int
        Resolution parameters.

    Returns
    -------
    fig : matplotlib.figure.Figure
    volumes : dict
        {n_value: volume} mapping.
    """
    from waverider_generator.vmplo.bspline import BSpline1D
    from waverider_generator.vmplo.osculating import OsculatingAssembly
    from waverider_generator.vmplo.geometry import VMPLOWaverider

    # ``dp`` (X1-X4) is unused in the new VMPLO method; we derive a
    # reasonable default ICC and vehicle length from width/height so
    # the sweep still runs without the caller having to change.
    half_span = width / 2.0 if width > 0 else width
    W = width   # treat ``width`` as half-span (matches VMPLO convention)
    L = 3.0 * height if height > 0 else 3.0
    x_LE = 0.05
    icc_center = 0.95 * height
    icc_tip = 0.30 * height
    volumes = {}

    Ma_sp = BSpline1D.constant(M_inf, 0.0, W, n_internal_knots=4)
    icc_sp = BSpline1D.linear(icc_center, icc_tip, 0.0, W, n_internal_knots=4)

    for n_val in n_values:
        n_sp = BSpline1D.constant(n_val, 0.0, W, n_internal_knots=4)
        try:
            assembly = OsculatingAssembly(
                Ma_spline=Ma_sp, n_spline=n_sp,
                ICC_spline=icc_sp, US_spline=None,
                beta_design=beta, L=L, W=W, H=height, x_LE=x_LE)
            wr = VMPLOWaverider(assembly, n_planes=n_planes,
                                n_streamwise=n_streamwise)
            volumes[n_val] = wr.compute_volume()
        except Exception as e:
            print(f"  n={n_val:.2f} failed: {e}")
            volumes[n_val] = np.nan

    fig, ax = plt.subplots(figsize=(7, 5))
    ns = sorted(volumes.keys())
    vols = [volumes[n] for n in ns]
    ax.plot(ns, vols, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel("Power-law exponent n")
    ax.set_ylabel("Volume [m^3]")
    ax.set_title("Internal volume vs power-law exponent")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, volumes
