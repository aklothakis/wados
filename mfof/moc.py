"""Method-of-Characteristics adapter for the MFOF framework.

Thin wrapper around the validated axisymmetric MOC solver in
``waverider_generator.vmplo.moc``. The wrapper exists for two reasons:

1. **Import workaround.** ``waverider_generator/__init__.py`` eager-imports
   ``cadquery``, which fails on this environment's NumPy 2.0 (the deprecated
   ``np.bool8`` / ``np.object0`` / ``np.float_`` aliases were removed). A
   plain ``from waverider_generator.vmplo.moc import MOCGrid`` therefore
   raises ``AttributeError`` in a fresh process. The fix is the same
   *fail-then-retry* pattern that ``liu2019.shock`` already uses
   successfully: trigger the parent ``__init__`` once, ignore its failure,
   and then import submodules directly. After the failed init,
   ``waverider_generator`` is partially in ``sys.modules`` and the parent
   ``__init__`` does not re-run. This is **localised to this file**;
   nothing else changes.

2. **Higher-level convenience.** The upstream solver returns just
   ``(x_arr, r_arr)`` from :func:`extract_streamline`. The
   :class:`mfof.basic_flowfield.StreamlineResult` contract also wants
   ``Ma_TE`` and the trailing-edge deflection angle, so we extract those
   from the MOC mesh's last column post-hoc.

Public surface:

* :func:`build_moc_grid` -- one-call factory: ``(Ma, beta, n, L, x_LE,
  theta_LE_deg, gamma, n_columns)`` -> ``(grid, body)``.
* :func:`trace_streamline_with_state` -- ``(grid, x_LE, r_LE,
  alpha0_rad, n_steps)`` -> ``(x_arr, r_arr, Ma_TE, delta_TE_deg)``.

These are the only entry points that :mod:`mfof.power_law_flowfield`
imports.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Import-chain workaround
# ---------------------------------------------------------------------------
# Triggers the ``waverider_generator`` package init once. The init's
# ``from waverider_generator import cad_export, plotting_tools`` line raises
# AttributeError due to the cadquery -> nptyping -> numpy.bool8 chain on
# NumPy >= 2.0. After the failed init Python keeps the partial module in
# ``sys.modules``; subsequent ``from waverider_generator.vmplo.X import Y``
# statements bypass the parent init and succeed.
try:
    import waverider_generator              # noqa: F401  (intentional side-effect)
except Exception:
    pass

from waverider_generator.vmplo.moc import (         # noqa: E402
    MOCGrid,
    initial_data_line,
    extract_streamline,
    prandtl_meyer,
    Ma_from_prandtl_meyer,
)
from waverider_generator.vmplo.powerlaw import PowerLawBody    # noqa: E402


# ---------------------------------------------------------------------------
# Public adapter API
# ---------------------------------------------------------------------------

def build_moc_grid(
    Ma_inf: float,
    beta_design_deg: float,
    n: float,
    L: float,
    x_LE: float,
    theta_LE_deg: float,
    gamma: float = 1.4,
    n_columns: int = 20,
    n_initial_points: int = 12,
) -> Tuple[MOCGrid, PowerLawBody]:
    """Build the axisymmetric MOC mesh for a power-law body.

    Parameters
    ----------
    Ma_inf : float
        Free-stream Mach number.
    beta_design_deg : float
        Design shock-cone angle (degrees).
    n : float
        Power-law body exponent. ``n = 1`` recovers the cone limit; values
        in ``(0, 2)`` are physically meaningful.
    L : float
        Body length (= vehicle length ``L_w`` of the parent waverider).
    x_LE : float
        Leading-edge x-coordinate in the body's frame. Must be > 0 (the
        body has a singular tangent at ``x = 0`` for ``n < 1``).
    theta_LE_deg : float
        Post-shock flow-deflection angle at the LE = body slope at
        ``x_LE``. From the standard theta-beta-M relation.
    gamma : float
        Specific-heat ratio. Default 1.4.
    n_columns : int
        Number of MOC columns to march downstream from the shock. The
        post-shock flow is smooth so the result is insensitive beyond
        ``n_columns ~ 10``; default 20.
    n_initial_points : int
        Number of points along the shock initial-data line. Default 12.

    Returns
    -------
    (grid, body) : (MOCGrid, PowerLawBody)
        ``grid`` carries ``cols``, ``interpolate_alpha(x, r)``,
        ``all_points()``. ``body`` exposes ``radius(x)``, ``slope(x)``,
        ``R_b``, ``n``, ``L``.
    """
    body = PowerLawBody.from_shock_condition(
        n=float(n), L=float(L), x_LE=float(x_LE),
        theta_deg=float(theta_LE_deg), gamma=float(gamma))
    r_LE = float(body.radius(x_LE))
    # Phase 4 Fix 2: pass n_body so initial_data_line can auto-boost the
    # resolution and switch to Chebyshev clustering for low-n bodies.
    # Default n_body=1.0 in the upstream function preserves legacy behaviour
    # for any caller that omits this kwarg.
    initial = initial_data_line(
        x_LE=float(x_LE), r_LE=r_LE,
        beta_design_deg=float(beta_design_deg),
        Ma_inf=float(Ma_inf), gamma=float(gamma),
        N=int(n_initial_points), n_body=float(n))
    grid = MOCGrid(initial_points=initial, body=body, gamma=float(gamma))
    grid.march(n_columns=int(n_columns))
    return grid, body


def trace_streamline_with_state(
    grid: MOCGrid,
    x_LE: float,
    r_LE: float,
    alpha0_rad: float,
    n_steps: int = 10,
    gamma: float = 1.4,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Trace one streamline through the MOC mesh and recover its end-state.

    Parameters
    ----------
    grid : MOCGrid
        MOC mesh from :func:`build_moc_grid`.
    x_LE, r_LE : float
        Streamline starting point on the shock (LE).
    alpha0_rad : float
        Initial flow-angle (post-shock deflection) at the LE, in radians.
    n_steps : int
        Number of RK4 steps. Profiling shows ``r_TE`` already converges by
        ``n_steps = 10`` (verified across 10/30/50/100/300 in the pre-plan
        probe), so we default to 10 to keep per-plane cost ~660 ms instead
        of the upstream default of 300.
    gamma : float
        Specific-heat ratio. Used only to convert Prandtl-Meyer angles back
        into Mach via :func:`Ma_from_prandtl_meyer`.

    Returns
    -------
    (x_arr, r_arr, Ma_TE, delta_TE_deg) : (ndarray, ndarray, float, float)
        Streamline samples, plus the trailing-edge Mach number and
        deflection angle interpolated from the mesh's last column.
    """
    x_arr, r_arr = extract_streamline(
        grid, x0=float(x_LE), r0=float(r_LE),
        alpha0=float(alpha0_rad), n_steps=int(n_steps))

    # Recover Ma_TE and delta_TE_deg from the mesh at (x_arr[-1], r_arr[-1]).
    # MOCGrid exposes interpolate_alpha(x, r) directly. For Mach we walk the
    # last column to find the closest point and use its (Ma, nu) pair.
    x_TE = float(x_arr[-1])
    r_TE = float(r_arr[-1])

    delta_TE_rad = float(grid.interpolate_alpha(x_TE, r_TE))
    delta_TE_deg = float(np.degrees(delta_TE_rad))

    last_col = grid.cols[-1] if grid.cols else []
    if last_col:
        # Nearest-point lookup on the last column. Cheap (n_points ~ 12).
        d2 = [(p["x"] - x_TE) ** 2 + (p["r"] - r_TE) ** 2 for p in last_col]
        nearest = last_col[int(np.argmin(d2))]
        Ma_TE = float(nearest.get("Ma", 0.0))
        # If the point's Ma is missing/zero, fall back via Prandtl-Meyer.
        if not (Ma_TE > 1.0):
            nu = float(nearest.get("nu", 0.0))
            try:
                Ma_TE = float(Ma_from_prandtl_meyer(nu, gamma))
            except Exception:
                Ma_TE = float("nan")
    else:
        Ma_TE = float("nan")

    return x_arr, r_arr, Ma_TE, delta_TE_deg


__all__ = [
    "MOCGrid",
    "PowerLawBody",
    "build_moc_grid",
    "trace_streamline_with_state",
    "prandtl_meyer",
    "Ma_from_prandtl_meyer",
]


# ---------------------------------------------------------------------------
# Smoke test (run directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time
    Ma_inf, beta_deg, n, L, x_LE = 6.0, 13.0, 0.7, 6.0, 1.0
    theta_LE = 4.8

    print(f"build_moc_grid(Ma={Ma_inf}, beta={beta_deg}, n={n}, L={L}, x_LE={x_LE})")
    t0 = time.time()
    grid, body = build_moc_grid(Ma_inf, beta_deg, n, L, x_LE, theta_LE,
                                 n_columns=20)
    print(f"  body.R_b = {body.R_b:.5f}, body.n = {body.n}")
    print(f"  grid: {len(grid.cols)} cols, "
          f"{sum(len(c) for c in grid.cols)} pts in {(time.time()-t0)*1000:.0f} ms")

    print()
    print("trace_streamline_with_state(...) at n_steps=10")
    t0 = time.time()
    x_arr, r_arr, Ma_TE, dTE = trace_streamline_with_state(
        grid, x_LE=x_LE, r_LE=body.radius(x_LE),
        alpha0_rad=np.radians(theta_LE), n_steps=10)
    print(f"  {len(x_arr)} pts in {(time.time()-t0)*1000:.0f} ms")
    print(f"  x:        {x_arr[0]:.4f} -> {x_arr[-1]:.4f}")
    print(f"  r:        {r_arr[0]:.4f} -> {r_arr[-1]:.4f}")
    print(f"  Ma_TE   = {Ma_TE:.4f}")
    print(f"  dTE     = {dTE:.4f} deg")
