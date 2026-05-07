"""Design-vector packing and feasibility checks for VMPLO.

Spec reference: VMPLO_implementation_prompt.md §8, "Module
Specifications" for ``vmplo/design_space.py``.

The design vector collects the scalar ``beta_design`` plus the B-spline
coefficient arrays for Ma(z), n(z), and ICC y_ICC(z).  Total length:
``1 + (k+4) + (k+4) + (m+4)``, i.e. 25 for the default ``k = m = 4``.
"""

from __future__ import annotations

import numpy as np

from waverider_generator.vmplo.bspline import BSpline1D
from waverider_generator.vmplo.shock import (
    beta_detachment, mach_angle, DetachedShockError,
)


DEFAULT_PARAMS: dict = {
    "L":             3.0,
    "W":             2.0,
    "H":             0.69,
    "x_LE":          0.05,
    "beta_design":  13.0,
    "Ma_center":     6.0,
    "Ma_tip":       10.0,
    "n_center":      0.7,
    "n_tip":         1.0,
    "icc_center":    0.95,   # fraction of H
    "icc_tip":       0.30,
    "n_knots_Ma":    4,
    "n_knots_n":     4,
    "n_knots_ICC":   4,
    "gamma":         1.4,
    "Ma_eval":      [6.0, 8.0, 10.0, 13.0],
    "Ma_weights":   [0.25, 0.25, 0.25, 0.25],
}


BOUNDS: dict = {
    "beta_design": (10.0, 35.0),
    "Ma":          (4.0, 14.0),
    "n":           (0.4, 1.2),
    "y_ICC_norm":  (0.05, 0.95),
}


def _spline_size(n_internal_knots: int) -> int:
    """Number of control coefficients for a clamped cubic B-spline."""
    return n_internal_knots + 4


def build_design_vector(beta_design: float,
                        Ma_coeffs: np.ndarray,
                        n_coeffs: np.ndarray,
                        ICC_coeffs: np.ndarray) -> np.ndarray:
    """Pack into a 1-D numpy array [beta, c_Ma..., c_n..., c_ICC...]."""
    return np.concatenate([
        np.array([float(beta_design)]),
        np.asarray(Ma_coeffs, dtype=float),
        np.asarray(n_coeffs, dtype=float),
        np.asarray(ICC_coeffs, dtype=float),
    ])


def unpack_design_vector(x: np.ndarray,
                         n_knots_Ma: int = 4,
                         n_knots_n: int = 4,
                         n_knots_ICC: int = 4
                         ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Unpack a design vector.

    Returns ``(beta_design, Ma_coeffs, n_coeffs, ICC_coeffs)``.
    """
    x = np.asarray(x, dtype=float)
    n_ma = _spline_size(n_knots_Ma)
    n_n = _spline_size(n_knots_n)
    n_icc = _spline_size(n_knots_ICC)
    expected = 1 + n_ma + n_n + n_icc
    if x.size != expected:
        raise ValueError(
            f"Design vector length {x.size} != expected {expected}.")
    i = 0
    beta = float(x[i]); i += 1
    Ma_c = x[i:i + n_ma]; i += n_ma
    n_c = x[i:i + n_n]; i += n_n
    ICC_c = x[i:i + n_icc]
    return beta, Ma_c, n_c, ICC_c


def default_design_vector(params: dict | None = None) -> np.ndarray:
    """Default design vector derived from ``DEFAULT_PARAMS``.

    The Ma, n, and ICC distributions are initialised as linear fits
    from their centre value to their tip value.
    """
    if params is None:
        params = DEFAULT_PARAMS

    k_ma = params["n_knots_Ma"]
    k_n = params["n_knots_n"]
    k_icc = params["n_knots_ICC"]
    W = params["W"]
    H = params["H"]

    Ma_sp = BSpline1D.linear(
        params["Ma_center"], params["Ma_tip"],
        0.0, W, n_internal_knots=k_ma)
    n_sp = BSpline1D.linear(
        params["n_center"], params["n_tip"],
        0.0, W, n_internal_knots=k_n)
    icc_sp = BSpline1D.linear(
        params["icc_center"] * H, params["icc_tip"] * H,
        0.0, W, n_internal_knots=k_icc)

    return build_design_vector(
        params["beta_design"],
        Ma_sp.to_coefficients(),
        n_sp.to_coefficients(),
        icc_sp.to_coefficients(),
    )


def check_feasibility(x: np.ndarray,
                      params: dict | None = None,
                      n_z: int = 50
                      ) -> list[tuple[str, float, bool]]:
    """Check design-vector feasibility on a dense z-grid.

    Returns a list of ``(constraint_name, worst_value, passed)`` tuples.
    """
    if params is None:
        params = DEFAULT_PARAMS

    beta, Ma_c, n_c, ICC_c = unpack_design_vector(
        x,
        n_knots_Ma=params["n_knots_Ma"],
        n_knots_n=params["n_knots_n"],
        n_knots_ICC=params["n_knots_ICC"],
    )

    W = params["W"]
    Ma_sp = BSpline1D(0.0, W, n_internal_knots=params["n_knots_Ma"],
                       symmetry=True).from_coefficients(Ma_c)
    n_sp = BSpline1D(0.0, W, n_internal_knots=params["n_knots_n"],
                      symmetry=True).from_coefficients(n_c)
    icc_sp = BSpline1D(0.0, W, n_internal_knots=params["n_knots_ICC"],
                        symmetry=True).from_coefficients(ICC_c)

    zs = np.linspace(0.0, W, n_z)
    Ma_vals = np.array([float(Ma_sp(z)) for z in zs])
    n_vals = np.array([float(n_sp(z)) for z in zs])
    icc_vals = np.array([float(icc_sp(z)) for z in zs])

    results: list[tuple[str, float, bool]] = []

    # 1. beta above Mach angle everywhere
    min_mu = max(0.0, float(np.nanmin([mach_angle(m) for m in Ma_vals])))
    results.append(("beta > mu(Ma)", beta - min_mu, beta > min_mu))

    # 2. beta below detachment everywhere
    try:
        max_det = max(beta_detachment(m) for m in Ma_vals)
    except Exception:
        max_det = 90.0
    results.append(("beta < beta_det(Ma)", max_det - beta, beta < max_det))

    # 3-4. n, Ma within bounds
    Ma_lo, Ma_hi = BOUNDS["Ma"]
    n_lo, n_hi = BOUNDS["n"]
    results.append(("Ma in bounds", float(np.min(Ma_vals) - Ma_lo),
                    bool(np.all((Ma_vals >= Ma_lo) & (Ma_vals <= Ma_hi)))))
    results.append(("n in bounds", float(np.min(n_vals) - n_lo),
                    bool(np.all((n_vals >= n_lo) & (n_vals <= n_hi)))))

    # 5. ICC positive and monotone or at least within normalised bounds
    H = params["H"]
    y_lo, y_hi = BOUNDS["y_ICC_norm"]
    icc_norm = icc_vals / max(H, 1e-9)
    results.append(("y_ICC/H in bounds", float(np.min(icc_norm) - y_lo),
                    bool(np.all((icc_norm >= y_lo) & (icc_norm <= y_hi)))))

    # 6. x_LE positive
    x_LE = params["x_LE"]
    results.append(("x_LE > 0", float(x_LE), x_LE > 0.0))

    return results
