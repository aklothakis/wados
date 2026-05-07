"""Lower-surface streamline tracing (PSWR-1 §5.1).

For the variable-wedge family the post-shock flow direction at each spanwise
station y is

    v_hat(y) = (cos theta(y), 0, -sin theta(y))

so streamlines are straight lines that stay at constant y.  The exact tracing
is therefore trivial; this module exists for API completeness and to provide a
verification that the geometry returned by :class:`VariableWedgeWaverider`
indeed has streamlines parallel to ``v_hat(y)``.
"""

from __future__ import annotations

import math
import numpy as np

from .variable_wedge import VariableWedgeWaverider


def trace_lower_streamline(wr: VariableWedgeWaverider, j: int,
                           n: int = 50) -> np.ndarray:
    """Return ``(n, 3)`` exact streamline at the j-th spanwise station."""
    if not (0 <= j < len(wr.lower_surface_streams)):
        raise IndexError(j)
    x0, yj, z0 = wr.leading_edge[j]
    theta = wr.theta_y[j]
    chord = wr.body_length - x0
    t = np.linspace(0.0, max(chord, 0.0), n)
    cos_th = math.cos(theta)
    sin_th = math.sin(theta)
    xs = x0 + t * cos_th
    zs = z0 - t * sin_th
    return np.column_stack([xs, np.full_like(t, yj), zs])


def lower_surface_grid(wr: VariableWedgeWaverider) -> tuple:
    """Pack the lower-surface streamlines into structured ``(X, Y, Z)`` arrays
    of shape ``(n_span, n_chord)`` for surface plotting and integration.
    """
    streams = wr.lower_surface_streams
    n_span = len(streams)
    n_chord = max(s.shape[0] for s in streams)
    X = np.zeros((n_span, n_chord))
    Y = np.zeros((n_span, n_chord))
    Z = np.zeros((n_span, n_chord))
    for i, s in enumerate(streams):
        # zero-pad short streamlines (only happens at exact tip)
        m = s.shape[0]
        X[i, :m] = s[:, 0]
        Y[i, :m] = s[:, 1]
        Z[i, :m] = s[:, 2]
        if m < n_chord:
            X[i, m:] = s[-1, 0]
            Y[i, m:] = s[-1, 1]
            Z[i, m:] = s[-1, 2]
    return X, Y, Z


def streamline_alignment_residual(wr: VariableWedgeWaverider) -> float:
    """Max deviation (radians) between numerical streamline tangents and v_hat(y).

    Used as a Phase-1 unit-test gate: must be < 1e-12 for the variable-wedge
    family (closed-form straight lines).
    """
    max_err = 0.0
    for j, s in enumerate(wr.lower_surface_streams):
        if s.shape[0] < 2:
            continue
        d = s[1:] - s[:-1]
        norms = np.linalg.norm(d, axis=1)
        valid = norms > 1e-14
        if not np.any(valid):
            continue   # degenerate (tip) streamline
        cos_th = math.cos(wr.theta_y[j])
        sin_th = math.sin(wr.theta_y[j])
        v_hat = np.array([cos_th, 0.0, -sin_th])
        cos_meas = (d[valid] @ v_hat) / norms[valid]
        err = float(np.max(np.abs(1.0 - cos_meas)))
        max_err = max(max_err, err)
    return max_err
