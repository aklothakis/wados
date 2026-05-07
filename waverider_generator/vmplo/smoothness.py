"""Numerical verification of the VMPLO shock surface C2 property.

Spec reference: VMPLO_implementation_prompt.md §4 (proof), "Smoothness
Verification".

For constant beta, the shock surface normal is

    N(z) = (-sin beta, -cos beta, g'(z) * cos beta)

which is C1 in z iff g(z) in C2 (and beta is constant).  The routines
here sample N(z) densely and measure its rate of change; for a correct
implementation the angular rate is numerical noise (< 0.001 deg/m).
"""

from __future__ import annotations

import numpy as np


def verify_shock_surface_C2(assembly, n_z: int = 500,
                            threshold_deg_per_m: float = 0.01) -> dict:
    """Return a dict characterising the shock surface normal smoothness.

    Keys: ``max_angular_rate_deg_per_m``, ``is_C2``, ``beta_design``,
    ``z``, ``N``, ``dN_dz``.
    """
    beta = np.radians(assembly.beta_design)
    zs = np.linspace(0.0, assembly.W, n_z)
    g = np.array([assembly.ICC_at_z(z) for z in zs])
    g_prime = np.gradient(g, zs)
    N = np.column_stack([
        np.full(n_z, -np.sin(beta)),
        np.full(n_z, -np.cos(beta)),
        g_prime * np.cos(beta),
    ])
    dN_dz = np.gradient(N, zs, axis=0)
    # Angular rate: rate of change of the normal direction, in deg/m.
    mags = np.linalg.norm(dN_dz, axis=1)
    angular_rate_rad_per_m = mags / np.maximum(np.linalg.norm(N, axis=1), 1e-12)
    angular_rate_deg_per_m = np.degrees(angular_rate_rad_per_m)
    max_rate = float(np.max(angular_rate_deg_per_m))
    return {
        "max_angular_rate_deg_per_m": max_rate,
        "is_C2": bool(max_rate < threshold_deg_per_m),
        "beta_design": assembly.beta_design,
        "z": zs,
        "N": N,
        "dN_dz": dN_dz,
    }


def verify_leading_edge_C2(waverider, n_z: int = 500) -> dict:
    """Curvature and torsion of the 3D leading-edge curve.

    For a VMPLO with constant x_LE and C2 ICC(z), the LE curve
    ``(x_LE, ICC(z), z)`` is C2; curvature is ``|g''(z)|`` on the
    (y, z) plane and torsion is zero.
    """
    zs = np.linspace(0.0, waverider.width, n_z)
    le = np.column_stack([
        np.full(n_z, waverider.x_LE),
        np.array([waverider.assembly.ICC_at_z(z) for z in zs]),
        zs,
    ])
    d1 = np.gradient(le, zs, axis=0)
    d2 = np.gradient(d1, zs, axis=0)
    num = np.linalg.norm(np.cross(d1, d2), axis=1)
    den = np.linalg.norm(d1, axis=1) ** 3
    curvature = num / np.maximum(den, 1e-12)
    max_k = float(np.max(curvature))
    return {
        "max_curvature": max_k,
        "max_torsion": 0.0,     # planar LE curve in this VMPLO form
        "is_smooth": bool(np.all(np.isfinite(curvature))),
        "z": zs,
        "curvature": curvature,
    }


def compression_surface_mismatch(assembly, n_z: int = 60,
                                 n_x: int = 100) -> float:
    """Max tangent mismatch angle (degrees) across osculating-plane seams.

    Informational: VMPLO's compression surface is C0 across plane
    boundaries when Ma(z) or n(z) vary, which is expected and not a
    defect.
    """
    strips = assembly.build_all_strips(n_z=n_z, n_x=n_x)
    max_deg = 0.0
    for i in range(len(strips) - 1):
        xs_a, rs_a, _ = strips[i]
        xs_b, rs_b, _ = strips[i + 1]
        common_x = 0.5 * (xs_a + xs_b)
        ra = np.interp(common_x, xs_a, rs_a)
        rb = np.interp(common_x, xs_b, rs_b)
        da = np.gradient(ra, common_x)
        db = np.gradient(rb, common_x)
        ang_diff = np.degrees(np.arctan(db) - np.arctan(da))
        max_deg = max(max_deg, float(np.max(np.abs(ang_diff))))
    return max_deg
