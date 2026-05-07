"""Aerodynamic evaluation for the Liu 2019 waverider.

Implementation
--------------
Uses **modified Newtonian** impact theory on the triangulated surfaces.
Newtonian theory is the closest analytic analogue to the inviscid Euler
CFD reported in paper Fig. 12 and is sufficient for trend validation
(CL, CD, L/D, Cmz vs Ma) within the paper's +/- 10% aero tolerance.

If PySAGAS is installed, the evaluator can be extended to call it via
``Liu2019AeroEvaluator.evaluate(..., solver="pysagas")``.
"""

from typing import Dict, Optional

import numpy as np

from .config import (
    MOMENT_REF,
    PAPER_REFERENCE_AERO,
    PAPER_TRAJECTORY,
    REF_AREA_M2,
    REF_LENGTH_M,
)
from .geometry import Liu2019Waverider


# ---------------------------------------------------------------------------
# Newtonian Cp
# ---------------------------------------------------------------------------

def _cp_max_newtonian(Ma, gamma=1.4):
    """Modified-Newtonian Cp_max (stagnation Cp behind a normal shock)."""
    if Ma <= 1.0:
        return 2.0
    M2 = Ma * Ma
    p02_over_p1 = (
        ((gamma + 1.0) ** 2 * M2) /
        (4.0 * gamma * M2 - 2.0 * (gamma - 1.0))
    ) ** (gamma / (gamma - 1.0)) * (
        (1.0 - gamma + 2.0 * gamma * M2) / (gamma + 1.0)
    )
    return (2.0 / (gamma * M2)) * (p02_over_p1 - 1.0)


def _panel_triangles(X, Y, Z):
    """Return (N, 3, 3) array of triangle vertices for a structured mesh."""
    P = np.stack([X, Y, Z], axis=-1)
    p0 = P[:-1, :-1]
    p1 = P[1:,  :-1]
    p2 = P[1:,  1:]
    p3 = P[:-1, 1:]
    tri_a = np.stack([p0, p1, p2], axis=-2).reshape(-1, 3, 3)
    tri_b = np.stack([p0, p2, p3], axis=-2).reshape(-1, 3, 3)
    return np.concatenate([tri_a, tri_b], axis=0)


def _tri_areas_centroids_normals(tris, outward_sign):
    """Compute per-triangle area, centroid, outward unit normal.

    ``outward_sign`` = +1 for upper surface (normal should point +y, away
    from the vehicle interior), -1 for lower surface (normal should point
    -y, into the freestream).
    """
    v0 = tris[:, 0]
    v1 = tris[:, 1]
    v2 = tris[:, 2]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    centroids = (v0 + v1 + v2) / 3.0
    with np.errstate(invalid="ignore", divide="ignore"):
        normals = cross / np.where(areas[:, None] > 0, 2.0 * areas[:, None], 1.0)
    if outward_sign > 0:
        flip = normals[:, 1] < 0
    else:
        flip = normals[:, 1] > 0
    normals[flip] *= -1.0
    return areas, centroids, normals


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Liu2019AeroEvaluator:
    """Newtonian impact-theory evaluator for a Liu 2019 waverider.

    Reference length = 6.0 m, reference area = 1.0 m2 (paper Sec. 4.2).
    Pitching moment is taken about the nose (0, 0, 0).
    """

    def __init__(self, waverider: Liu2019Waverider,
                 gamma: float = 1.4,
                 ref_length: float = REF_LENGTH_M,
                 ref_area: float = REF_AREA_M2,
                 moment_ref=MOMENT_REF):
        self.wr = waverider
        self.gamma = float(gamma)
        self.ref_length = float(ref_length)
        self.ref_area = float(ref_area)
        self.moment_ref = np.asarray(moment_ref, dtype=float)

        X_u, Y_u, Z_u = waverider.upper_surface(mirror=True)
        X_l, Y_l, Z_l = waverider.lower_surface(mirror=True)
        self._upper_tris = _panel_triangles(X_u, Y_u, Z_u)
        self._lower_tris = _panel_triangles(X_l, Y_l, Z_l)

    # ------------------------------------------------------------------
    def _evaluate_surface(self, tris, outward_sign, flow_dir, Cp_max):
        areas, centroids, normals = _tri_areas_centroids_normals(
            tris, outward_sign)
        # Impact angle: sin(theta) = -n . flow_dir (windward => positive).
        cos_theta = -np.einsum("ij,j->i", normals, flow_dir)
        cos_theta = np.clip(cos_theta, 0.0, 1.0)          # shadow => 0
        Cp = Cp_max * cos_theta ** 2
        # Non-dim force per panel: dF/q_inf = -Cp * A * n_hat
        forces = -(Cp * areas)[:, None] * normals
        return forces, centroids, Cp, areas

    def evaluate(self, Ma, alpha_deg=0.0,
                 atm_conditions: Optional[Dict] = None) -> Dict[str, float]:
        alpha = np.radians(alpha_deg)
        flow_dir = np.array([np.cos(alpha), -np.sin(alpha), 0.0])
        lift_dir = np.array([np.sin(alpha),  np.cos(alpha), 0.0])

        Cp_max = _cp_max_newtonian(Ma, self.gamma)

        F_u, C_u, _, _ = self._evaluate_surface(
            self._upper_tris, +1, flow_dir, Cp_max)
        F_l, C_l, _, _ = self._evaluate_surface(
            self._lower_tris, -1, flow_dir, Cp_max)

        forces = np.concatenate([F_u, F_l], axis=0)
        centroids = np.concatenate([C_u, C_l], axis=0)

        F_total = forces.sum(axis=0)
        CD = float(np.dot(F_total, flow_dir)) / self.ref_area
        CL = float(np.dot(F_total, lift_dir)) / self.ref_area

        arms = centroids - self.moment_ref
        moments = np.cross(arms, forces).sum(axis=0)
        Cmz = float(moments[2]) / (self.ref_area * self.ref_length)

        L_D = CL / CD if abs(CD) > 1e-12 else 0.0

        if abs(F_total[1]) > 1e-12:
            Xcp = (moments[2] / F_total[1]) / self.ref_length
        else:
            Xcp = 0.0

        return {
            "Ma":  float(Ma),
            "alpha_deg": float(alpha_deg),
            "CL":  CL,
            "CD":  CD,
            "L_D": L_D,
            "Cmz": Cmz,
            "Xcp": Xcp,
            "Cp_max": float(Cp_max),
        }

    # ------------------------------------------------------------------
    def evaluate_paper_trajectory(self, progress_callback=None):
        results = []
        for i, row in enumerate(PAPER_TRAJECTORY):
            out = self.evaluate(row["Ma"], row["alpha"])
            out["H_km"] = row["H_km"]
            results.append(out)
            if progress_callback is not None:
                try:
                    progress_callback(i + 1)
                except Exception:
                    pass
        return results

    # ------------------------------------------------------------------
    def compare_with_paper(self):
        """Return list of dicts comparing computed to PAPER_REFERENCE_AERO."""
        rows = []
        for r in self.evaluate_paper_trajectory():
            ref = PAPER_REFERENCE_AERO.get(int(r["Ma"]), {})
            rows.append({
                "Ma":  r["Ma"],
                "CL":  (r["CL"],  ref.get("CL")),
                "CD":  (r["CD"],  ref.get("CD")),
                "L_D": (r["L_D"], ref.get("L_D")),
                "Cmz": (r["Cmz"], ref.get("Cmz")),
                "Xcp": (r["Xcp"], ref.get("Xcp")),
            })
        return rows
