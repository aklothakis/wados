"""Triangulated surface-mesh primitives (GVWD).

A geometry generator produces a closed manifold ``Mesh`` consisting of
vertex coordinates and triangle indices, optionally with per-triangle
surface labels (e.g. ``"upper"``, ``"lower"``, ``"base"``, ``"fin_1"``).
The Mesh is consumed by the panel-method aero (Phase 4), the heating
solver (Phase 4), and the export adapters (Phase 6).

Frame: x downstream, y spanwise, z vertical (matches PSWR-1 convention).
Outward normal direction is determined by the triangle's vertex ordering
(right-hand rule). The ``flip_normals`` argument lets a generator request
that all triangles in a surface group be reversed if the natural ordering
points inward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Mesh:
    """Closed triangulated surface mesh.

    Attributes
    ----------
    vertices : (N, 3) ndarray of vertex positions [m]
    faces    : (M, 3) ndarray of triangle vertex indices (int)
    labels   : (M,) ndarray of string labels per face (e.g. ``"upper"``,
               ``"lower"``, ``"base"``, ``"fin_root"``, ...). Used by the
               panel method to group surfaces and by the export adapter
               to write surface groups.
    """

    vertices: np.ndarray
    faces: np.ndarray
    labels: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=float).reshape(-1, 3)
        self.faces = np.asarray(self.faces, dtype=int).reshape(-1, 3)
        if self.labels is None:
            self.labels = np.array(["unknown"] * len(self.faces), dtype=object)
        else:
            self.labels = np.asarray(self.labels, dtype=object)
            if len(self.labels) != len(self.faces):
                raise ValueError(
                    f"labels length {len(self.labels)} != faces length "
                    f"{len(self.faces)}"
                )

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])

    def face_normals(self) -> np.ndarray:
        """Outward unit normals per face (right-hand rule)."""
        v = self.vertices[self.faces]   # (M, 3, 3)
        e1 = v[:, 1] - v[:, 0]
        e2 = v[:, 2] - v[:, 0]
        n = np.cross(e1, e2)
        norms = np.linalg.norm(n, axis=1, keepdims=True)
        return n / np.where(norms > 1e-30, norms, 1.0)

    def face_areas(self) -> np.ndarray:
        """Triangle areas [m^2]."""
        v = self.vertices[self.faces]
        e1 = v[:, 1] - v[:, 0]
        e2 = v[:, 2] - v[:, 0]
        return 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)

    def face_centroids(self) -> np.ndarray:
        """Triangle centroids."""
        return self.vertices[self.faces].mean(axis=1)

    def total_area(self) -> float:
        return float(self.face_areas().sum())

    def label_mask(self, name: str) -> np.ndarray:
        return self.labels == name

    def with_labels(self, faces: np.ndarray, label: str) -> "Mesh":
        """Return a new Mesh with ``label`` applied to the given faces."""
        new_labels = self.labels.copy()
        new_labels[faces] = label
        return Mesh(self.vertices, self.faces, new_labels, dict(self.metadata))


def mesh_volume_signed(mesh: Mesh) -> float:
    """Signed volume enclosed by a closed triangulated surface (divergence
    theorem). Returns the absolute value as a positive volume; the sign
    reflects whether outward normals are correctly oriented.

    For each triangle (a, b, c), volume contribution = (a . (b x c)) / 6.
    Sum across all triangles gives the enclosed volume. If the result is
    negative, the normals point inward; the absolute value is still the
    enclosed volume.

    For correctly-oriented outward normals, the sum is positive.
    """
    v = mesh.vertices[mesh.faces]   # (M, 3, 3)
    a, b, c = v[:, 0], v[:, 1], v[:, 2]
    return float(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0)
