"""Binary STL writer for GVWD meshes.

Pure-Python implementation (no numpy-stl dependency). Format reference:
https://en.wikipedia.org/wiki/STL_(file_format)#Binary_STL

Header (80 bytes) + triangle count (uint32) + per-triangle:
  - normal (3 floats)
  - vertex 1 (3 floats)
  - vertex 2 (3 floats)
  - vertex 3 (3 floats)
  - attribute byte count (uint16, =0)

Total bytes per triangle: 50.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Optional

import numpy as np

from gvwd.geometry.mesh import Mesh


def write_stl(mesh: Mesh, path: str | Path, *,
                header: Optional[str] = None,
                scale: float = 1.0) -> Path:
    """Write a GVWD :class:`Mesh` as binary STL.

    Parameters
    ----------
    mesh   : closed surface mesh
    path   : output file path
    header : optional 80-character header string (truncated if longer).
             If None, encodes a brief gvwd identifier.
    scale  : multiplicative scale applied to all vertex coordinates
             (default 1.0; use 1000.0 to write meters as millimeters).

    Returns
    -------
    Path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if header is None:
        kind = mesh.metadata.get("kind", "gvwd_mesh")
        header = f"gvwd-export {kind} {mesh.n_faces} faces"
    header_bytes = header.encode("ascii", errors="replace")[:80]
    header_bytes = header_bytes.ljust(80, b"\x00")

    vertices = mesh.vertices * float(scale)
    normals = mesh.face_normals()

    with open(path, "wb") as f:
        f.write(header_bytes)
        f.write(struct.pack("<I", mesh.n_faces))
        for k in range(mesh.n_faces):
            n = normals[k]
            tri = vertices[mesh.faces[k]]
            f.write(struct.pack("<fff", float(n[0]), float(n[1]), float(n[2])))
            for v in tri:
                f.write(struct.pack("<fff",
                                     float(v[0]), float(v[1]), float(v[2])))
            f.write(struct.pack("<H", 0))   # attribute byte count
    return path


def read_stl(path: str | Path) -> tuple:
    """Read a binary STL into ``(vertices, faces, normals)``. Used in
    round-trip tests and as a sanity-check utility."""
    path = Path(path)
    with open(path, "rb") as f:
        f.read(80)   # header
        (n_tri,) = struct.unpack("<I", f.read(4))
        verts: list = []
        normals: list = []
        for _ in range(n_tri):
            n = struct.unpack("<fff", f.read(12))
            v1 = struct.unpack("<fff", f.read(12))
            v2 = struct.unpack("<fff", f.read(12))
            v3 = struct.unpack("<fff", f.read(12))
            f.read(2)
            verts.extend([v1, v2, v3])
            normals.append(n)
    vertices = np.asarray(verts, dtype=float)
    faces = np.arange(n_tri * 3, dtype=int).reshape(n_tri, 3)
    normals_arr = np.asarray(normals, dtype=float)
    return vertices, faces, normals_arr
