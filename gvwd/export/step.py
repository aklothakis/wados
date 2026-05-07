"""STEP export via cadquery / OpenCascade.

Builds a triangulated surface model: each Mesh triangle becomes a Face;
all faces are sewn into a Shell; the Shell is wrapped in a Solid where
possible (closed manifold) or exported as a multi-face Compound when
not.

This is sufficient for downstream CAD viewing (FreeCAD, Solidworks,
Fusion 360) and for handoff to CFD pre-processors that accept STEP. It
does NOT produce the smooth NURBS B-rep that a parametric modeller
would generate; the output is a "tessellated solid" with each triangle
as a planar face.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from gvwd.geometry.mesh import Mesh


class CadqueryUnavailableError(ImportError):
    """Raised when STEP / IGES export is requested but cadquery is not
    importable in the current environment."""


def _import_cadquery():
    try:
        import cadquery as cq
    except ImportError as e:
        raise CadqueryUnavailableError(
            "cadquery is required for STEP / IGES export; "
            "install with `pip install cadquery` or use `write_stl`."
        ) from e
    return cq


def write_step(mesh: Mesh, path: str | Path, *,
                scale: float = 1000.0,
                close_to_solid: bool = True) -> Path:
    """Write a GVWD Mesh as STEP via cadquery.

    Parameters
    ----------
    mesh             : closed surface mesh
    path             : output file path
    scale            : multiplier applied to vertex coordinates. Default
                       1000 converts meters to millimeters (CAD-convention).
    close_to_solid   : if True, attempt to wrap the Shell in a Solid
                       before exporting. Falls back to Shell-only export
                       on failure (e.g. mesh has small geometric defects).

    Returns
    -------
    Path written.
    """
    cq = _import_cadquery()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    verts = mesh.vertices * float(scale)
    faces_idx = mesh.faces

    cq_faces = []
    for tri in faces_idx:
        a, b, c = (cq.Vector(*verts[tri[i]]) for i in range(3))
        # Build a polygonal wire from the three vertices, close it,
        # then make a planar face.
        edges = [
            cq.Edge.makeLine(a, b),
            cq.Edge.makeLine(b, c),
            cq.Edge.makeLine(c, a),
        ]
        wire = cq.Wire.assembleEdges(edges)
        face = cq.Face.makeFromWires(wire)
        cq_faces.append(face)

    shell = cq.Shell.makeShell(cq_faces)
    shape = shell
    if close_to_solid:
        try:
            shape = cq.Solid.makeSolid(shell)
        except Exception:
            # Fallback: ship the Shell as a Compound
            shape = cq.Compound.makeCompound(cq_faces)

    cq.exporters.export(shape, str(path), exportType=cq.exporters.ExportTypes.STEP)
    return path
