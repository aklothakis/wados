"""IGES export.

cadquery >= 2.6 no longer exposes IGES via its high-level exporters
(``ExportTypes`` only includes AMF, BREP, DXF, STEP, STL, SVG, 3MF, TJS,
VRML, VTP). Older cadquery 2.x and the underlying OCCT do support IGES
through ``OCP.IGESControl_Writer``, but driving it bypasses cadquery's
public API and is fragile across cadquery / OCCT versions.

For Phase 6 the recommended interchange formats are STEP (preferred for
modern CAD round-trip) and STL (preferred for tessellated handoff to
mesh-based downstream tools). IGES is provided for legacy compatibility:
this module attempts the OCP path first and raises a clear message if
unavailable, so callers can fall back to STEP.
"""

from __future__ import annotations

from pathlib import Path

from gvwd.geometry.mesh import Mesh
from .step import _import_cadquery, CadqueryUnavailableError


class IGESUnavailableError(RuntimeError):
    """Raised when IGES export is unavailable in the current
    cadquery / OCP installation."""


def write_iges(mesh: Mesh, path: str | Path, *,
                scale: float = 1000.0) -> Path:
    """Write a GVWD Mesh as IGES.

    Tries the OCP IGES writer directly. If that fails, raises
    :class:`IGESUnavailableError` with a hint to use STEP instead.
    """
    cq = _import_cadquery()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Build the same shell construction as STEP
    verts = mesh.vertices * float(scale)
    cq_faces = []
    for tri in mesh.faces:
        a, b, c = (cq.Vector(*verts[tri[i]]) for i in range(3))
        edges = [
            cq.Edge.makeLine(a, b),
            cq.Edge.makeLine(b, c),
            cq.Edge.makeLine(c, a),
        ]
        wire = cq.Wire.assembleEdges(edges)
        cq_faces.append(cq.Face.makeFromWires(wire))
    shell = cq.Shell.makeShell(cq_faces)

    try:
        from OCP.IGESControl import IGESControl_Writer
    except ImportError as e:
        raise IGESUnavailableError(
            "IGES export requires OCP.IGESControl, not available in this "
            "cadquery install. Use write_step() instead — STEP is the "
            "modern CAD interchange standard."
        ) from e

    writer = IGESControl_Writer()
    writer.AddShape(shell.wrapped)
    writer.ComputeModel()
    ok = writer.Write(str(path))
    if not ok:
        raise IGESUnavailableError(f"IGESControl_Writer failed for {path}")
    return path
