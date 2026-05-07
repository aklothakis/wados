"""Geometry export adapters (GVWD §5.6).

Phase 6 supports:
  - STL (binary, pure-python writer)
  - STEP (via cadquery / OpenCascade)
  - IGES (via cadquery)

STEP and IGES require ``cadquery`` (already a dependency of the broader
waverider hub). When unavailable, the corresponding ``write_*`` functions
raise :class:`CadqueryUnavailableError` with a clear message.
"""

from .stl import write_stl
from .step import write_step, CadqueryUnavailableError
from .iges import write_iges

__all__ = [
    "write_stl", "write_step", "write_iges",
    "CadqueryUnavailableError",
]
