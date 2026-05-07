"""Default parameters for the MFOF framework.

Phase 2 simply re-exports the Liu 2019 paper parameters and tolerances so
``MFOF`` reproduces Liu's reference geometry by default. Future phases will
add MFOF-specific parameters here (e.g. per-region flowfield-type maps).
"""

from liu2019.config import (
    MOMENT_REF,
    PAPER_PARAMS,
    PAPER_REFERENCE_AERO,
    PAPER_REFERENCE_GEOMETRY,
    PAPER_TRAJECTORY,
    REF_AREA_M2,
    REF_LENGTH_M,
    TOLERANCES,
)

# MFOF-specific default parameter dict (mutable copy of PAPER_PARAMS).
DEFAULT_PARAMS = dict(PAPER_PARAMS)

__all__ = [
    "DEFAULT_PARAMS",
    "PAPER_PARAMS",
    "PAPER_TRAJECTORY",
    "PAPER_REFERENCE_GEOMETRY",
    "PAPER_REFERENCE_AERO",
    "TOLERANCES",
    "MOMENT_REF",
    "REF_AREA_M2",
    "REF_LENGTH_M",
]
