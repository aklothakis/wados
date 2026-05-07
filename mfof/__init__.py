"""Multi-Flowfield Osculating Framework (MFOF).

Phase 2 architectural refactor of the Liu 2019 osculating-cone waverider into
a plug-in framework. Each spanwise osculating plane carries its own
:class:`BasicFlowfield` instance, supplied by a factory at sweep time. Phase 2
ships only :class:`ConeFlowfield`; future phases will add ``PowerLawFlowfield``,
``WedgeFlowfield``, etc., and eventually allow different flowfield types to
coexist in a single waverider.

When the all-cone factory is used, ``MFOF`` reproduces ``liu2019`` numerically
to within ``1e-6`` -- see :func:`mfof.validate.run_equivalence_test`.
"""

from .basic_flowfield import BasicFlowfield, StreamlineResult
from .cone_flowfield import ConeFlowfield
from .wedge_flowfield import WedgeFlowfield
from .power_law_flowfield import PowerLawFlowfield

# Lazy imports: osculating, geometry, aero pull in numpy / scipy / liu2019.
# Re-exported here for the canonical public surface.
from .osculating import (
    OsculatingPlaneData,
    OsculatingPlaneSet,
    build_all_osculating_planes,
)
from .geometry import MFOFWaverider, build_mfof_waverider
from .aero import MFOFAeroEvaluator

from .config import (
    DEFAULT_PARAMS,
    PAPER_PARAMS,
    PAPER_TRAJECTORY,
    PAPER_REFERENCE_GEOMETRY,
    PAPER_REFERENCE_AERO,
    TOLERANCES,
    MOMENT_REF,
    REF_AREA_M2,
    REF_LENGTH_M,
)
from .distributions import (
    Ma_distribution,
    shock_curve,
    shock_curve_coefficient,
    upper_surface_coefficients,
    upper_surface_trailing_edge,
)

__all__ = [
    # Flowfield interface
    "BasicFlowfield",
    "StreamlineResult",
    "ConeFlowfield",
    "WedgeFlowfield",
    "PowerLawFlowfield",
    # Osculating sweep
    "OsculatingPlaneData",
    "OsculatingPlaneSet",
    "build_all_osculating_planes",
    # 3D geometry + aero
    "MFOFWaverider",
    "build_mfof_waverider",
    "MFOFAeroEvaluator",
    # Config
    "DEFAULT_PARAMS",
    "PAPER_PARAMS",
    "PAPER_TRAJECTORY",
    "PAPER_REFERENCE_GEOMETRY",
    "PAPER_REFERENCE_AERO",
    "TOLERANCES",
    "MOMENT_REF",
    "REF_AREA_M2",
    "REF_LENGTH_M",
    # Distributions
    "Ma_distribution",
    "shock_curve",
    "shock_curve_coefficient",
    "upper_surface_coefficients",
    "upper_surface_trailing_edge",
]
