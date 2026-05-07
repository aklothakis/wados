"""VMPLO — Variable Mach Power-Law Osculating flowfield waverider package.

Implements the method specified in ``VMPLO_implementation_prompt.md``: each
osculating plane uses a power-law axisymmetric generating body with
spanwise-varying exponent ``n(z)`` and Mach number ``Ma(z)``, while the
shock angle ``beta`` is held constant across the span.  The constant-beta
choice guarantees a provably C2 shock surface (see ``smoothness.py``).

The public entry point for the GUI is :class:`geometry.VMPLOWaverider`.
It consumes an :class:`osculating.OsculatingAssembly` and exposes both the
spec-style API (``lower_surface``, ``upper_surface``, ``volume``, ...) and
the compatibility shim expected by the existing CAD pipeline
(``upper_surface_streams``, ``lower_surface_streams``, ``leading_edge``,
``to_CAD``).

Coordinate convention (matches the rest of ``waverider_generator``):
    x — streamwise (nose at x=0, base plane at x=L)
    y — vertical
    z — spanwise (symmetry plane at z=0, wingtip at z=W)
"""

from waverider_generator.vmplo.bspline import BSpline1D
from waverider_generator.vmplo.shock import (
    DetachedShockError,
    theta_from_beta_Ma,
    beta_from_theta_Ma,
    oblique_shock_ratios,
    beta_detachment,
    theta_max,
    mach_angle,
)
from waverider_generator.vmplo.powerlaw import (
    PowerLawBody,
    taylor_maccoll_cone_angle,
    solve_osculating_plane,
    CONE_TOL,
)
from waverider_generator.vmplo.osculating import OsculatingAssembly
from waverider_generator.vmplo.geometry import VMPLOWaverider
from waverider_generator.vmplo.design_space import (
    DEFAULT_PARAMS,
    BOUNDS,
    build_design_vector,
    unpack_design_vector,
    default_design_vector,
    check_feasibility,
)

__all__ = [
    "BSpline1D",
    "DetachedShockError",
    "theta_from_beta_Ma",
    "beta_from_theta_Ma",
    "oblique_shock_ratios",
    "beta_detachment",
    "theta_max",
    "mach_angle",
    "PowerLawBody",
    "taylor_maccoll_cone_angle",
    "solve_osculating_plane",
    "CONE_TOL",
    "OsculatingAssembly",
    "VMPLOWaverider",
    "DEFAULT_PARAMS",
    "BOUNDS",
    "build_design_vector",
    "unpack_design_vector",
    "default_design_vector",
    "check_feasibility",
]
