"""Liu et al. 2019 Variable-Mach Osculating Flowfield Waverider.

Reference:
    Liu, J., Liu, Z., Wen, X., Ding, F. (2019).
    "Novel osculating flowfield methodology for wide-speed range waverider
    vehicles across variable Mach number."
    Acta Astronautica 162, 160-167. DOI: 10.1016/j.actaastro.2019.05.056.
"""

from .config import (
    PAPER_PARAMS,
    PAPER_TRAJECTORY,
    PAPER_REFERENCE_GEOMETRY,
    PAPER_REFERENCE_AERO,
    TOLERANCES,
)

from .distributions import (
    Ma_distribution,
    upper_surface_trailing_edge,
    shock_curve,
    upper_surface_coefficients,
    shock_curve_coefficient,
)

from .shock import (
    DetachedShockError,
    theta_from_beta_Ma,
    beta_from_theta_Ma,
    beta_detachment,
    mach_angle,
    oblique_shock_ratios,
    taylor_maccoll_cone_angle,
)

from .osculating import (
    OsculatingPlaneData,
    curvature_radius_ICC,
    leading_edge_point,
    trailing_edge_of_compression,
    build_all_osculating_planes,
)

from .geometry import Liu2019Waverider, build_liu2019_waverider

__all__ = [
    "PAPER_PARAMS",
    "PAPER_TRAJECTORY",
    "PAPER_REFERENCE_GEOMETRY",
    "PAPER_REFERENCE_AERO",
    "TOLERANCES",
    "Ma_distribution",
    "upper_surface_trailing_edge",
    "shock_curve",
    "upper_surface_coefficients",
    "shock_curve_coefficient",
    "DetachedShockError",
    "theta_from_beta_Ma",
    "beta_from_theta_Ma",
    "beta_detachment",
    "mach_angle",
    "oblique_shock_ratios",
    "taylor_maccoll_cone_angle",
    "OsculatingPlaneData",
    "curvature_radius_ICC",
    "leading_edge_point",
    "trailing_edge_of_compression",
    "build_all_osculating_planes",
    "Liu2019Waverider",
    "build_liu2019_waverider",
]
