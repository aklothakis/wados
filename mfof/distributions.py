"""Spanwise distribution helpers for the MFOF framework.

Phase 2 thin re-export of :mod:`liu2019.distributions`. Future phases may add
MFOF-specific distributions (B-spline ``Ma(z)``, variable shock-curve
exponents, etc.).
"""

from liu2019.distributions import (
    Ma_distribution,
    shock_curve,
    shock_curve_coefficient,
    upper_surface_coefficients,
    upper_surface_trailing_edge,
)

__all__ = [
    "Ma_distribution",
    "shock_curve",
    "shock_curve_coefficient",
    "upper_surface_coefficients",
    "upper_surface_trailing_edge",
]
