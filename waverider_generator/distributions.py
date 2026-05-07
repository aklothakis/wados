"""Spanwise distribution functions for variable-Mach waverider design.

Provides B-spline parametrised distributions of quantities (Mach number,
power-law exponent, shock angle) across the half-span, with enforced
symmetry about z = 0.
"""

import numpy as np
from scipy.interpolate import CubicSpline, interp1d


class SpanwiseDistribution:
    """B-spline parametrised distribution of a quantity across the half-span.

    The distribution is defined by control-point values at uniformly spaced
    stations from z = 0 to z = half_span.  A cubic spline with
    dF/dz = 0 at z = 0 (symmetry condition) is fitted through the points.

    Parameters
    ----------
    control_points : list[float]
        Values at uniformly spaced stations from z=0 to z=half_span.
    half_span : float
        W/2, half the waverider width.
    name : str
        Descriptive name (e.g. "Mach", "exponent", "shock_angle").
    """

    def __init__(self, control_points: list, half_span: float, name: str = ""):
        self.control_points = np.asarray(control_points, dtype=float)
        self.half_span = float(half_span)
        self.name = name
        self.n_cp = len(self.control_points)
        if self.n_cp < 2:
            raise ValueError("Need at least 2 control points.")
        self._build_spline()

    def _build_spline(self):
        z_knots = np.linspace(0, self.half_span, self.n_cp)
        if self.n_cp == 2:
            self.spline = interp1d(z_knots, self.control_points, kind='linear',
                                   fill_value='extrapolate')
            self._deriv_spline = None
        else:
            # CubicSpline with clamped left BC (dF/dz=0 at z=0) for symmetry
            self.spline = CubicSpline(z_knots, self.control_points,
                                      bc_type=((1, 0.0), 'not-a-knot'))
            self._deriv_spline = self.spline.derivative()

    def __call__(self, z):
        """Evaluate distribution at spanwise position(s) z.  Handles symmetry."""
        z = np.asarray(z, dtype=float)
        scalar = z.ndim == 0
        z = np.atleast_1d(z)
        z_abs = np.clip(np.abs(z), 0, self.half_span)
        result = self.spline(z_abs)
        return float(result[0]) if scalar else result

    def derivative(self, z):
        """Evaluate dF/dz at position(s) z.  Derivative is antisymmetric."""
        z = np.asarray(z, dtype=float)
        scalar = z.ndim == 0
        z = np.atleast_1d(z)
        z_abs = np.clip(np.abs(z), 0, self.half_span)
        sign = np.sign(z)
        if self._deriv_spline is not None:
            result = sign * self._deriv_spline(z_abs)
        else:
            # Linear spline: constant derivative
            if self.n_cp == 2:
                slope = (self.control_points[-1] - self.control_points[0]) / self.half_span
                result = sign * np.full_like(z_abs, slope)
            else:
                result = sign * np.zeros_like(z_abs)
        return float(result[0]) if scalar else result

    def to_vector(self) -> np.ndarray:
        """Return control points as a flat array (for optimiser)."""
        return self.control_points.copy()

    @classmethod
    def from_vector(cls, vector, half_span: float, name: str = ""):
        return cls(control_points=np.asarray(vector).tolist(),
                   half_span=half_span, name=name)

    @classmethod
    def constant(cls, value: float, half_span: float, n_cp: int = 4, name: str = ""):
        return cls(control_points=[value] * n_cp, half_span=half_span, name=name)

    @classmethod
    def linear(cls, center_value: float, tip_value: float, half_span: float,
               n_cp: int = 5, name: str = ""):
        cp = np.linspace(center_value, tip_value, n_cp).tolist()
        return cls(control_points=cp, half_span=half_span, name=name)

    @classmethod
    def quadratic_liu(cls, Ma_center: float, Ma_tip: float, half_span: float,
                      n_cp: int = 6, name: str = "Mach"):
        """Quadratic distribution matching Liu et al. (2019):
        Ma(z) = m*z^2 + Ma_center  where m = (Ma_tip - Ma_center) / half_span^2.
        """
        z_vals = np.linspace(0, half_span, n_cp)
        m = (Ma_tip - Ma_center) / half_span**2
        values = (m * z_vals**2 + Ma_center).tolist()
        return cls(control_points=values, half_span=half_span, name=name)

    def __repr__(self):
        return (f"SpanwiseDistribution(name='{self.name}', n_cp={self.n_cp}, "
                f"range=[{self.control_points.min():.3f}, {self.control_points.max():.3f}])")
