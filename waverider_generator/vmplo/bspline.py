"""Clamped cubic B-spline on [z_min, z_max] with optional symmetry BC.

Spec reference: VMPLO_implementation_prompt.md §3, §8, "Module Specifications"
for ``vmplo/bspline.py``.

The B-spline is a *representation* layer: the user supplies control
coefficients, the class evaluates the spline and its derivatives.
When ``symmetry=True`` (default) the left endpoint has dF/dz = 0, which
is required for smooth mirroring across the symmetry plane at z=0 for
Ma(z), n(z), ICC(z), and upper-surface y(z).

Backend: :class:`scipy.interpolate.BSpline`.  Degree 3 (cubic).
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline, splrep


class BSpline1D:
    """Clamped cubic B-spline on ``[z_min, z_max]``.

    Parameters
    ----------
    z_min, z_max : float
        Parameter domain.  For VMPLO, ``z_min = 0`` (symmetry plane) and
        ``z_max = W`` (half-span wingtip).
    n_internal_knots : int, optional
        Number of *internal* knots (default 4).  Together with the
        degree-(k+1) clamped endpoint knot multiplicities this gives
        ``n_internal_knots + 4`` control coefficients.
    symmetry : bool, optional
        If True, derivative is zero-clamped at ``z_min`` (default True).
        Implemented by constraining the first two control points to be
        equal whenever coefficients are set — the underlying spline is
        still the raw B-spline, so values are exact at the knots.
    degree : int, optional
        Polynomial degree (default 3, cubic).  Only 3 is tested.

    Notes
    -----
    The public coefficient vector has length ``n_internal_knots + 4``
    and is the *free* parameter set.  Internally we construct a clamped
    cubic B-spline with ``2*(degree+1) + n_internal_knots`` knots (the
    clamped ends have multiplicity ``degree+1``) and ``n_internal_knots
    + 4`` control points, matching the public size.
    """

    def __init__(self, z_min: float, z_max: float,
                 n_internal_knots: int = 4,
                 symmetry: bool = True,
                 degree: int = 3):
        if z_max <= z_min:
            raise ValueError("z_max must be greater than z_min.")
        if n_internal_knots < 1:
            raise ValueError("Need at least 1 internal knot.")
        if degree != 3:
            raise ValueError("Only cubic (degree=3) is supported.")

        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.degree = int(degree)
        self.n_internal_knots = int(n_internal_knots)
        self.symmetry = bool(symmetry)

        # Clamped knot vector: degree+1 repeats at each end, interior knots
        # uniform in (z_min, z_max).
        interior = np.linspace(self.z_min, self.z_max,
                               self.n_internal_knots + 2)[1:-1]
        self._knots = np.concatenate([
            np.full(self.degree + 1, self.z_min),
            interior,
            np.full(self.degree + 1, self.z_max),
        ])

        # Number of coefficients = n_knots - degree - 1
        self.n_coeffs = len(self._knots) - self.degree - 1

        # Default coefficients: zeros.  Caller must set via
        # ``from_coefficients`` or ``fit_values`` before evaluating.
        self._coeffs = np.zeros(self.n_coeffs, dtype=float)
        self._spline = BSpline(self._knots, self._coeffs, self.degree,
                               extrapolate=False)

    # ------------------------------------------------------------------ #
    #  Coefficient I/O                                                    #
    # ------------------------------------------------------------------ #

    def from_coefficients(self, coeffs):
        """Set the control-point vector (length = ``n_coeffs``).

        If ``symmetry=True``, the first two coefficients are forced
        equal (imposing dF/dz = 0 at ``z_min`` for a cubic clamped
        B-spline).
        """
        coeffs = np.asarray(coeffs, dtype=float).copy()
        if coeffs.shape != (self.n_coeffs,):
            raise ValueError(
                f"Expected {self.n_coeffs} coefficients, got {coeffs.shape}.")
        if self.symmetry and self.n_coeffs >= 2:
            coeffs[1] = coeffs[0]
        self._coeffs = coeffs
        self._spline = BSpline(self._knots, self._coeffs, self.degree,
                               extrapolate=False)
        return self

    def to_coefficients(self) -> np.ndarray:
        """Return the current coefficient vector (copy)."""
        return self._coeffs.copy()

    # ------------------------------------------------------------------ #
    #  Evaluation                                                         #
    # ------------------------------------------------------------------ #

    def __call__(self, z):
        """Evaluate the spline at scalar or array ``z``."""
        z = np.asarray(z, dtype=float)
        scalar = (z.ndim == 0)
        z_arr = np.atleast_1d(z)
        z_clip = np.clip(z_arr, self.z_min, self.z_max)
        out = self._spline(z_clip)
        # scipy returns NaN for values exactly outside the domain with
        # extrapolate=False; the clip above should already prevent this,
        # but guard against NaN from numerical edge cases.
        if np.any(np.isnan(out)):
            out = np.nan_to_num(out, nan=0.0)
        return float(out[0]) if scalar else out

    def derivative(self, z, order: int = 1):
        """Evaluate the ``order``-th derivative at ``z``."""
        z = np.asarray(z, dtype=float)
        scalar = (z.ndim == 0)
        z_arr = np.atleast_1d(z)
        z_clip = np.clip(z_arr, self.z_min, self.z_max)
        d_spline = self._spline.derivative(order)
        out = d_spline(z_clip)
        if np.any(np.isnan(out)):
            out = np.nan_to_num(out, nan=0.0)
        return float(out[0]) if scalar else out

    # ------------------------------------------------------------------ #
    #  Convenience constructors                                           #
    # ------------------------------------------------------------------ #

    def fit_values(self, z_pts, f_pts):
        """Fit the B-spline to ``(z_pts, f_pts)`` via least squares.

        The fitted coefficients are stored and the spline is updated in
        place.  Returns ``self`` for chaining.  Symmetry BC is
        post-enforced (the first two coefficients are equalised, which
        may perturb the fit slightly at z=z_min — this is intentional
        for VMPLO where symmetry is a hard requirement).
        """
        z_pts = np.asarray(z_pts, dtype=float)
        f_pts = np.asarray(f_pts, dtype=float)
        if z_pts.shape != f_pts.shape:
            raise ValueError("z_pts and f_pts must have matching shape.")

        # Use splrep on (z_pts, f_pts) with the clamped knot vector's
        # interior knots as prescribed.  If z_pts doesn't cover the
        # knot span this can fail — fall back to uniform resampling.
        interior = self._knots[self.degree + 1:-(self.degree + 1)]
        try:
            tck = splrep(z_pts, f_pts, k=self.degree, t=interior)
            coeffs = tck[1][:self.n_coeffs]
        except Exception:
            # Fallback: sample f at the Greville abscissae and use those
            # as coefficients (approximate, but robust).
            greville = np.convolve(self._knots, np.ones(self.degree) / self.degree,
                                    mode="valid")[:self.n_coeffs]
            from scipy.interpolate import interp1d
            fi = interp1d(z_pts, f_pts, bounds_error=False,
                          fill_value=(f_pts[0], f_pts[-1]))
            coeffs = fi(np.clip(greville, z_pts[0], z_pts[-1]))

        return self.from_coefficients(coeffs)

    # ------------------------------------------------------------------ #
    #  Factory helpers that match the old SpanwiseDistribution API        #
    # ------------------------------------------------------------------ #

    @classmethod
    def constant(cls, value: float, z_min: float, z_max: float,
                 n_internal_knots: int = 4) -> "BSpline1D":
        """Constant distribution at ``value`` over the whole span."""
        sp = cls(z_min, z_max, n_internal_knots=n_internal_knots,
                 symmetry=True)
        sp.from_coefficients(np.full(sp.n_coeffs, float(value)))
        return sp

    @classmethod
    def linear(cls, value_center: float, value_tip: float,
               z_min: float, z_max: float,
               n_internal_knots: int = 4) -> "BSpline1D":
        """Linear distribution from centre to tip.

        Uses a least-squares fit to a dense sampling of the linear
        reference profile so the B-spline approximates it as closely
        as the degree+knot count allow.
        """
        sp = cls(z_min, z_max, n_internal_knots=n_internal_knots,
                 symmetry=True)
        z = np.linspace(z_min, z_max, 50)
        f = np.linspace(value_center, value_tip, 50)
        sp.fit_values(z, f)
        return sp

    @classmethod
    def quadratic_liu(cls, value_center: float, value_tip: float,
                      z_min: float, z_max: float,
                      n_internal_knots: int = 4) -> "BSpline1D":
        """Quadratic Liu-style distribution: f(z) = m*z^2 + f_center,
        with m chosen so that f(z_max) = value_tip.  dF/dz = 0 at z=0
        is automatic.
        """
        sp = cls(z_min, z_max, n_internal_knots=n_internal_knots,
                 symmetry=True)
        half = z_max - z_min
        if half <= 0:
            raise ValueError("z_max must exceed z_min.")
        m = (value_tip - value_center) / half**2
        z = np.linspace(z_min, z_max, 50)
        f = m * (z - z_min)**2 + value_center
        sp.fit_values(z, f)
        return sp

    def __repr__(self):
        return (f"BSpline1D(z=[{self.z_min}, {self.z_max}], "
                f"n_internal_knots={self.n_internal_knots}, "
                f"symmetry={self.symmetry}, "
                f"n_coeffs={self.n_coeffs})")
