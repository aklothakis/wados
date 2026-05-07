"""Variable-wedge waverider geometry (PSWR-1 §5.1).

Frame:
    x = streamwise, +x downstream from apex (origin).
    y = spanwise.
    z = vertical (lift direction, +z up).

Sign / sweep convention (intentional deviation from PSWR-1 prompt §5.1 sign
of ``x_LE`` and ``z_LE``): ``Lambda`` is the conventional aerospace
leading-edge sweep angle measured from the +y (spanwise) axis. With this
choice ``Lambda`` in [55, 80] deg gives the highly-swept LE typical of
waveriders, and the apex is forward of the base plane (``x_LE >= 0``,
``x_b > x_LE``). The shock sits below the symmetry plane (``z_LE <= 0``).

Variable-wedge approximation: each spanwise station is treated as locally 2D
with shock angle beta(y) and wedge half-angle theta(y) from theta-beta-M.
Validity check: |dbeta/dy| L_chord / b << 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.interpolate import CubicSpline

from ..thermo.oblique_shock import (
    mach_angle,
    detachment_beta,
    theta_from_beta_M_array,
)


# ----------------------------------------------------------------------
#  beta(y) clamped cubic spline
# ----------------------------------------------------------------------

@dataclass
class BetaSpline:
    """Clamped cubic spline beta(eta) on three knots eta in {0, 0.5, 1}.

    eta = y / y_tip in [0, 1]. ``beta_knots`` are the shock angles (rad)
    at eta = 0 (centreline), 0.5, 1 (wingtip).

    Boundary condition: zero second derivative at both endpoints (natural).
    """

    beta0: float
    beta1: float
    beta2: float
    _spline: Optional[CubicSpline] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        eta = np.array([0.0, 0.5, 1.0])
        beta = np.array([self.beta0, self.beta1, self.beta2])
        # Natural BC = zero curvature at endpoints (equivalent to clamped d2/d eta2 = 0).
        self._spline = CubicSpline(eta, beta, bc_type="natural")

    def __call__(self, eta: np.ndarray) -> np.ndarray:
        return self._spline(np.asarray(eta))

    def smoothness_residual(self, n: int = 50) -> float:
        """Mean absolute second derivative — used as a soft regularizer."""
        eta = np.linspace(0.0, 1.0, n)
        d2 = self._spline(eta, 2)
        return float(np.mean(np.abs(d2)))


# ----------------------------------------------------------------------
#  Variable-wedge waverider
# ----------------------------------------------------------------------

@dataclass
class VariableWedgeWaverider:
    """Variable-wedge waverider parameterised by (beta_knots, Lambda).

    Inputs (SI / radians):
        M_inf         : freestream Mach number
        beta_knots    : (beta0, beta1, beta2) shock angles at eta=0, 0.5, 1 [rad]
        Lambda        : LE sweep from spanwise axis [rad]; 0 < Lambda < pi/2
        body_length   : x_b, base-plane streamwise station [m]
        flat_fraction : X1 in [0, 1) — fraction of half-span occupied by a
                        centerline flat-nose strip with no sweep (analogous
                        to the OC / cone-derived waverider's X1). Inside
                        |y| <= y_flat the LE sits at x=0, z=0; outside it
                        sweeps back at angle Lambda. Default 0 = sharp apex.
        n_span        : number of spanwise stations stored
        n_chord       : number of chordwise samples per streamline
        gamma         : specific heat ratio
        T_inf, p_inf  : freestream T [K] / p [Pa]

    On construction, the geometry is built and stored as:
        leading_edge          : (n_span, 3) ndarray, full-span y in [-y_tip, +y_tip]
        upper_surface         : (n_span, 2, 3)  — two-point line at each y (LE, base)
        lower_surface_streams : list of (n_chord, 3) ndarrays — one streamline per y
        beta_y, theta_y       : (n_span,) shock and wedge angles per station
        y_tip, y_flat         : tip half-span and flat-region half-width [m]
    """

    M_inf: float
    beta_knots: tuple
    Lambda: float
    body_length: float = 10.0
    flat_fraction: float = 0.0       # X1: fraction of half-span that is flat-nose
    n_span: int = 41                # symmetric about y=0
    n_chord: int = 30
    gamma: float = 1.4
    T_inf: float = 226.65            # K, US Std 30 km
    p_inf: float = 1197.03           # Pa, US Std 30 km

    # populated by build()
    beta_spline: Optional[BetaSpline] = field(default=None, init=False, repr=False)
    y_tip: float = field(default=0.0, init=False)
    y_flat: float = field(default=0.0, init=False)
    y_grid: np.ndarray = field(default=None, init=False, repr=False)
    eta_grid: np.ndarray = field(default=None, init=False, repr=False)
    beta_y: np.ndarray = field(default=None, init=False, repr=False)
    theta_y: np.ndarray = field(default=None, init=False, repr=False)
    leading_edge: np.ndarray = field(default=None, init=False, repr=False)
    upper_surface: np.ndarray = field(default=None, init=False, repr=False)
    lower_surface_streams: list = field(default_factory=list, init=False, repr=False)
    warnings: list = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self._validate_inputs()
        self._build()

    # ------------------------------------------------------------------
    #  Validation
    # ------------------------------------------------------------------

    def _validate_inputs(self) -> None:
        if self.M_inf <= 1.0:
            raise ValueError(f"Supersonic only: M_inf={self.M_inf}")
        if not (0.0 < self.Lambda < math.pi / 2):
            raise ValueError(
                f"Lambda must be in (0, pi/2); got {math.degrees(self.Lambda):.2f} deg")
        if self.body_length <= 0:
            raise ValueError(f"body_length must be > 0; got {self.body_length}")

        b0, b1, b2 = self.beta_knots
        mu = mach_angle(self.M_inf)
        beta_det = detachment_beta(self.M_inf, self.gamma)
        for i, b in enumerate((b0, b1, b2)):
            if b <= mu:
                raise ValueError(
                    f"beta_knot[{i}]={math.degrees(b):.3f}deg below Mach angle "
                    f"mu={math.degrees(mu):.3f}deg at M={self.M_inf}")
            if b >= beta_det:
                raise ValueError(
                    f"beta_knot[{i}]={math.degrees(b):.3f}deg at/above detachment "
                    f"beta_det={math.degrees(beta_det):.3f}deg at M={self.M_inf}")

    # ------------------------------------------------------------------
    #  Build
    # ------------------------------------------------------------------

    def _build(self) -> None:
        b0, b1, b2 = self.beta_knots
        self.beta_spline = BetaSpline(b0, b1, b2)

        tan_L = math.tan(self.Lambda)
        sin_L = math.sin(self.Lambda)

        # ---- Flat-nose / sweep geometry ---------------------------------
        # Tip half-span solved from "the swept portion of the LE must reach
        # x = body_length at the wingtip":
        #     y_tip - y_flat = body_length / tan(Lambda)
        #     y_flat         = X1 * y_tip
        # =>  y_tip = body_length / [(1 - X1) tan(Lambda)]
        X1 = max(0.0, min(0.99, float(self.flat_fraction)))
        if X1 >= 0.99:
            # Degenerate: pure flat (no swept portion). Fall back to caret
            # tip-span and emit a warning.
            self.y_tip = self.body_length / tan_L
            X1 = 0.0
            self.warnings.append("flat_fraction clamped to 0 (>=0.99 degenerate)")
        else:
            self.y_tip = self.body_length / ((1.0 - X1) * tan_L)
        self.y_flat = X1 * self.y_tip

        # Symmetric spanwise grid (full body, y in [-y_tip, +y_tip])
        n = self.n_span if self.n_span % 2 == 1 else self.n_span + 1
        y = np.linspace(-self.y_tip, self.y_tip, n)
        # eta-on-spline: 0 inside the flat region, then linear from 0 -> 1
        # across the swept portion.
        abs_y = np.abs(y)
        in_flat = abs_y <= self.y_flat
        denom = max(self.y_tip - self.y_flat, 1e-12)
        eta = np.where(in_flat, 0.0, (abs_y - self.y_flat) / denom)
        eta = np.clip(eta, 0.0, 1.0)
        self.y_grid = y
        self.eta_grid = eta

        beta_y = self.beta_spline(eta)
        theta_y = theta_from_beta_M_array(beta_y, self.M_inf, self.gamma)
        self.beta_y = beta_y
        self.theta_y = theta_y

        # ---- Leading edge ------------------------------------------------
        # In flat region: LE at (0, y, 0).
        # In swept region: x_LE = (|y| - y_flat) tan Lambda,
        #                  z_LE = -(|y| - y_flat) tan beta(y) / sin Lambda.
        x_LE = np.where(in_flat, 0.0, (abs_y - self.y_flat) * tan_L)
        z_LE = np.where(in_flat, 0.0,
                         -(abs_y - self.y_flat) * np.tan(beta_y) / sin_L)
        self.leading_edge = np.column_stack([x_LE, y, z_LE])

        # ---- Upper surface (horizontal plane at z_LE per station) --------
        upper = np.zeros((n, 2, 3))
        upper[:, 0, :] = self.leading_edge                      # LE
        upper[:, 1, 0] = self.body_length                       # x_b
        upper[:, 1, 1] = y
        upper[:, 1, 2] = z_LE                                   # same z
        self.upper_surface = upper

        # ---- Lower surface streamlines ----------------------------------
        # Each streamline goes from (x_LE, y, z_LE) to (x_b, y, z_TE) with
        # z_TE = z_LE - (x_b - x_LE) tan(theta(y)). Parameterise by streamwise
        # position so the streamline truly reaches x=x_b.
        streams: list = []
        for j in range(n):
            x0, yj, z0 = self.leading_edge[j]
            chord = self.body_length - x0
            if chord <= 1e-9:
                streams.append(np.array([[x0, yj, z0],
                                          [x0, yj, z0]]))
                continue
            t_x = np.linspace(0.0, chord, self.n_chord)
            tan_th = math.tan(theta_y[j])
            xs = x0 + t_x
            zs = z0 - t_x * tan_th
            ys = np.full_like(t_x, yj)
            streams.append(np.column_stack([xs, ys, zs]))
        self.lower_surface_streams = streams

        # ---- Variable-wedge consistency warning ------------------------
        # |d beta/d y| * L_chord / b << 1, evaluated on the swept portion
        # (the spline is constant in the flat region by construction).
        d_beta_d_eta = self.beta_spline._spline(self.eta_grid, 1)
        max_dbeta_dy = float(np.max(np.abs(d_beta_d_eta))) / max(denom, 1e-12)
        if max_dbeta_dy * self.body_length > 0.5:
            self.warnings.append(
                f"Variable-wedge consistency: max|dbeta/dy|*L = "
                f"{max_dbeta_dy * self.body_length:.3f} (should be << 1)")

    # ------------------------------------------------------------------
    #  Convenience properties
    # ------------------------------------------------------------------

    @property
    def beta_knots_deg(self) -> tuple:
        return tuple(math.degrees(b) for b in self.beta_knots)

    @property
    def Lambda_deg(self) -> float:
        return math.degrees(self.Lambda)

    @property
    def theta_y_deg(self) -> np.ndarray:
        return np.degrees(self.theta_y)

    @property
    def beta_y_deg(self) -> np.ndarray:
        return np.degrees(self.beta_y)

    # ------------------------------------------------------------------
    #  Helper: spanwise queries that respect the flat-nose mapping
    # ------------------------------------------------------------------

    def eta_for_y(self, y) -> np.ndarray:
        """Spline parameter eta in [0, 1] for a given (array of) y-positions,
        respecting the flat-nose mapping (eta=0 inside |y|<=y_flat)."""
        abs_y = np.abs(np.asarray(y, dtype=float))
        denom = max(self.y_tip - self.y_flat, 1e-12)
        eta = np.where(abs_y <= self.y_flat, 0.0,
                        (abs_y - self.y_flat) / denom)
        return np.clip(eta, 0.0, 1.0)

    def beta_at_y(self, y) -> np.ndarray:
        """Local oblique-shock angle beta(y) [rad] at arbitrary y."""
        return self.beta_spline(self.eta_for_y(y))

    def x_LE_at_y(self, y) -> np.ndarray:
        """Streamwise leading-edge position [m] at arbitrary y."""
        abs_y = np.abs(np.asarray(y, dtype=float))
        tan_L = math.tan(self.Lambda)
        return np.where(abs_y <= self.y_flat,
                         0.0,
                         (abs_y - self.y_flat) * tan_L)

    def z_LE_at_y(self, y) -> np.ndarray:
        """LE vertical position [m] at arbitrary y (LE rides on the shock)."""
        abs_y = np.abs(np.asarray(y, dtype=float))
        sin_L = math.sin(self.Lambda)
        beta = self.beta_at_y(abs_y)
        return np.where(abs_y <= self.y_flat,
                         0.0,
                         -(abs_y - self.y_flat) * np.tan(beta) / sin_L)


# ----------------------------------------------------------------------
#  Frame conversion to GUI canvas convention
# ----------------------------------------------------------------------

def to_gui_frame(points_xyz: np.ndarray) -> np.ndarray:
    """PSWR-1 (x_stream, y_span, z_up) -> GUI (x_stream, y_vertical, z_span).

    Permutation: (x, y, z)_pswr -> (x, z, y)_gui.
    """
    pts = np.asarray(points_xyz)
    out = np.empty_like(pts)
    out[..., 0] = pts[..., 0]
    out[..., 1] = pts[..., 2]   # z_pswr -> y_gui (vertical)
    out[..., 2] = pts[..., 1]   # y_pswr -> z_gui (span)
    return out
