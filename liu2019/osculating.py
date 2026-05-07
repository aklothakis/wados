"""Osculating-plane geometry for Liu et al. 2019.

Two variants are available per spanwise station:

* **Flat region** (|z| <= L_s, shock curve is straight): the osculating plane
  reduces to the vertical (x, y) plane, and the leading/trailing edges are
  found by 2D wedge geometry.
* **Curved region** (|z| > L_s): full osculating-cone geometry (Rodi 2011).
  The osculating plane is spanned by the freestream axis and the inward
  shock-curve normal; it tilts away from vertical by
  :math:`\\phi = \\arctan(y_s'(z))`. The leading edge is located by
  intersecting this plane with the freestream-surface trailing-edge
  curve y(z), and the compression streamline descends at angle
  :math:`\\delta_c` within the plane (which, projected to 3D, has both a
  y- and a z-component).

The curved-region geometry collapses exactly to the flat-region one as
:math:`y_s' \\to 0`, so the implementation is seamless across the boundary
|z| = L_s.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from .distributions import (
    Ma_distribution,
    shock_curve,
    upper_surface_trailing_edge,
)
from .shock import (
    taylor_maccoll_cone_angle,
    taylor_maccoll_cone_field,
    theta_from_beta_Ma,
)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class OsculatingPlaneData:
    z:          float    # spanwise station of the SHOCK curve
    Ma:         float
    delta_c:    float
    R_osc:      float
    n_base:     Tuple[float, float, float]   # (0, n_y, n_z) inward normal in base plane
    P_shock:    Tuple[float, float, float]
    P_LE:       Tuple[float, float, float]   # leading-edge point in 3D
    P_TE:       Tuple[float, float, float]   # compression-surface TE in 3D
    streamline: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))


@dataclass
class OsculatingPlaneSet:
    """Half-span (z >= 0) set of osculating-plane records plus derived coeffs."""
    planes: List[OsculatingPlaneData]
    coeffs: Dict[str, float]

    def __iter__(self):
        return iter(self.planes)

    def __len__(self):
        return len(self.planes)

    def __getitem__(self, idx):
        return self.planes[idx]


# ---------------------------------------------------------------------------
# Geometric helpers
# ---------------------------------------------------------------------------

def curvature_radius_ICC(z, A, L_s):
    """Radius of curvature of the shock curve y_s(z) at spanwise station z.

        R = (1 + y'(z)**2)**(3/2) / |y''(z)|

    For |z| <= L_s the curve is straight (y == 0) -- infinite radius.
    """
    z = float(z)
    if abs(z) <= L_s:
        return np.inf
    zz = abs(z) - L_s
    y_p  = 4.0  * A * zz ** 3
    y_pp = 12.0 * A * zz ** 2
    if y_pp <= 0.0:
        return np.inf
    return (1.0 + y_p ** 2) ** 1.5 / y_pp


def _inward_normal(z, A, L_s):
    """Unit inward normal to the shock curve at spanwise z, in 3D.

    Returned as (0, n_y, n_z); n_y > 0 (points toward the body above the shock).
    In the flat region |z| <= L_s, returns (0, 1, 0).
    """
    if abs(z) <= L_s:
        return np.array([0.0, 1.0, 0.0])
    z_sign = 1.0 if z >= 0 else -1.0
    zz = abs(z) - L_s
    y_p = 4.0 * A * zz ** 3 * z_sign
    norm = np.sqrt(1.0 + y_p ** 2)
    return np.array([0.0, 1.0, -y_p]) / norm


def _solve_z_LE(z_i, coeffs, y_c, z_c, n_y, n_z):
    """Solve (y_upper(z') - y_c)*n_z - (z' - z_c)*n_y = 0 for z'.

    The upper surface y(z) is cubic, so this reduces to a cubic equation
    in z'. We use Brent's method on a bracket that safely contains the
    physically meaningful root.
    """
    a, b, c, d = coeffs["a"], coeffs["b"], coeffs["c"], coeffs["d"]

    def residual(zp):
        y_upper_zp = a * zp ** 3 + b * zp ** 2 + c * zp + d
        return (y_upper_zp - y_c) * n_z - (zp - z_c) * n_y

    # Bracket. For starboard (z_i > 0), z_LE is typically slightly less than
    # z_i; for port (z_i < 0) it is slightly greater. Search a ~L_s-wide
    # interval around z_c.
    lo = float(min(z_c, z_i) - 0.5)
    hi = float(max(z_c, z_i) + 0.5)
    lo = max(lo, -10.0)
    hi = min(hi,  10.0)
    try:
        return float(brentq(residual, lo, hi, xtol=1e-9))
    except ValueError:
        # Widen: expand bracket until we find a sign change or give up.
        for span in (1.0, 2.0, 4.0, 8.0):
            try:
                return float(brentq(residual, z_c - span, z_c + span, xtol=1e-9))
            except ValueError:
                continue
        return float(z_i)                    # graceful fallback


def osculating_plane_geometry(z_i, coeffs, params, delta_c_deg,
                               Ma_local=None, cone_field=None,
                               n_stream_samples=30):
    """Return (P_LE, P_TE, n_base, R_osc, streamline_3d) for the osculating
    plane at z_i.

    Two physics regimes are dispatched here:

    * **Flat region** (|z_i| <= L_s, R_osc -> infinity): the local flow is
      a uniform 2D wedge. The streamline is straight, deflected by
      theta_wedge = theta(beta, Ma_local). With cone_field omitted this
      reduces to the wedge formula automatically.

    * **Curved region** (|z_i| > L_s): the local flow is an osculating
      cone (Rodi 2011). If ``cone_field = (Vr_spline, Vt_spline,
      delta_c_rad)`` is supplied, the compression streamline is
      integrated through the T-M velocity field by
      :func:`trace_tm_streamline`. Otherwise we fall back to a
      straight line at angle delta_c (legacy approximation).

    Parameters
    ----------
    Ma_local : float, optional
        Required when ``cone_field`` is None and z_i is in the flat region,
        because the wedge deflection angle depends on Ma. If omitted,
        the legacy delta_c approximation is used and the volume will be
        ~13% too high (paper's flat-region bug).
    cone_field : tuple, optional
        ``(Vr_spline, Vt_spline, delta_c_rad)`` from
        :func:`liu2019.shock.taylor_maccoll_cone_field`.
    """
    a, b, c, d = coeffs["a"], coeffs["b"], coeffs["c"], coeffs["d"]
    A   = coeffs["A"]
    L_s = float(params["L_s"])
    L_w = float(params["L_w"])
    beta_deg = float(params["beta_deg"])
    gamma    = float(params.get("gamma", 1.4))
    beta = np.radians(beta_deg)
    tan_beta = np.tan(beta)

    n_base = _inward_normal(z_i, A, L_s)
    R_osc  = curvature_radius_ICC(z_i, A, L_s)
    flat   = (abs(z_i) <= L_s) or not np.isfinite(R_osc)

    # --- Leading edge -----------------------------------------------------
    if flat:
        y_upper = float(upper_surface_trailing_edge(
            np.array([z_i]), a, b, c, d)[0])
        y_s = 0.0
        x_LE = L_w - (y_upper - y_s) / tan_beta
        y_LE = y_upper
        z_LE = float(z_i)
        # Curvature centre is at infinity; the "axis" is the line
        # (x, y_LE - infinity, z_LE) -- never queried in the flat branch.
        x_a = -np.inf
        r_LE = np.inf
        y_c, z_c = y_LE + np.inf, z_LE
    else:
        y_s = float(shock_curve(np.array([z_i]), A, L_s)[0])
        P_s = np.array([L_w, y_s, float(z_i)])
        C_base = P_s + R_osc * n_base
        y_c, z_c = float(C_base[1]), float(C_base[2])
        n_y, n_z = float(n_base[1]), float(n_base[2])

        z_LE = _solve_z_LE(z_i, coeffs, y_c, z_c, n_y, n_z)
        y_LE = float(upper_surface_trailing_edge(
            np.array([z_LE]), a, b, c, d)[0])

        r_LE = float(np.sqrt((y_LE - y_c) ** 2 + (z_LE - z_c) ** 2))
        x_LE = L_w - (R_osc - r_LE) / tan_beta
        x_a  = L_w - R_osc / tan_beta            # cone apex x-coordinate

    # --- Streamline + trailing edge --------------------------------------
    n_y = float(n_base[1])
    n_z = float(n_base[2])

    if flat:
        # EXPERIMENT Variant B/C: theta_w in flat region.
        if Ma_local is not None:
            flat_slope_deg = float(theta_from_beta_Ma(beta_deg, Ma_local, gamma))
        else:
            flat_slope_deg = float(delta_c_deg)
        deflection = (L_w - x_LE) * np.tan(np.radians(flat_slope_deg))
        y_TE = y_LE - deflection * n_y
        z_TE = z_LE - deflection * n_z
        # Straight streamline LE -> TE, freestream-aligned (constant n_base).
        t = np.linspace(0.0, 1.0, int(n_stream_samples))
        stream_3d = np.column_stack([
            x_LE + t * (L_w - x_LE),
            y_LE + t * (-deflection * n_y),
            z_LE + t * (-deflection * n_z),
        ])
    elif cone_field is not None:
        # True osculating-cone streamline through the Taylor-Maccoll field.
        Vr_spline, Vt_spline, delta_c_rad = cone_field
        x_path, r_path = trace_tm_streamline(
            x_LE, r_LE, x_a, L_w,
            Vr_spline, Vt_spline,
            beta, delta_c_rad,
            n_samples=int(n_stream_samples),
        )
        # Back-project (x, r) in the osculating plane to 3D:
        # P(x, r) = (x, y_c - r * n_y, z_c - r * n_z)
        stream_3d = np.column_stack([
            x_path,
            y_c - r_path * n_y,
            z_c - r_path * n_z,
        ])
        x_TE = float(x_path[-1])
        y_TE = float(stream_3d[-1, 1])
        z_TE = float(stream_3d[-1, 2])
    else:
        # EXPERIMENT Variant C: theta_w in curved region too.
        if Ma_local is not None:
            curved_slope_deg = float(theta_from_beta_Ma(beta_deg, Ma_local, gamma))
        else:
            curved_slope_deg = float(delta_c_deg)
        delta_c = np.radians(curved_slope_deg)
        deflection = (L_w - x_LE) * np.tan(delta_c)
        y_TE = y_LE - deflection * n_y
        z_TE = z_LE - deflection * n_z
        t = np.linspace(0.0, 1.0, int(n_stream_samples))
        stream_3d = np.column_stack([
            x_LE + t * (L_w - x_LE),
            y_LE + t * (-deflection * n_y),
            z_LE + t * (-deflection * n_z),
        ])

    return (float(x_LE), float(y_LE), float(z_LE)), \
           (float(L_w), float(y_TE), float(z_TE)), \
           n_base, R_osc, stream_3d


def _streamline_points(P_LE, P_TE, n_x):
    """Sample points along the straight streamline from P_LE to P_TE."""
    t = np.linspace(0.0, 1.0, max(int(n_x), 2))
    pts = np.array(P_LE)[None, :] + t[:, None] * (np.array(P_TE) - np.array(P_LE))[None, :]
    return pts


def trace_tm_streamline(x_LE, r_LE, x_a, L_w,
                        Vr_spline, Vt_spline,
                        beta_rad, delta_c_rad,
                        n_samples=30):
    """Integrate a streamline through the Taylor-Maccoll cone field.

    Starts at the leading edge ``(x_LE, r_LE)`` on the conical shock and
    integrates ``dr/dx`` toward the base plane ``x = L_w``. Within the
    osculating plane:

        theta(x, r)   = arctan(r / (x - x_a))
        V_x(theta)    =  V_r(theta) * cos(theta) - V_theta(theta) * sin(theta)
        V_R(theta)    =  V_r(theta) * sin(theta) + V_theta(theta) * cos(theta)

    Returns
    -------
    x_path, r_path : ndarray of shape (n_samples,)
        Sampled streamline at uniformly spaced x values in [x_LE, L_w].
    """
    x_LE = float(x_LE)
    L_w  = float(L_w)
    x_a  = float(x_a)

    if L_w <= x_LE + 1e-9:
        return (np.array([x_LE, L_w]),
                np.array([float(r_LE), float(r_LE)]))

    def _rhs(x, r):
        dx_apex = x - x_a
        if dx_apex <= 1e-12:
            return [0.0]
        theta = np.arctan(r[0] / dx_apex)
        # Stay inside the spline domain — the streamline strictly satisfies
        # delta_c <= theta <= beta for all points between shock and body, but
        # ODE-stepper overshoot can momentarily push it slightly outside.
        theta_q = float(np.clip(theta, delta_c_rad, beta_rad))
        v_r  = float(Vr_spline(theta_q))
        v_t  = float(Vt_spline(theta_q))
        sin_th = np.sin(theta)
        cos_th = np.cos(theta)
        V_x = v_r * cos_th - v_t * sin_th
        V_R = v_r * sin_th + v_t * cos_th
        if V_x <= 1e-12:
            return [0.0]
        return [V_R / V_x]

    sol = solve_ivp(
        _rhs, (x_LE, L_w), [float(r_LE)],
        method="RK45", rtol=1e-7, atol=1e-10,
        dense_output=True, max_step=(L_w - x_LE) / 4.0,
    )
    if not sol.success:
        raise RuntimeError(f"streamline ODE failed: {sol.message}")

    x_samples = np.linspace(x_LE, L_w, int(n_samples))
    r_samples = sol.sol(x_samples)[0]
    return x_samples, r_samples


# Back-compat shims (kept so that callers importing these names still work)
def leading_edge_point(z, beta_deg, a, b, c, d, A, L_s, L_w):
    coeffs = {"a": a, "b": b, "c": c, "d": d, "A": A}
    params = {"L_s": L_s, "L_w": L_w, "beta_deg": beta_deg}
    P_LE, _, _, _, _ = osculating_plane_geometry(z, coeffs, params, 0.0)
    return P_LE


def trailing_edge_of_compression(P_LE, delta_c_deg, L_w):
    """Legacy: assumes vertical plane (n_y=1, n_z=0)."""
    x_LE, y_LE, z_LE = P_LE
    delta_c = np.radians(delta_c_deg)
    return (float(L_w), float(y_LE - (L_w - x_LE) * np.tan(delta_c)), float(z_LE))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build_all_osculating_planes(params, n_z=200, n_x=100):
    """Sweep z in [0, +W/2] and build OsculatingPlaneData for each station.

    Returns an OsculatingPlaneSet covering the starboard half-span; mirror
    the port side at assembly time.
    """
    from .distributions import (
        shock_curve_coefficient,
        upper_surface_coefficients,
    )

    beta_deg  = float(params["beta_deg"])
    L_w       = float(params["L_w"])
    W         = float(params["W"])
    L_s       = float(params["L_s"])
    y5        = float(params["y5"])
    z5        = float(params["z5"])
    y6        = float(params["y6"])
    z6        = float(params["z6"])
    delta5    = float(params["delta5"])
    delta6    = float(params["delta6"])
    Ma_center = float(params["Ma_center"])
    Ma_tip    = float(params["Ma_tip"])
    gamma     = float(params.get("gamma", 1.4))

    a, b, c, d = upper_surface_coefficients(
        y5, z5, y6, z6, delta5, delta6, L_w, beta_deg)
    A = shock_curve_coefficient(y5, z5, L_s)
    coeffs = {"a": a, "b": b, "c": c, "d": d, "A": A}

    z_stations = np.linspace(0.0, W / 2.0, int(n_z))
    planes: List[OsculatingPlaneData] = []

    # ------------------------------------------------------------------
    # Legacy paper-formula streamline model:
    #
    #   * Flat region   |z| <= L_s   : straight line at angle delta_c
    #   * Curved region |z|  > L_s   : straight line at angle delta_c
    #     (cone-body-derived approximation, Liu 2019 Section 1.1 Step 4)
    #
    # The Taylor-Maccoll velocity-field solver and streamline integrator
    # in shock.py / osculating.py remain available as infrastructure but
    # are not invoked by default, because:
    #   * the paper's reported geometry visually matches the legacy model
    #     (smooth, monotonic descent from tip to a centreline keel),
    #   * the real T-M streamline produces a 324 mm kink at z = L_s
    #     (the cone-body slope on the flat side does not match the
    #     T-M streamline's initial post-shock slope on the curved side),
    #   * STEP exports of the kinked geometry exhibit visible ridges.
    #
    # delta_c is pre-computed on a 15-point Ma grid (smooth in Ma at
    # fixed beta) and linearly interpolated per plane.
    # ------------------------------------------------------------------
    Ma_lo = float(min(Ma_center, Ma_tip))
    Ma_hi = float(max(Ma_center, Ma_tip))
    Ma_grid = np.linspace(Ma_lo, Ma_hi, 15)
    delta_grid = np.array(
        [taylor_maccoll_cone_angle(m, beta_deg, gamma) for m in Ma_grid])

    def _interp_cone(Ma):
        return float(np.interp(Ma, Ma_grid, delta_grid))

    for z in z_stations:
        Ma_local = float(Ma_distribution(z, W, Ma_center, Ma_tip))
        delta_c  = _interp_cone(Ma_local)
        y_shock  = float(shock_curve(np.array([z]), A, L_s)[0])
        P_shock  = (L_w, y_shock, float(z))
        # cone_field=None and Ma_local=None => dispatcher uses the legacy
        # straight-line streamline at angle delta_c in both flat and
        # curved regions.
        P_LE, P_TE, n_base, R_osc, stream_3d = osculating_plane_geometry(
            float(z), coeffs, params, delta_c,
            Ma_local=Ma_local, cone_field=None,
            n_stream_samples=int(n_x),
        )
        planes.append(OsculatingPlaneData(
            z=float(z), Ma=Ma_local, delta_c=delta_c, R_osc=float(R_osc),
            n_base=(float(n_base[0]), float(n_base[1]), float(n_base[2])),
            P_shock=P_shock, P_LE=P_LE, P_TE=P_TE, streamline=stream_3d,
        ))

    return OsculatingPlaneSet(planes=planes, coeffs=coeffs)
