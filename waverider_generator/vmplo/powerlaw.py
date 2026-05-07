"""Power-law axisymmetric generating body + per-plane flowfield dispatcher.

Spec reference: VMPLO_implementation_prompt.md §2, §6, ``vmplo/powerlaw.py``.

Each osculating plane is modelled as an axisymmetric power-law body:

    r(x) = R_b * (x/L)^n ,   x in [x_LE, L]

with ``R_b`` chosen so that the body slope at ``x_LE`` equals the post-shock
flow deflection angle (attached-shock condition).  For ``n=1`` the body is
a cone and the Taylor-Maccoll ODE applies exactly, giving a straight-line
streamline along the cone surface.  For ``n != 1`` the flow is not
self-similar and must be computed via the axisymmetric Method of
Characteristics (``vmplo.moc``).

The dispatcher :func:`solve_osculating_plane` is the single entry point
used by :class:`osculating.OsculatingAssembly`.
"""

from __future__ import annotations

import logging
import numpy as np
from scipy.integrate import solve_ivp

from waverider_generator.vmplo.shock import (
    DetachedShockError, theta_from_beta_Ma, oblique_shock_ratios,
)

logger = logging.getLogger(__name__)

# Below this |n - 1| threshold the body is treated as a cone and the
# Taylor-Maccoll path is used.  Above it, the MOC solver is invoked.
CONE_TOL = 0.02


# ---------------------------------------------------------------------- #
#  Power-law body geometry                                                #
# ---------------------------------------------------------------------- #

class PowerLawBody:
    """Axisymmetric power-law body ``r(x) = R_b * (x/L)^n``.

    Provides geometry, slope, and slope-angle queries needed by the MOC
    wall-point solver.  The ``from_shock_condition`` classmethod builds
    the body directly from the per-plane flowfield data (closed form,
    no iteration).
    """

    def __init__(self, n: float, L: float, R_b: float,
                 gamma: float = 1.4):
        self.n = float(n)
        self.L = float(L)
        self.R_b = float(R_b)
        self.gamma = float(gamma)

    # -- geometry --------------------------------------------------------

    def radius(self, x):
        x = np.asarray(x, dtype=float)
        return self.R_b * (x / self.L) ** self.n

    def slope(self, x):
        x = np.asarray(x, dtype=float)
        # Avoid division-by-zero at x=0 when n<1
        x_safe = np.where(x <= 0.0, 1e-12, x)
        return self.n * self.R_b / self.L * (x_safe / self.L) ** (self.n - 1.0)

    def slope_angle_rad(self, x):
        return np.arctan(self.slope(x))

    def slope_angle_deg(self, x):
        return np.degrees(self.slope_angle_rad(x))

    # -- factory --------------------------------------------------------

    @classmethod
    def from_shock_condition(cls, n: float, L: float, x_LE: float,
                             theta_deg: float,
                             gamma: float = 1.4) -> "PowerLawBody":
        """Construct the body such that ``slope(x_LE) == tan(theta_deg)``.

        Closed-form: R_b = tan(theta) * L / (n * (x_LE / L)^(n-1)).
        Requires ``x_LE > 0`` for n<1 (singular otherwise).
        """
        if x_LE <= 0.0:
            raise ValueError(
                f"x_LE must be > 0 (got {x_LE}); n<1 would be singular.")
        theta = np.radians(theta_deg)
        R_b = np.tan(theta) * L / (n * (x_LE / L) ** (n - 1.0))
        return cls(n, L, R_b, gamma)

    def __repr__(self):
        return (f"PowerLawBody(n={self.n:.4f}, L={self.L}, "
                f"R_b={self.R_b:.6f})")


# ---------------------------------------------------------------------- #
#  Taylor-Maccoll cone solver (n == 1)                                    #
# ---------------------------------------------------------------------- #

def taylor_maccoll_cone_angle(Ma: float, beta_deg: float,
                              gamma: float = 1.4) -> float:
    """Cone half-angle ``delta_c`` (degrees) for a conical shock at
    ``beta_deg`` and freestream ``Ma``.

    Integrates the Taylor-Maccoll ODE inward from ``theta=beta`` until
    the radial velocity perpendicular to the ray, ``V_theta``, reaches
    zero (the cone surface, where flow is parallel to the body).
    Inputs:  Ma (freestream), beta (shock angle deg), gamma.
    Output:  cone half-angle in degrees.
    """
    beta = np.radians(beta_deg)
    g = gamma

    # Initial conditions at the shock via oblique-shock relations.
    post = oblique_shock_ratios(Ma, beta_deg, gamma)
    Ma2 = post["Ma2"]
    theta_deg = post["theta_deg"]
    theta_flow = np.radians(theta_deg)

    # Non-dimensional velocity magnitude V' = V / V_max where
    # V_max = sqrt(2 h0 / (gamma-1)) and h0 is stagnation enthalpy.
    # Under our normalisation V' = 1 / sqrt(1 + 2/((g-1)*Ma2^2)).
    V_prime = 1.0 / np.sqrt(1.0 + 2.0 / ((g - 1.0) * Ma2**2))

    # Velocity components in the shock-ray frame at the shock surface:
    # post-shock flow deflected by theta_flow below the freestream, and the
    # ray direction makes angle beta with the freestream.  Angle between
    # the flow and the ray is (beta - theta_flow).
    V_r0 = V_prime * np.cos(beta - theta_flow)
    V_t0 = -V_prime * np.sin(beta - theta_flow)   # V_theta < 0 (compression)

    def tm_rhs(theta: float, y):
        V_r, V_t = y
        term = (g - 1.0) / 2.0 * (1.0 - V_r**2 - V_t**2)
        denom = term - V_t**2
        if abs(denom) < 1e-14:
            return [0.0, 0.0]
        # Anderson, Modern Compressible Flow 3rd ed. eq. 10.16:
        #   dV_θ/dθ = (V_r V_θ² − term (2 V_r + V_θ cot θ)) / (term − V_θ²)
        # (Not V_θ V_r² — easy to swap and gives wildly wrong δ_c.)
        dV_t = (V_r * V_t**2 - term * (2.0 * V_r + V_t / np.tan(theta))) / denom
        dV_r = V_t
        return [dV_r, dV_t]

    def cone_surface(theta: float, y):
        return y[1]   # V_theta = 0 at the cone surface
    cone_surface.terminal = True
    cone_surface.direction = 1.0   # V_t was negative; crosses zero from below

    # Integrate from beta down toward 0; stop at V_theta = 0.
    sol = solve_ivp(
        tm_rhs, (beta, 1e-4), [V_r0, V_t0],
        events=cone_surface,
        rtol=1e-9, atol=1e-11, max_step=np.radians(0.5),
    )
    if sol.t_events[0].size == 0:
        raise DetachedShockError(
            f"Taylor-Maccoll integration failed to reach V_theta=0 "
            f"for Ma={Ma}, beta={beta_deg}°.")
    delta_c_rad = float(sol.t_events[0][0])
    return float(np.degrees(delta_c_rad))


# ---------------------------------------------------------------------- #
#  Per-plane dispatcher                                                   #
# ---------------------------------------------------------------------- #

def solve_osculating_plane(Ma_i: float, n_i: float, beta_design_deg: float,
                           x_LE: float, L: float,
                           gamma: float = 1.4,
                           n_moc_init: int = 12,
                           n_moc_cols: int = 30,
                           n_stream: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """Leading-edge streamline ``(x, r)`` for one VMPLO osculating plane.

    Dispatches to Taylor-Maccoll (n ≈ 1, cone) or the axisymmetric MOC
    (general power-law body).  Returns two 1-D arrays of length
    ``n_stream + 1`` tracing the LE streamline from ``(x_LE, r_LE)`` to
    ``x = L``.

    MOC failures fall back to the Taylor-Maccoll straight-line path
    (logged at WARNING level).  This matches the plan's risk mitigation:
    one bad plane doesn't kill the whole geometry.
    """
    theta_i = theta_from_beta_Ma(beta_design_deg, Ma_i, gamma)
    body = PowerLawBody.from_shock_condition(n_i, L, x_LE, theta_i, gamma)
    r_LE = body.radius(x_LE)

    if abs(n_i - 1.0) < CONE_TOL:
        return _trace_cone(body, Ma_i, beta_design_deg, x_LE, r_LE,
                           gamma, n_stream)

    # MOC path (imported lazily to avoid a cycle with osculating.py
    # during package init).
    try:
        from waverider_generator.vmplo.moc import (
            MOCGrid, initial_data_line, extract_streamline,
        )
        # Phase 4 Fix 2: pass the power-law exponent so initial_data_line
        # can auto-boost the resolution and switch to Chebyshev clustering
        # when n is low and post-shock alpha gradients are steep.
        shock_pts = initial_data_line(
            x_LE, r_LE, beta_design_deg, Ma_i, gamma,
            N=n_moc_init, n_body=n_i)
        grid = MOCGrid(shock_pts, body, gamma)
        grid.march(n_columns=n_moc_cols)

        post = oblique_shock_ratios(Ma_i, beta_design_deg, gamma)
        alpha0 = np.radians(post["theta_deg"])
        xs, rs = extract_streamline(grid, x_LE, r_LE, alpha0, n_steps=n_stream)
        if xs.size < 2 or not np.all(np.isfinite(rs)):
            raise RuntimeError("MOC produced non-finite streamline.")
        return xs, rs
    except Exception as exc:
        logger.warning(
            "[VMPLO] MOC failed for Ma=%.3f, n=%.3f, beta=%.2f° (%s); "
            "falling back to Taylor-Maccoll straight-line trace.",
            Ma_i, n_i, beta_design_deg, exc)
        return _trace_cone(body, Ma_i, beta_design_deg, x_LE, r_LE,
                           gamma, n_stream)


def _trace_cone(body: PowerLawBody, Ma_i: float, beta_deg: float,
                x_LE: float, r_LE: float, gamma: float,
                n_stream: int) -> tuple[np.ndarray, np.ndarray]:
    """Taylor-Maccoll streamline: straight line r(x) = r_LE + tan(delta_c)(x-x_LE).

    Returns (n_stream+1,) arrays.
    """
    try:
        delta_c_deg = taylor_maccoll_cone_angle(Ma_i, beta_deg, gamma)
    except Exception as exc:
        logger.warning(
            "[VMPLO] Taylor-Maccoll failed for Ma=%.3f, beta=%.2f° (%s); "
            "using local body slope.", Ma_i, beta_deg, exc)
        delta_c_deg = np.degrees(body.slope_angle_rad(x_LE))

    xs = np.linspace(x_LE, body.L, n_stream + 1)
    rs = r_LE + np.tan(np.radians(delta_c_deg)) * (xs - x_LE)
    return xs, rs
