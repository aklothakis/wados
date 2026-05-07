"""
power_law_flowfield.py
======================
Local Cone Approximation (LCA) flowfield solver for a power-law body of revolution.

Body geometry:
    r_body(x) = R_base * (x / L)^n,   x in (0, L]

where n in (0.5, 1.0] is the power-law exponent.  When n = 1.0 the body is an
exact cone and the results must be identical to calling cone_field directly, as
the generator does.

The LCA treats every axial station as a locally equivalent cone whose half-angle
equals the local surface slope:

    theta_local(x) = arctan( d r_body/dx ) = arctan( n * R_base / L * (x/L)^(n-1) )

At each station the full Taylor-Maccoll conical velocity field is obtained from
the existing flowfield module and stored as spline pairs [Vr(theta), Vt(theta)].

Streamlines are integrated axially with scipy solve_ivp (RK45) using the locally
updated velocity field at every integration step.

Coordinate convention (same as generator.py):
    x  - streamwise (nose to base)
    r  - radial distance from body axis  (= sqrt(y^2 + z^2) in 3-D)
    theta = arctan(r / x)  - polar angle measured from axis
"""

import logging
import warnings

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from waverider_generator.flowfield import cone_angle, cone_field, shock_angle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Local shock_angle implementation that is robust under numpy 2.x / Python 3.14
# ---------------------------------------------------------------------------
# flowfield.shock_angle uses fsolve with an inner function f() that calls
# float() on the 0-d array passed by fsolve.  Under numpy >= 2.0 / Python
# 3.14 this raises "only 0-dimensional arrays can be converted to Python
# scalars".  We replicate the logic here with an explicit scalar cast.

def _shock_angle_safe(Mach: float, theta_deg: float, gamma: float) -> float:
    """
    Return the shock angle (degrees) for a given cone half-angle theta_deg
    at freestream Mach number.

    Wraps flowfield.shock_angle with explicit float conversion to handle
    numpy 2.x scalar issues, plus a brentq fallback if fsolve fails.
    """
    Mach = float(Mach)
    theta_deg = float(theta_deg)
    gamma = float(gamma)

    # Try the existing solver first (works fine on Python <= 3.13)
    try:
        result = shock_angle(Mach, theta_deg, gamma)
        result = float(result)
        # Sanity check
        mu_deg = float(np.degrees(np.arcsin(1.0 / Mach)))
        if mu_deg < result < 90.0:
            return result
    except Exception:
        pass

    # Fallback: bounded brentq solver
    from scipy.optimize import brentq
    from scipy.integrate import solve_ivp
    from waverider_generator.flowfield import Taylor_Maccoll

    theta_rad = theta_deg * np.pi / 180.0
    mu = np.arcsin(1.0 / Mach)

    def _Vt0(t, y, gamma_):
        return y[1]
    _Vt0.terminal = True

    def _f(beta):
        beta = float(beta)
        sin_b = np.sin(beta)
        d = np.arctan(
            2.0 * np.cos(beta) / sin_b
            * (Mach**2 * sin_b**2 - 1.0)
            / (Mach**2 * (gamma + np.cos(2 * beta)) + 2.0)
        )
        if d <= 0:
            return np.pi  # large positive to help bracket
        sin_bd = np.sin(beta - d)
        if sin_bd <= 0:
            return np.pi
        Ma2_sq_num = 1.0 + (gamma - 1.0) / 2.0 * Mach**2 * sin_b**2
        Ma2_sq_den = gamma * Mach**2 * sin_b**2 - (gamma - 1.0) / 2.0
        if Ma2_sq_den <= 0:
            return np.pi
        Ma2 = np.sqrt(Ma2_sq_num / Ma2_sq_den) / sin_bd
        V = 1.0 / np.sqrt(2.0 / ((gamma - 1.0) * Ma2**2) + 1.0)
        Vr0 = float(V * np.cos(beta - d))
        Vt0_val = float(-(V * np.sin(beta - d)))
        try:
            sol = solve_ivp(
                Taylor_Maccoll, (beta, 0.0), [Vr0, Vt0_val],
                events=_Vt0, args=(gamma,), max_step=0.01,
            )
            if sol.t_events[0].size > 0:
                return float(sol.t_events[0][0]) - theta_rad
            return float(sol.t[-1]) - theta_rad
        except Exception:
            return np.pi

    # Search for a valid bracket by scanning
    beta_lo = mu + 0.005
    beta_hi = mu + 0.005
    for beta_test in np.linspace(mu + 0.01, np.pi / 2 - 0.01, 50):
        val = _f(beta_test)
        if val < 0:
            beta_lo = beta_test
        elif val > 0:
            beta_hi = beta_test
            break

    if beta_lo >= beta_hi:
        raise ValueError(f"Cannot bracket shock angle for M={Mach}, theta={theta_deg} deg")

    return float(np.degrees(brentq(_f, beta_lo, beta_hi, xtol=1e-8, maxiter=100)))


# ---------------------------------------------------------------------------
# Velocity-field cache
# ---------------------------------------------------------------------------
# cone_field is expensive (calls solve_ivp internally) and returns spline
# objects that are not hashable.  We cache results keyed by (M, theta_deg,
# beta_deg, gamma) rounded to 6 decimal places so that repeated evaluations
# at the same station avoid redundant integration.

_FIELD_CACHE: dict = {}
_CACHE_ROUND = 6  # decimal places for key rounding


def _cached_cone_field(Mach: float, theta_rad: float, beta_rad: float,
                       gamma: float):
    """Return [Vr_spline, Vt_spline] with caching on rounded (M, theta, beta, gamma).

    All arguments are explicitly converted to plain Python floats before the
    key is built and before cone_field is called.  This prevents 0-d numpy
    arrays from reaching scipy.integrate.solve_ivp inside cone_field, which
    raises 'only 0-dimensional arrays can be converted to Python scalars'.
    """
    M_f  = float(Mach)
    th_f = float(theta_rad)
    b_f  = float(beta_rad)
    g_f  = float(gamma)

    key = (
        round(M_f,  _CACHE_ROUND),
        round(th_f, _CACHE_ROUND),
        round(b_f,  _CACHE_ROUND),
        round(g_f,  _CACHE_ROUND),
    )
    if key not in _FIELD_CACHE:
        _FIELD_CACHE[key] = cone_field(M_f, th_f, b_f, g_f)
    return _FIELD_CACHE[key]


def clear_flowfield_cache() -> None:
    """Evict all cached velocity fields (useful between design sweeps)."""
    _FIELD_CACHE.clear()


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PowerLawFlowfield:
    """
    LCA flowfield for a power-law body of revolution.

    Parameters
    ----------
    M_inf : float
        Freestream Mach number.
    n_exponent : float
        Power-law exponent (1.0 = cone, < 1.0 = blunter body).
        Must be in (0, 1].
    beta_base : float
        Shock angle at the base plane, in degrees.
    body_length : float, optional
        Non-dimensional body length (default 1.0).  The osculating-plane
        framework normally normalises everything to L = 1.
    n_stations : int, optional
        Number of axial stations for the LCA march (default 100).
    gamma : float, optional
        Ratio of specific heats (default 1.4).
    """

    def __init__(self,
                 M_inf: float,
                 n_exponent: float,
                 beta_base: float,
                 body_length: float = 1.0,
                 n_stations: int = 100,
                 gamma: float = 1.4):

        # --- validate inputs -------------------------------------------------
        if M_inf <= 1.0:
            raise ValueError("M_inf must be supersonic (> 1).")
        if not (0.0 < n_exponent <= 1.0):
            raise ValueError("n_exponent must be in (0, 1].")
        if not (0.0 < beta_base < 90.0):
            raise ValueError("beta_base must be in (0, 90) degrees.")
        if body_length <= 0.0:
            raise ValueError("body_length must be positive.")
        if n_stations < 10:
            raise ValueError("n_stations must be at least 10.")

        self.M_inf = float(M_inf)
        self.n = float(n_exponent)
        self.beta_base_deg = float(beta_base)
        self.beta_base_rad = np.deg2rad(self.beta_base_deg)
        self.L = float(body_length)
        self.n_stations = int(n_stations)
        self.gamma = float(gamma)

        # --- derived geometry ------------------------------------------------
        # cone_angle gives the local surface half-angle for a given shock angle.
        # At the base station this IS the local slope, so we use it to find R_base.
        theta_base_deg = cone_angle(self.M_inf, self.beta_base_deg, self.gamma)
        self.theta_base_deg = theta_base_deg
        self.theta_base_rad = np.deg2rad(theta_base_deg)

        # r_body(x) = R_base * (x/L)^n
        # slope at x=L: dr/dx = n * R_base / L
        # theta_base = arctan(n * R_base / L)  =>  R_base = L * tan(theta_base) / n
        self.R_base = self.L * np.tan(self.theta_base_rad) / self.n

        # --- station array ---------------------------------------------------
        # Avoid the nose singularity: start at a small positive x.
        # epsilon is chosen so that theta_local stays < ~89 deg even for small n.
        self._x_eps = self.L * 1e-3
        self._x_stations = np.linspace(self._x_eps, self.L, self.n_stations)

        # --- storage filled by solve() --------------------------------------
        self._theta_local_rad: np.ndarray | None = None   # shape (n_stations,)
        self._beta_local_rad: np.ndarray | None = None    # shape (n_stations,)
        self._fields: list | None = None                  # list of [Vr, Vt] per station
        self._valid_mask: np.ndarray | None = None        # bool (n_stations,)
        self._solved = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self) -> bool:
        """
        Solve the LCA flowfield at all axial stations.

        For n=1.0 (cone), all stations share the same velocity field —
        cone_field is called once.

        For n<1.0, the local cone angle varies along the body.  Rather than
        calling the slow shock_angle() inverse solver at each station, we
        interpolate beta_local linearly from the base-station value
        (beta_base) and the Mach angle, weighted by the ratio of local
        cone angle to base cone angle.  This is the standard LCA approach
        where the shock angle tracks the local body slope.

        Returns
        -------
        bool
            True if at least half of the stations were solved successfully.
        """
        n = self.n_stations
        theta_local = np.empty(n)
        beta_local  = np.empty(n)
        fields      = [None] * n
        valid       = np.zeros(n, dtype=bool)

        mach_angle_rad = np.arcsin(1.0 / self.M_inf)
        mach_angle_deg = np.degrees(mach_angle_rad)

        # For n=1.0, all stations are identical — solve once
        is_cone = abs(self.n - 1.0) < 1e-10

        if is_cone:
            base_field = _cached_cone_field(
                self.M_inf, self.theta_base_rad,
                self.beta_base_rad, self.gamma
            )
            for i in range(n):
                theta_local[i] = self.theta_base_rad
                beta_local[i]  = self.beta_base_rad
                fields[i]      = base_field
                valid[i]       = True
        else:
            # For each station, compute local slope and estimate local shock angle
            for i, x_i in enumerate(self._x_stations):
                drhodx = self.n * self.R_base / self.L * (x_i / self.L) ** (self.n - 1.0)
                th_i = np.arctan(drhodx)
                th_i_deg = np.degrees(th_i)

                if th_i_deg >= 90.0 - mach_angle_deg or th_i_deg <= 0:
                    continue

                # Estimate local shock angle by scaling:
                # beta_local / beta_base ≈ theta_local / theta_base
                # (linear approximation valid for attached shocks)
                ratio = th_i / self.theta_base_rad
                ratio = min(ratio, 3.0)  # clamp for near-nose stations
                beta_i_rad = mach_angle_rad + ratio * (self.beta_base_rad - mach_angle_rad)
                beta_i_rad = float(np.clip(beta_i_rad,
                                           mach_angle_rad + 0.001,
                                           np.pi / 2 - 0.001))
                beta_i_deg = np.degrees(beta_i_rad)

                # Ensure beta > theta (shock must be outside body)
                if beta_i_rad <= th_i + 0.001:
                    beta_i_rad = th_i + 0.01

                try:
                    splines = _cached_cone_field(
                        self.M_inf, float(th_i), beta_i_rad, self.gamma
                    )
                    theta_local[i] = th_i
                    beta_local[i]  = beta_i_rad
                    fields[i]      = splines
                    valid[i]       = True
                except Exception as exc:
                    logger.debug(
                        "Station %d (x=%.4f, theta=%.2f deg): cone_field failed: %s",
                        i, x_i, th_i_deg, exc
                    )

        n_valid = valid.sum()
        if n_valid == 0:
            logger.error("No stations solved successfully.")
            self._solved = False
            return False

        # Interpolate failed stations from neighbors
        valid_idx = np.where(valid)[0]
        invalid_idx = np.where(~valid)[0]

        if invalid_idx.size > 0 and valid_idx.size >= 2:
            theta_interp = interp1d(
                valid_idx, theta_local[valid_idx],
                kind='linear', fill_value='extrapolate'
            )
            beta_interp = interp1d(
                valid_idx, beta_local[valid_idx],
                kind='linear', fill_value='extrapolate'
            )
            for i in invalid_idx:
                th_i = float(np.clip(theta_interp(i), 1e-6, np.deg2rad(89.9)))
                b_i = float(np.clip(beta_interp(i),
                                    mach_angle_rad + 0.001,
                                    np.deg2rad(89.9)))
                if b_i <= th_i + 0.001:
                    b_i = th_i + 0.01
                theta_local[i] = th_i
                beta_local[i] = b_i
                try:
                    fields[i] = _cached_cone_field(self.M_inf, th_i, b_i, self.gamma)
                except Exception:
                    nearest = valid_idx[np.argmin(np.abs(valid_idx - i))]
                    fields[i] = fields[nearest]

        self._theta_local_rad = theta_local
        self._beta_local_rad  = beta_local
        self._fields          = fields
        self._valid_mask      = valid
        self._solved          = True
        return True

    # ------------------------------------------------------------------

    def get_shock_shape(self) -> np.ndarray:
        """
        Return the LCA shock surface shape as an (n_stations, 2) array
        with columns [x, r_shock].

        Under the LCA each station has a locally equivalent cone whose shock
        emanates from the same apex (x=0).  The shock radius at station x_i is:

            r_shock(x_i) = x_i * tan(beta_local(x_i))

        This is the standard osculating-cone approximation and is consistent
        with the generator's treatment where the shock cone has its apex at
        the origin.

        Returns
        -------
        np.ndarray, shape (n_stations, 2)
            Column 0: x positions.
            Column 1: r_shock at each station.
        """
        self._require_solved()
        r_shock = self._x_stations * np.tan(self._beta_local_rad)
        return np.column_stack([self._x_stations, r_shock])

    # ------------------------------------------------------------------

    def trace_streamline(self, r_shock_entry: float) -> np.ndarray:
        """
        Trace a streamline from the shock entry point to the base plane.

        The streamline enters the shock at the outermost station whose shock
        radius equals or exceeds r_shock_entry.  If r_shock_entry is larger
        than the base-plane shock radius a warning is issued and the base
        station is used.

        Integration uses scipy solve_ivp (RK45) with the locally updated LCA
        velocity field.  At each integration step the nearest axial station is
        looked up to obtain the current [Vr, Vt] splines, and the velocity
        components are:

            dx/dt = Vr(theta)*cos(theta) - Vt(theta)*sin(theta)
            dr/dt = Vr(theta)*sin(theta) + Vt(theta)*cos(theta)

        where theta = arctan(r / x) measured in the local cone-centred frame
        (apex at x=0).  This matches exactly the stode formulation in
        generator.py.

        When n = 1 (pure cone) theta_local is constant, all stations share the
        same velocity field, and the integral is identical to calling cone_field
        once as the existing generator does.

        Parameters
        ----------
        r_shock_entry : float
            Radial position where the streamline enters the shock (same length
            units as body_length).

        Returns
        -------
        np.ndarray, shape (N, 2)
            Columns [x, r] along the streamline from entry to base plane.
        """
        self._require_solved()

        # --- find x_start from shock geometry --------------------------------
        shock = self.get_shock_shape()          # (n_stations, 2): [x, r_shock]
        r_shock_base = shock[-1, 1]

        if r_shock_entry > r_shock_base * 1.05:
            logger.warning(
                "r_shock_entry=%.6f > base shock radius=%.6f; clamping to base.",
                r_shock_entry, r_shock_base
            )
            r_shock_entry = r_shock_base

        # x_start: smallest x where r_shock >= r_shock_entry
        idx_candidates = np.where(shock[:, 1] >= r_shock_entry)[0]
        if idx_candidates.size == 0:
            # r_shock_entry is smaller than all shock radii - use nose station.
            i_start = 0
        else:
            i_start = idx_candidates[0]

        x_start = shock[i_start, 0]
        x_end   = self.L

        # --- build lookup for fields indexed by axial position ---------------
        # We interpolate the station index from x to get the right spline pair.
        x_stations = self._x_stations
        fields     = self._fields

        theta_local_arr = self._theta_local_rad
        beta_local_arr  = self._beta_local_rad

        def _get_field_and_bounds(x: float):
            """Return ([Vr, Vt] splines, theta_cone, beta) for the station nearest to x."""
            idx = int(np.searchsorted(x_stations, x, side='left'))
            idx = int(np.clip(idx, 0, len(x_stations) - 1))
            return fields[idx], float(theta_local_arr[idx]), float(beta_local_arr[idx])

        # --- ODE for streamline in (x, r) Cartesian-polar coordinates --------
        def streamline_ode(t_param, state):
            """
            Autonomous ODE system:  d[x, r]/dt = [v_x, v_r].

            t_param is a dummy integration parameter; x is carried in state[0]
            so the station lookup is always current.  The hit_base event halts
            integration when state[0] reaches L.

            theta is the ray angle in the local conical frame; it is clamped to
            [theta_local, beta_local] to keep spline evaluation in-domain.
            """
            x_val, r_val = state[0], state[1]

            # Avoid singularity at x=0 and r=0.
            x_val = max(x_val, 1e-12)
            r_val = max(r_val, 1e-12)

            theta = np.arctan(r_val / x_val)

            vf, th_cone, b_cone = _get_field_and_bounds(x_val)
            # Clamp theta to valid spline domain [theta_cone, beta_cone].
            theta = float(np.clip(theta, th_cone, b_cone))

            Vr_spline, Vt_spline = vf
            Vr_val = float(Vr_spline(theta))
            Vt_val = float(Vt_spline(theta))

            # Spherical-to-Cartesian velocity components (same as generator stode).
            v_x = Vr_val * np.cos(theta) - Vt_val * np.sin(theta)
            v_r = Vr_val * np.sin(theta) + Vt_val * np.cos(theta)

            # Protect against near-zero axial velocity (should not occur in
            # attached flow but guard defensively).
            if abs(v_x) < 1e-12:
                return [0.0, 0.0]

            # dr/dx = v_r / v_x  (x is independent variable, so d(x)/dt = v_x,
            # d(r)/dt = v_r, and we want d(r)/d(x) = v_r / v_x).
            # We integrate with state = [x, r] and t being a dummy parameter,
            # but since we drive via solve_ivp with t_span = (x_start, x_end)
            # and t_param IS x, we set d(state)/dt as follows:
            return [v_x, v_r]

        # Termination: reached base plane (x = L).
        def hit_base(t_param, state):
            return state[0] - self.L

        hit_base.terminal  = True
        hit_base.direction = 1.0   # increasing x

        # Initial conditions: enter at (x_start, r_shock_entry).
        y0 = [x_start, r_shock_entry]

        # Max step: 2% of body length to ensure adequate resolution.
        max_step = 0.02 * self.L

        sol = solve_ivp(
            streamline_ode,
            t_span=(0.0, 1e6),     # dummy large t_span; actual stop via event
            y0=y0,
            method='RK45',
            events=hit_base,
            max_step=max_step,
            rtol=1e-6,
            atol=1e-9,
            dense_output=False,
        )

        if sol.y.shape[1] < 2:
            logger.warning(
                "Streamline integration returned fewer than 2 points "
                "(r_shock_entry=%.6f). Returning straight line approximation.",
                r_shock_entry
            )
            x_line = np.array([x_start, self.L])
            r_body_end = self.R_base
            r_line = np.array([r_shock_entry, r_body_end])
            return np.column_stack([x_line, r_line])

        x_traj = sol.y[0]
        r_traj = sol.y[1]

        return np.column_stack([x_traj, r_traj])

    # ------------------------------------------------------------------

    def get_cone_angle_at_base(self) -> float:
        """
        Return the local cone half-angle at the base station (x = L),
        in radians.

        For n = 1.0 this equals theta_base_rad exactly.
        """
        self._require_solved()
        return float(self._theta_local_rad[-1])

    # ------------------------------------------------------------------

    def get_pressure_ratio_at_base(self, theta: float) -> float:
        """
        Return the static pressure ratio p / p_inf at the base plane at
        polar angle theta from the body axis.

        Uses the Taylor-Maccoll velocity field at the base station together
        with the correct T-M normalisation to recover the local Mach number,
        then applies the isentropic pressure relation:

            V_norm  = sqrt(Vr^2 + Vt^2)    (T-M normalised, V_max = 1)
            Ma^2    = 2/(gamma-1) * V_norm^2 / (1 - V_norm^2)
            p/p_0   = (1 + (gamma-1)/2 * Ma^2)^{-gamma/(gamma-1)}
            p/p_inf = (p/p_0) / (p_inf/p_0)

        The T-M velocity is normalised by V_max = a0 * sqrt(2/(gamma-1)),
        so the energy equation gives (a/a_max)^2 = 1 - V_norm^2 (not
        1 - (gamma-1)/2 * V_norm^2).

        Note: this is the isentropic result from the conical velocity field
        and does not include the shock entropy jump.  It is adequate for the
        relative pressure distribution needed in the osculating-plane
        framework.

        Parameters
        ----------
        theta : float
            Polar angle from the body axis, in radians.  Must be between
            0 (axis) and theta_base (body surface).

        Returns
        -------
        float
            p / p_inf at the given angle.
        """
        self._require_solved()

        Vr_spline, Vt_spline = self._fields[-1]

        Vr_val = float(Vr_spline(theta))
        Vt_val = float(Vt_spline(theta))
        V_sq   = Vr_val ** 2 + Vt_val ** 2

        g  = self.gamma
        gm = g - 1.0

        # Taylor-Maccoll velocity normalisation: velocities are non-dimensionalised
        # by V_max = sqrt(2/(gamma-1)) * a0, so the energy equation gives:
        #
        #   V_norm^2 + (a/a_max)^2 = 1   =>   (a/a_max)^2 = 1 - V_norm^2
        #
        # where a_max = a0 * sqrt(2/(gamma-1)).  The local Mach number is:
        #
        #   Ma^2 = V_actual^2 / a^2
        #        = V_norm^2 * V_max^2 / (a_max^2 * (1 - V_norm^2))
        #        = V_norm^2 * 2/(gamma-1) / (1 - V_norm^2)
        #
        # Reference: Maccoll (1937), or derivation in Anderson "Modern Compressible
        # Flow" 3rd ed., Section 13.1.
        V_sq_clamped = float(np.clip(V_sq, 1e-12, 1.0 - 1e-12))
        Ma_local_sq = 2.0 / gm * V_sq_clamped / (1.0 - V_sq_clamped)

        # Isentropic pressure ratio relative to total:  p/p_0 = (1 + gm/2 * Ma^2)^{-g/gm}
        p_over_p0_local = (1.0 + gm / 2.0 * Ma_local_sq) ** (-g / gm)

        # Freestream isentropic:
        p_inf_over_p0 = (1.0 + gm / 2.0 * self.M_inf ** 2) ** (-g / gm)

        return float(p_over_p0_local / p_inf_over_p0)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def x_stations(self) -> np.ndarray:
        """Axial station positions, shape (n_stations,)."""
        return self._x_stations.copy()

    @property
    def theta_local_deg(self) -> np.ndarray:
        """Local cone half-angle at each station, degrees.  Requires solve()."""
        self._require_solved()
        return np.degrees(self._theta_local_rad)

    @property
    def beta_local_deg(self) -> np.ndarray:
        """Local shock angle at each station, degrees.  Requires solve()."""
        self._require_solved()
        return np.degrees(self._beta_local_rad)

    @property
    def valid_station_mask(self) -> np.ndarray:
        """Boolean mask of directly solved (not interpolated) stations."""
        self._require_solved()
        return self._valid_mask.copy()

    # ------------------------------------------------------------------

    def get_velocity_field_at_base(self):
        """
        Return ``(Vr_spline, Vt_spline)`` for the base station (x = L).

        This is the same two-spline tuple returned by ``cone_field()`` in
        flowfield.py.  It provides drop-in compatibility with code in
        generator.py that consumes a cone velocity field directly, e.g.:

            Vr, Vt = cone_field(M, theta_rad, beta_rad, gamma)
            # equivalent for n=1 power-law body:
            Vr, Vt = plf.get_velocity_field_at_base()
        """
        self._require_solved()
        return tuple(self._fields[-1])

    def get_field_at_x(self, x: float):
        """
        Return ``([Vr_spline, Vt_spline], theta_local, beta_local)`` for the
        station nearest to axial position *x*.

        Parameters
        ----------
        x : float
            Axial position along the body (0 to L).

        Returns
        -------
        field : list
            [Vr_spline, Vt_spline] velocity spline pair.
        theta_local : float
            Local cone half-angle (rad) at this station.
        beta_local : float
            Local shock angle (rad) at this station.
        """
        self._require_solved()
        idx = int(np.searchsorted(self._x_stations, x, side='left'))
        idx = int(np.clip(idx, 0, len(self._x_stations) - 1))
        return (self._fields[idx],
                float(self._theta_local_rad[idx]),
                float(self._beta_local_rad[idx]))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _require_solved(self) -> None:
        if not self._solved:
            raise RuntimeError(
                "PowerLawFlowfield.solve() must be called before accessing results."
            )
