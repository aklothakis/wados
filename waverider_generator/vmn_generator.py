"""
vmn_generator.py
================
Variable Mach Number (VMN) waverider generator.

Implements the method of Li et al. (2018):
"Design and Aerodynamic Performance Analysis of a Variable Mach Number
Waverider for a Wide-Speed-Range Vehicle"

Key concept
-----------
Instead of a single design Mach number the design Mach varies spanwise from
Ma_min to Ma_max.  Each spanwise station j is assigned Ma_j and the
corresponding cone angle delta_j is found via Taylor-Maccoll at a shared shock
angle beta.  The shock surface is a single circular cone at angle beta.

Paper coordinate system (used internally during generation)
-----------------------------------------------------------
    z  — streamwise, cone apex at origin, base plane at z = L0
    x  — cross-plane vertical  (x > 0 is the flat "upper" side)
    y  — cross-plane spanwise  (y = 0 is symmetry plane, y > 0 outboard)
    Shock surface: x^2 + y^2 = (z * tan(beta))^2

GUI / output coordinate system
-------------------------------
    X  — streamwise  (nose at X=0, base at X=length)
    Y  — vertical    (Y=0 at top/upper surface level)
    Z  — spanwise    (Z=0 at symmetry plane)

Coordinate mapping from paper to GUI:
    GUI X  =  paper z  (re-scaled so vehicle length = self.length)
    GUI Y  = -paper x  (paper x points "up" toward freestream; GUI Y is "down")
    GUI Z  =  paper y  (spanwise, identical sign convention)

Surface data structures (compatible with to_CAD() in cad_export.py)
--------------------------------------------------------------------
    self.upper_surface_streams : list of ndarray (n_streamwise, 3)
        One stream per spanwise station.  Stream goes from LE (j=0) to TE (j=-1).
        Represents the freestream-aligned upper surface (straight line from
        UE point back along -freestream to the leading edge).
    self.lower_surface_streams : list of ndarray (n_streamwise, 3)
        One stream per spanwise station.  Streamline traced through the
        conical T-M flowfield from LE to base plane.
    self.leading_edge : ndarray (N, 3)
        Leading edge points in GUI coordinates.
    self.length : float
        Physical vehicle length [m].
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

try:
    from waverider_generator.flowfield import cone_field, cone_angle as _cone_angle_fn
except ImportError:
    # Standalone fallback — mirrors flowfield.py
    from scipy.interpolate import UnivariateSpline as _US

    def Taylor_Maccoll(t, x, gamma):
        A = (gamma - 1.0) / 2.0 * (1.0 - x[0]**2 - x[1]**2)
        dxdt = np.zeros(2)
        dxdt[0] = x[1]
        dxdt[1] = (x[1]*x[0]*x[1] - A*(2.0*x[0] + x[1]/np.tan(t))) / (A - x[1]**2)
        return dxdt

    def cone_field(Mach, theta_rad, beta_rad, gamma):
        d = np.arctan(2.0/np.tan(beta_rad) *
                      (Mach**2*np.sin(beta_rad)**2 - 1.0) /
                      (Mach**2*(gamma + np.cos(2*beta_rad)) + 2.0))
        Ma2 = (1.0/np.sin(beta_rad - d) *
               np.sqrt((1.0 + (gamma-1.0)/2.0*Mach**2*np.sin(beta_rad)**2) /
                       (gamma*Mach**2*np.sin(beta_rad)**2 - (gamma-1.0)/2.0)))
        V = 1.0 / np.sqrt(2.0/((gamma-1.0)*Ma2**2) + 1.0)
        Vr0 = V*np.cos(beta_rad - d)
        Vt0 = -(V*np.sin(beta_rad - d))
        sol = solve_ivp(Taylor_Maccoll, (beta_rad, theta_rad), [Vr0, Vt0], args=(gamma,))
        Vrf = _US(sol.t[::-1], sol.y[0, ::-1], k=min(3, sol.t.size-1))
        Vtf = _US(sol.t[::-1], sol.y[1, ::-1], k=min(3, sol.t.size-1))
        return [Vrf, Vtf]

    def _cone_angle_fn(Mach, shock_angle_deg, gamma):
        from scipy.integrate import solve_ivp
        beta_rad = np.radians(shock_angle_deg)
        d = np.arctan(2.0/np.tan(beta_rad) *
                      (Mach**2*np.sin(beta_rad)**2 - 1.0) /
                      (Mach**2*(gamma + np.cos(2*beta_rad)) + 2.0))
        Ma2 = (1.0/np.sin(beta_rad - d) *
               np.sqrt((1.0 + (gamma-1.0)/2.0*Mach**2*np.sin(beta_rad)**2) /
                       (gamma*Mach**2*np.sin(beta_rad)**2 - (gamma-1.0)/2.0)))
        V = 1.0 / np.sqrt(2.0/((gamma-1.0)*Ma2**2) + 1.0)
        Vr0 = V*np.cos(beta_rad - d)
        Vt0 = -(V*np.sin(beta_rad - d))
        def ev(t, y, gamma): return y[1]
        ev.terminal = True
        sol = solve_ivp(Taylor_Maccoll, (beta_rad, 0.0), [Vr0, Vt0],
                        events=ev, args=(gamma,))
        return float(np.degrees(sol.t_events[0][0]))


# =========================================================================== #
#  VMNWaverider class                                                          #
# =========================================================================== #

class VMNWaverider:
    """
    Variable Mach Number waverider (Li et al. 2018).

    Parameters
    ----------
    Ma_min : float
        Minimum design Mach number (at the spanwise tip when
        direction='decreasing', at the centre when direction='increasing').
    Ma_max : float
        Maximum design Mach number.
    beta_deg : float
        Shared shock angle [degrees] for all spanwise stations.
    L0 : float
        Normalised base-plane distance used during generation (paper parameter).
        Physical dimensions are L0 * length.  Default 1.0.
    S : float
        Conic curve shift parameter.  Controls how far inside the shock the UE
        curve sits.  Default 0.4 (paper value for Case 3).
    A0 : float
        Conic curve coefficient in x_ue = A0*y_ue^2 + R0.
        Positive value (paper Table 1: A0 = 1.7233). The UE starts at x=R0
        on the symmetry plane and curves outward (x increases with y²) toward
        the shock boundary. The tip F is where (A0*y²+R0)²+y²=R².
    length : float
        Physical vehicle length [m].  Scales the normalised geometry. Default 1.0.
    direction : str
        'decreasing' (default) — Ma decreases from centre (Ma_max) to tip
        (Ma_min); corresponds to Li et al. Case 3, which yields higher volume.
        'increasing' — Ma increases from centre (Ma_min) to tip (Ma_max).
    n_points : int
        Number of spanwise stations (half-span).  Default 30.
    n_streamwise : int
        Number of points along each streamline (LE → TE).  Default 20.
    gamma : float
        Ratio of specific heats.  Default 1.4.

    Attributes set after construction
    ----------------------------------
    upper_surface_streams : list of ndarray (n_streamwise, 3)
    lower_surface_streams : list of ndarray (n_streamwise, 3)
    leading_edge          : ndarray (n_points, 3)
    length                : float
    beta_deg              : float
    Ma_min, Ma_max        : float
    """

    def __init__(
        self,
        Ma_min: float,
        Ma_max: float,
        beta_deg: float,
        L0: float = 1.0,
        S: float = 0.4,
        A0: float = 1.7233,
        length: float = 1.0,
        direction: str = 'decreasing',
        n_points: int = 30,
        n_streamwise: int = 20,
        gamma: float = 1.4,
    ):
        # ---- validate ----
        if Ma_min <= 1.0:
            raise ValueError("Ma_min must be greater than 1.0 (supersonic).")
        if Ma_max < Ma_min:
            raise ValueError("Ma_max must be >= Ma_min.")
        if not (0 < beta_deg < 90):
            raise ValueError("beta_deg must be between 0 and 90 degrees.")
        if L0 <= 0:
            raise ValueError("L0 must be positive.")
        if length <= 0:
            raise ValueError("length must be positive.")
        if n_points < 4:
            raise ValueError("n_points must be at least 4.")
        if n_streamwise < 4:
            raise ValueError("n_streamwise must be at least 4.")
        if direction not in ('decreasing', 'increasing'):
            raise ValueError("direction must be 'decreasing' or 'increasing'.")

        # ---- store parameters ----
        self.Ma_min = float(Ma_min)
        self.Ma_max = float(Ma_max)
        self.beta_deg = float(beta_deg)
        self.beta_rad = np.radians(self.beta_deg)
        self.L0 = float(L0)
        self.S = float(S)
        self.A0 = float(A0)
        self.length = float(length)
        self.direction = direction
        self.n_points = int(n_points)
        self.n_streamwise = int(n_streamwise)
        self.gamma = float(gamma)

        # Normalised R0 (paper Eq. 1)
        self.R0 = self.S * self.L0 * np.tan(self.beta_rad)

        # Shock radius at base plane
        self.R_shock = self.L0 * np.tan(self.beta_rad)

        # Output containers
        self.upper_surface_streams: list = []
        self.lower_surface_streams: list = []
        self.leading_edge = np.zeros((self.n_points, 3))

        # ---- generate ----
        self._build()

    # ===================================================================== #
    #  Private helpers                                                        #
    # ===================================================================== #

    def _mach_at_station(self, j: int) -> float:
        """
        Return the design Mach number at spanwise station j (0 = centre, N-1 = tip).

        Equation 5 of Li et al. (2018), adjusted for direction:
            direction='decreasing' : centre gets Ma_max, tip gets Ma_min
            direction='increasing' : centre gets Ma_min, tip gets Ma_max
        """
        N = self.n_points - 1  # number of intervals
        frac = j / N if N > 0 else 0.0
        if self.direction == 'decreasing':
            # j=0 -> Ma_max, j=N -> Ma_min
            return self.Ma_max - (self.Ma_max - self.Ma_min) * frac
        else:
            # j=0 -> Ma_min, j=N -> Ma_max
            return self.Ma_min + (self.Ma_max - self.Ma_min) * frac

    def _compute_cone_angles(self) -> np.ndarray:
        """
        Compute the cone half-angle delta_j [deg] for each spanwise station j
        using Taylor-Maccoll at shock angle beta and local Mach Ma_j.

        Returns ndarray of shape (n_points,).
        """
        deltas = np.zeros(self.n_points)
        for j in range(self.n_points):
            Ma_j = self._mach_at_station(j)
            try:
                deltas[j] = _cone_angle_fn(Ma_j, self.beta_deg, self.gamma)
            except Exception as exc:
                raise RuntimeError(
                    f"Taylor-Maccoll failed for station j={j}, "
                    f"Ma={Ma_j:.4f}, beta={self.beta_deg:.4f} deg: {exc}"
                ) from exc
        return deltas

    def _ue_curve(self) -> np.ndarray:
        """
        Build the Upper Edge (UE) conic curve on the base plane z = L0.

        Paper Eq. 1:  x_ue = A0 * y_ue^2 + R0

        A0 is POSITIVE (Table 1: A0 = 1.7233).  The curve starts at
        (x, y) = (R0, 0) on the symmetry plane and curves outward (x increases
        with y²).  The tip F is where the UE meets the shock circle:
            (A0*y² + R0)² + y² = R²
        This is a quartic in y; we solve it numerically via bisection.

        Returns
        -------
        pts : ndarray (n_points, 2)
            Columns are [x_ue, y_ue] in paper base-plane coordinates.
        """
        if self.A0 <= 0:
            raise ValueError(
                "A0 must be positive (paper Table 1 value is 1.7233). "
                f"Got A0={self.A0:.6f}."
            )
        if self.R0 <= 0:
            raise ValueError(
                "R0 must be positive (S and beta must give a positive shift). "
                f"Got R0={self.R0:.6f}."
            )

        R = self.R_shock  # shock radius at base plane

        # Sanity: the centre point A must lie inside the shock
        if self.R0 >= R:
            raise ValueError(
                f"R0={self.R0:.6f} >= R_shock={R:.6f}. "
                "S must be less than 1.0 so that R0 < R."
            )

        # Find y_max where (A0*y² + R0)² + y² = R²
        # Define f(y) = (A0*y² + R0)² + y² - R²
        # f(0)  = R0² - R² < 0  (since R0 < R)
        # f(y)  grows without bound for large y (A0 > 0)
        # There is exactly one positive root.
        def f(y):
            x = self.A0 * y**2 + self.R0
            return x**2 + y**2 - R**2

        # Bracket: find an upper bound where f > 0
        y_hi = R  # safe upper bound (at y=R, x=A0*R²+R0 > R0 > 0, so x²+y²>R²)
        y_lo = 0.0
        # Bisection
        for _ in range(60):
            y_mid = 0.5 * (y_lo + y_hi)
            if f(y_mid) < 0:
                y_lo = y_mid
            else:
                y_hi = y_mid
        y_max = 0.5 * (y_lo + y_hi)

        # Sample n_points along y from 0 to y_max
        y_vals = np.linspace(0.0, y_max, self.n_points)
        x_vals = self.A0 * y_vals**2 + self.R0

        return np.column_stack([x_vals, y_vals])

    def _leading_edge_point(self, x_ue: float, y_ue: float):
        """
        Project the UE point (x_ue, y_ue, L0) upstream along -freestream
        (i.e., in the -z direction) until it hits the shock cone.

        Shock cone:  x^2 + y^2 = z^2 * tan^2(beta)
        Along the ray (x_ue, y_ue, z) the radial distance eta = sqrt(x_ue^2 + y_ue^2)
        is constant, so:
            eta = z_le * tan(beta)  ->  z_le = eta / tan(beta)

        Returns (x_ue, y_ue, z_le) in paper coordinates.
        """
        eta = np.sqrt(x_ue**2 + y_ue**2)
        z_le = eta / np.tan(self.beta_rad)
        return x_ue, y_ue, z_le

    def _trace_lower_streamline(
        self,
        x_ue: float,
        y_ue: float,
        z_le: float,
        delta_deg: float,
        Ma_j: float,
    ) -> np.ndarray:
        """
        Trace a T-M streamline from LE point to base plane for one station.

        The streamline lives in a cone-local 2D coordinate system (x_loc, eta_loc):
            x_loc  — axial distance from cone apex (along cone axis = freestream)
            eta_loc — radial distance from cone axis in the osculating plane

        At the LE:
            eta_le = sqrt(x_ue^2 + y_ue^2)   (paper-plane radial at z = z_le)
            x_le_local = z_le                  (the cone axis IS the freestream z)

        The ODE is:
            dx_loc/dt = Vr(theta)*cos(theta) - Vt(theta)*sin(theta)
            deta_loc/dt = Vr(theta)*sin(theta) + Vt(theta)*cos(theta)
            theta = arctan(eta_loc / x_loc)

        Integration terminates when x_loc reaches L0 (base plane).

        Parameters
        ----------
        x_ue, y_ue : float
            UE point in paper base-plane coordinates.
        z_le : float
            Streamwise (paper z) position of the LE point.
        delta_deg : float
            Cone half-angle [deg] for this station.
        Ma_j : float
            Design Mach number for this station.

        Returns
        -------
        stream_paper : ndarray (n_streamwise, 3)
            Streamline in paper coordinates (x, y, z) sampled at n_streamwise
            uniformly-spaced z stations from z_le to L0.
        """
        delta_rad = np.radians(delta_deg)

        # Compute T-M splines for this station
        Vr_fn, Vt_fn = cone_field(Ma_j, delta_rad, self.beta_rad, self.gamma)

        # Cone-local initial conditions
        eta_le = np.sqrt(x_ue**2 + y_ue**2)
        x_le_local = z_le          # paper z = cone axis position

        # Azimuthal angle in the base plane: direction from cone axis to UE point
        phi = np.arctan2(y_ue, x_ue)   # angle in (x, y) base plane

        def stode(t, state):
            x_loc, eta_loc = state
            if x_loc < 1e-12 or eta_loc < 1e-12:
                return [0.0, 0.0]
            th = np.arctan(eta_loc / x_loc)
            # Clip theta to valid T-M range [delta, beta]
            # Below delta = inside body (T-M blows up)
            # Above beta = outside shock (unphysical)
            th = np.clip(th, delta_rad + 1e-6,
                         self.beta_rad - 1e-6)
            vr = float(Vr_fn(th))
            vt = float(Vt_fn(th))
            dxdt  = vr * np.cos(th) - np.sin(th) * vt
            detat = vr * np.sin(th) + np.cos(th) * vt
            return [dxdt, detat]

        # Terminate when axial position reaches base plane (x_loc = L0)
        def base_event(t, state):
            return state[0] - self.L0
        base_event.terminal = True
        base_event.direction = 1.0   # only trigger when x_loc increases through L0

        # Initial state
        state0 = [x_le_local, eta_le]

        # Bound integration time: velocities are O(1),
        # distance is L0 - x_le_local, so t ~ distance
        t_max = 20.0 * max(self.L0 - x_le_local, 0.1)

        try:
            sol = solve_ivp(
                stode,
                (0.0, t_max),
                state0,
                events=base_event,
                method='RK45',
                rtol=1e-4,
                atol=1e-7,
            )
        except Exception:
            # Degenerate — return straight line from LE to base
            sol = None

        if sol is not None and sol.y.shape[1] >= 2:
            x_arr  = sol.y[0]
            eta_arr = sol.y[1]
        else:
            # Fallback: straight line in cone-local coords
            x_arr   = np.array([x_le_local, self.L0])
            eta_arr = np.array([eta_le, eta_le])

        # Resample to n_streamwise points at uniform z (= x_loc) stations
        # (ensures consistent grid for surface lofting)
        z_target = np.linspace(x_le_local, self.L0, self.n_streamwise)

        # Interpolate eta as function of x_loc
        # Sort to ensure monotone x
        sort_idx = np.argsort(x_arr)
        x_sorted   = x_arr[sort_idx]
        eta_sorted  = eta_arr[sort_idx]

        # Remove duplicate x values
        _, uniq = np.unique(x_sorted, return_index=True)
        x_u   = x_sorted[uniq]
        eta_u = eta_sorted[uniq]

        if x_u.size < 2:
            eta_interp = np.full(self.n_streamwise, eta_le)
        else:
            f_eta = interp1d(x_u, eta_u, kind='linear',
                             bounds_error=False,
                             fill_value=(eta_u[0], eta_u[-1]))
            eta_interp = f_eta(z_target)

        # Ensure non-negative eta
        eta_interp = np.maximum(eta_interp, 0.0)

        # Convert back to paper (x, y, z) coordinates using the azimuthal angle phi
        # (phi is fixed along a streamline: the azimuth does not rotate for axisymmetric flow)
        x_paper = eta_interp * np.cos(phi)
        y_paper = eta_interp * np.sin(phi)
        z_paper = z_target

        return np.column_stack([x_paper, y_paper, z_paper])

    def _paper_to_gui(self, stream_paper: np.ndarray) -> np.ndarray:
        """
        Transform a point array from paper coordinates to GUI coordinates.

        Paper -> GUI mapping:
            GUI X (streamwise) = (paper_z - z_nose) * scale
                where z_nose = min(z_le) over all LE stations (the centreline
                LE point at the symmetry plane), and
                scale = length / (L0 - z_nose) so that nose maps to X=0 and
                the base plane (paper z=L0) maps to X=length.
            GUI Y (vertical)   = -paper_x   (paper x is "up", GUI Y is inverted)
            GUI Z (spanwise)   = paper_y

        self._z_nose and self._z_scale must be set in _build() before any
        calls to this method are made.

        Parameters
        ----------
        stream_paper : ndarray (..., 3)
            Points with columns [x_paper, y_paper, z_paper].

        Returns
        -------
        ndarray (..., 3) with columns [X_gui, Y_gui, Z_gui].
        """
        gui = np.empty_like(stream_paper)
        gui[..., 0] = (stream_paper[..., 2] - self._z_nose) * self._z_scale
        gui[..., 1] = -stream_paper[..., 0]
        gui[..., 2] = stream_paper[..., 1]
        return gui

    @staticmethod
    def _smooth_streams(streams: list, n_passes: int = 5) -> list:
        """
        Apply spanwise smoothing across a list of streamlines.

        Uses a 5-point weighted kernel [1, 4, 6, 4, 1] / 16 for n_passes passes.
        Interior stations only — symmetry-plane (i=0) and wingtip (i=-1) are pinned.

        Parameters
        ----------
        streams : list of ndarray (n_pts, 3)
        n_passes : int

        Returns
        -------
        list of ndarray (n_pts, 3)  (deep copies, originals unchanged)
        """
        if len(streams) < 5:
            return [s.copy() for s in streams]

        result = [s.copy() for s in streams]
        N = len(result)
        kernel = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0

        for _ in range(n_passes):
            smoothed = [result[i].copy() for i in range(N)]
            for i in range(2, N - 2):
                smoothed[i] = (
                    kernel[0] * result[i-2]
                    + kernel[1] * result[i-1]
                    + kernel[2] * result[i]
                    + kernel[3] * result[i+1]
                    + kernel[4] * result[i+2]
                )
            # Pin symmetry plane and wingtip
            smoothed[0]    = result[0].copy()
            smoothed[1]    = result[1].copy()
            smoothed[-1]   = result[-1].copy()
            smoothed[-2]   = result[-2].copy()
            result = smoothed

        return result

    # ===================================================================== #
    #  Main build sequence                                                    #
    # ===================================================================== #

    def _build(self):
        """
        Execute the full VMN waverider construction sequence.

        Step 1 — Upper Edge (UE) conic curve on base plane
        Step 2 — Leading edge by upstream projection to shock cone
        Step 3 — Cone angles for each station via T-M
        Step 4 — Trace lower streamlines (LE to base plane)
        Step 5 — Build upper surface streams (UE to LE, freestream-aligned)
        Step 6 — Transform to GUI coordinates
        Step 7 — Spanwise smoothing
        """

        # ---- Step 1: UE conic curve on base plane z = L0 ----------------
        ue_pts = self._ue_curve()   # shape (n_points, 2): [x_ue, y_ue]

        # ---- Step 2: Leading edge points (project upstream to shock cone) --
        le_paper = np.zeros((self.n_points, 3))  # [x_ue, y_ue, z_le]
        for j in range(self.n_points):
            x_ue, y_ue = ue_pts[j]
            x_le, y_le, z_le = self._leading_edge_point(x_ue, y_ue)
            le_paper[j] = [x_le, y_le, z_le]

        # ---- Step 3: Cone angles -----------------------------------------
        print("[VMN] Computing cone angles via Taylor-Maccoll...")
        delta_arr = self._compute_cone_angles()
        self.cone_angles_deg = delta_arr.copy()
        self._mach_per_station = np.array(
            [self._mach_at_station(j) for j in range(self.n_points)])

        # ---- Step 4: Trace lower streamlines -----------------------------
        print("[VMN] Tracing lower surface streamlines...")
        lower_paper = []
        for j in range(self.n_points):
            x_ue, y_ue = ue_pts[j]
            z_le = le_paper[j, 2]
            delta_j = delta_arr[j]
            Ma_j = self._mach_at_station(j)

            stream = self._trace_lower_streamline(
                x_ue, y_ue, z_le, delta_j, Ma_j
            )
            lower_paper.append(stream)

        # ---- Step 5: Upper surface (freestream-aligned, LE to UE) --------
        # The upper surface is bounded by the freestream.  For each station j,
        # the stream is a straight line in the -z (upstream) direction from
        # the LE point back to the UE point (which is on the base plane z=L0).
        # We parameterise uniformly from LE (z = z_le) to UE (z = L0).
        upper_paper = []
        for j in range(self.n_points):
            x_ue, y_ue = ue_pts[j]
            z_le = le_paper[j, 2]

            # Both LE and UE share the same (x, y): the ray is along z only
            z_line = np.linspace(z_le, self.L0, self.n_streamwise)
            x_line = np.full(self.n_streamwise, x_ue)
            y_line = np.full(self.n_streamwise, y_ue)
            stream = np.column_stack([x_line, y_line, z_line])
            upper_paper.append(stream)

        # ---- Step 6: Transform to GUI coordinates ------------------------
        # Compute the z_nose (minimum z_le across all stations = symmetry-plane LE).
        # At the symmetry plane (j=0), z_le = R0/tan(beta) = S*L0.
        # The vehicle nose is at this z, and the base is at z=L0.
        self._z_nose = float(np.min(le_paper[:, 2]))
        z_span = self.L0 - self._z_nose
        if z_span < 1e-12:
            raise RuntimeError(
                "Vehicle has zero length (z_nose == L0). "
                "Check that S < 1.0 and A0 > 0."
            )
        self._z_scale = self.length / z_span

        self.leading_edge = self._paper_to_gui(le_paper)

        upper_gui = [self._paper_to_gui(s) for s in upper_paper]
        lower_gui = [self._paper_to_gui(s) for s in lower_paper]

        # Enforce that LE points match exactly between upper and lower streams
        for j in range(self.n_points):
            le_j = self.leading_edge[j].copy()
            upper_gui[j][0] = le_j
            lower_gui[j][0] = le_j

        # ---- Step 7: Spanwise smoothing ----------------------------------
        upper_gui = self._smooth_streams(upper_gui, n_passes=5)
        lower_gui = self._smooth_streams(lower_gui, n_passes=5)

        # Re-pin LE after smoothing (smoothing must not move the leading edge)
        for j in range(self.n_points):
            le_j = self.leading_edge[j].copy()
            upper_gui[j][0] = le_j
            lower_gui[j][0] = le_j

        self.upper_surface_streams = upper_gui
        self.lower_surface_streams = lower_gui

        print(
            f"[VMN] Done. {self.n_points} stations, "
            f"Ma {self.Ma_min:.1f}-{self.Ma_max:.1f}, "
            f"beta={self.beta_deg:.1f} deg, "
            f"length={self.length:.3f} m"
        )

    # ===================================================================== #
    #  Mesh export helpers                                                    #
    # ===================================================================== #

    def get_mesh(self):
        """
        Build a triangulated mesh for STL / TRI export.

        Returns
        -------
        vertices : ndarray (n_verts, 3)
        triangles : ndarray (n_tri, 3)  — 0-based vertex indices
        """
        us = self.upper_surface_streams
        ls = self.lower_surface_streams

        n_span   = len(us)          # number of spanwise stations
        n_stream = us[0].shape[0]   # points per streamline

        # Collect vertices: upper surface then lower surface
        verts_upper = np.vstack([s for s in us])   # (n_span * n_stream, 3)
        verts_lower = np.vstack([s for s in ls])   # (n_span * n_stream, 3)

        vertices = np.vstack([verts_upper, verts_lower])
        lower_start = n_span * n_stream

        triangles = []

        # Upper surface quads (outward normal points away from body interior)
        for i in range(n_span - 1):
            for j in range(n_stream - 1):
                v00 = i * n_stream + j
                v01 = i * n_stream + j + 1
                v10 = (i + 1) * n_stream + j
                v11 = (i + 1) * n_stream + j + 1
                triangles.append([v00, v10, v01])
                triangles.append([v01, v10, v11])

        # Lower surface quads (reversed winding for inward normal on lower side)
        for i in range(n_span - 1):
            for j in range(n_stream - 1):
                v00 = lower_start + i * n_stream + j
                v01 = lower_start + i * n_stream + j + 1
                v10 = lower_start + (i + 1) * n_stream + j
                v11 = lower_start + (i + 1) * n_stream + j + 1
                triangles.append([v00, v01, v10])
                triangles.append([v01, v11, v10])

        # Leading edge cap: connect upper LE row to lower LE row
        for i in range(n_span - 1):
            u0 = i * n_stream
            u1 = (i + 1) * n_stream
            l0 = lower_start + i * n_stream
            l1 = lower_start + (i + 1) * n_stream
            triangles.append([u0, l0, u1])
            triangles.append([u1, l0, l1])

        # Base/trailing-edge cap
        for i in range(n_span - 1):
            u0 = i * n_stream + (n_stream - 1)
            u1 = (i + 1) * n_stream + (n_stream - 1)
            l0 = lower_start + i * n_stream + (n_stream - 1)
            l1 = lower_start + (i + 1) * n_stream + (n_stream - 1)
            triangles.append([u0, u1, l0])
            triangles.append([u1, l1, l0])

        # Symmetry-plane cap (i = 0): close the Z=0 face
        for j in range(n_stream - 1):
            u0 = 0 * n_stream + j
            u1 = 0 * n_stream + j + 1
            l0 = lower_start + 0 * n_stream + j
            l1 = lower_start + 0 * n_stream + j + 1
            triangles.append([u0, u1, l0])
            triangles.append([u1, l1, l0])

        # Wingtip cap (i = n_span - 1): close the tip face
        i_tip = n_span - 1
        for j in range(n_stream - 1):
            u0 = i_tip * n_stream + j
            u1 = i_tip * n_stream + j + 1
            l0 = lower_start + i_tip * n_stream + j
            l1 = lower_start + i_tip * n_stream + j + 1
            triangles.append([u0, l0, u1])
            triangles.append([u1, l0, l1])

        return vertices, np.array(triangles, dtype=int)

    def export_stl(self, filename: str):
        """
        Export the waverider mesh as an ASCII STL file.

        Parameters
        ----------
        filename : str
            Output file path (should end in .stl).
        """
        vertices, triangles = self.get_mesh()

        with open(filename, 'w') as f:
            f.write("solid vmn_waverider\n")
            for tri in triangles:
                v0 = vertices[tri[0]]
                v1 = vertices[tri[1]]
                v2 = vertices[tri[2]]
                e1 = v1 - v0
                e2 = v2 - v0
                normal = np.cross(e1, e2)
                nrm = np.linalg.norm(normal)
                if nrm > 1e-14:
                    normal /= nrm
                else:
                    normal = np.array([0.0, 0.0, 1.0])
                f.write(
                    f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n"
                )
                f.write("    outer loop\n")
                f.write(
                    f"      vertex {v0[0]:.8f} {v0[1]:.8f} {v0[2]:.8f}\n"
                )
                f.write(
                    f"      vertex {v1[0]:.8f} {v1[1]:.8f} {v1[2]:.8f}\n"
                )
                f.write(
                    f"      vertex {v2[0]:.8f} {v2[1]:.8f} {v2[2]:.8f}\n"
                )
                f.write("    endloop\n")
                f.write("  endfacet\n")
            f.write("endsolid vmn_waverider\n")

        print(f"[VMN] STL exported to {filename}")

    def export_tri(self, filename: str):
        """
        Export the waverider mesh in NASA Cart3D TRI format.

        Parameters
        ----------
        filename : str
            Output file path (should end in .tri).
        """
        vertices, triangles = self.get_mesh()
        n_v = len(vertices)
        n_t = len(triangles)

        with open(filename, 'w') as f:
            f.write(f"{n_v}\n")
            f.write(f"{n_t}\n")
            for v in vertices:
                f.write(f"{v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
            for tri in triangles:
                # TRI format uses 1-based vertex indices
                f.write(f"{tri[0]+1} {tri[1]+1} {tri[2]+1}\n")

        print(f"[VMN] TRI exported to {filename}")

    # ===================================================================== #
    #  CAD export via cad_export.to_CAD                                       #
    # ===================================================================== #

    def to_CAD(self, sides: str = 'both', export: bool = False,
               filename: str = 'vmn_waverider.step', **kwargs):
        """
        Export to STEP using B-spline grid surfaces.

        Uses _make_bspline_face + _sew_faces_to_solid directly
        (the interpPlate path in to_CAD doesn't handle VMN geometry).
        """
        from waverider_generator.cad_export import (
            _make_bspline_face, _sew_faces_to_solid)
        import cadquery as cq

        scale = kwargs.get('scale', 1.0)

        us = self.upper_surface_streams
        ls = self.lower_surface_streams

        # Scale streams
        us_s = [s * scale for s in us]
        ls_s = [s * scale for s in ls]

        # Build upper and lower B-spline faces
        upper_faces = _make_bspline_face(us_s)
        lower_faces = _make_bspline_face(ls_s)

        # Build back face from TE cross-section using B-spline grid
        te_upper = np.array([s[-1] for s in us_s])
        te_lower = np.array([s[-1] for s in ls_s])
        # Create a ruled surface: 2 rows (upper TE, lower TE)
        back_streams = [te_upper.reshape(1, -1, 3)[:, :, :].reshape(-1, 3),
                        te_lower.reshape(1, -1, 3)[:, :, :].reshape(-1, 3)]
        # Actually: use 2-row grid for back face
        back_grid = [te_upper, te_lower]
        try:
            back_faces = _make_bspline_face(back_grid)
        except Exception:
            back_faces = []

        # Build symmetry face (z=0 plane) — 2 rows
        sym_upper = us_s[0]
        sym_lower = ls_s[0]
        try:
            sym_faces = _make_bspline_face([sym_upper, sym_lower])
        except Exception:
            sym_faces = []

        # Sew into solid (use larger tolerance for VMN edge matching)
        all_faces = upper_faces + lower_faces + back_faces + sym_faces
        solid = _sew_faces_to_solid(all_faces, tolerance=1.0)

        # Mirror for right side
        if sides == 'both':
            mirrored = solid.mirror("XY")
            solid = solid.fuse(mirrored)

        if export:
            cq.exporters.export(
                cq.Workplane().add(solid), filename)
            print(f"[VMN] STEP exported to {filename}")

        return solid

