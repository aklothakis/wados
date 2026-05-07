"""Variable Mach Osculating Flowfield (VMOF) waverider generator.

Liu et al. (2019) 'Novel osculating flowfield methodology for
wide-speed range waverider vehicles across variable Mach number',
Acta Astronautica 162, pp. 160-167.

Key concept
-----------
The OC (osculating cone) generator provides all planform geometry: Bezier
shockwave curve, cone centres, leading edge, and upper surface.  The VMOF
method replaces the single-Mach lower surface streamline tracing with a
per-plane Taylor-Maccoll integration at a spatially varying Mach number
Ma(z).  Each osculating plane j has its own Mach number Ma_j and cone angle
delta_j (via T-M at the shared shock angle beta), giving a smoothly varying
lower surface that blends aerodynamics across a speed range.

Coordinate system (matches the GUI and generator.py)
----------------------------------------------------
    X  — streamwise (nose at X=0, base plane at X=length)
    Y  — vertical   (Y=0 at upper-surface datum, positive downward)
    Z  — spanwise   (Z=0 at symmetry plane, positive outboard)

Surface data structures
-----------------------
    upper_surface_streams : list of ndarray (n_streamwise, 3)
        Copied directly from the OC generator.  One stream per spanwise
        station (n_planes + 2 entries, including tip).  Compatible with
        upper_surface_streams in cad_export.py.
    lower_surface_streams : list of ndarray (n_streamwise, 3)
        Per-plane T-M streamlines, same list structure as upper.
    leading_edge : ndarray (n_planes + 2, 3)
        Leading edge points in GUI coordinates.  Identical to OC generator.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


class VMOFWaverider:
    """
    Variable Mach Osculating Flowfield waverider (Liu et al. 2019).

    Combines the OC generator's planform Bezier curves (X1-X4) with a
    spatially varying design Mach number per osculating plane.  Each plane
    receives its own cone angle from Taylor-Maccoll at the shared shock angle
    beta, giving a lower surface that accounts for varying local Ma.

    Parameters
    ----------
    M_inf_min : float
        Minimum design Mach number (tip when direction='decreasing').
    M_inf_max : float
        Maximum design Mach number (centre when direction='decreasing').
    beta_deg : float
        Shared shock angle [degrees] for all osculating planes.
    height : float
        Vehicle height [m] (Y-extent from upper surface to lower surface at
        the symmetry plane, passed directly to the OC generator).
    width : float
        Half-span [m].
    dp : list of float
        [X1, X2, X3, X4] OC planform parameters (same as generator.waverider).
    direction : str
        'decreasing' (default) — Ma decreases from centre (M_inf_max) to
        tip (M_inf_min).  Produces thicker tips.
        'increasing' — Ma increases from centre (M_inf_min) to tip
        (M_inf_max).
    n_planes : int
        Number of interior osculating planes (spanwise resolution, >= 10).
    n_streamwise : int
        Number of streamwise points per stream (>= 10).
    gamma : float
        Ratio of specific heats.  Default 1.4.

    Attributes (set after construction)
    ------------------------------------
    upper_surface_streams : list of ndarray (n_streamwise, 3)
    lower_surface_streams : list of ndarray (n_streamwise, 3)
    leading_edge : ndarray (n_planes + 2, 3)
    length : float
        Physical vehicle length [m].
    beta : float
        Shock angle [degrees].
    theta : float
        Deflection angle at M_inf_max [degrees].
    cone_angles_deg : ndarray (n_planes + 2,)
        Per-station cone half-angle [degrees].
    _mach_per_station : ndarray (n_planes + 2,)
        Per-station design Mach number.
    """

    def __init__(
        self,
        M_inf_min,
        M_inf_max,
        beta_deg,
        height,
        width,
        dp,
        direction='decreasing',
        n_planes=20,
        n_streamwise=20,
        gamma=1.4,
    ):
        # ------------------------------------------------------------------ #
        #  Validate and store parameters                                       #
        # ------------------------------------------------------------------ #
        if float(M_inf_min) <= 1.0:
            raise ValueError("M_inf_min must be greater than 1.0 (supersonic).")
        if float(M_inf_max) < float(M_inf_min):
            raise ValueError("M_inf_max must be >= M_inf_min.")
        if not (0 < float(beta_deg) < 90):
            raise ValueError("beta_deg must be between 0 and 90 degrees.")
        if float(height) <= 0:
            raise ValueError("height must be positive.")
        if float(width) <= 0:
            raise ValueError("width must be positive.")
        if not isinstance(dp, list) or len(dp) != 4:
            raise ValueError("dp must be a list of 4 float/int values [X1, X2, X3, X4].")
        if not isinstance(n_planes, int) or n_planes < 10:
            raise ValueError("n_planes must be an integer >= 10.")
        if not isinstance(n_streamwise, int) or n_streamwise < 10:
            raise ValueError("n_streamwise must be an integer >= 10.")
        if direction not in ('decreasing', 'increasing'):
            raise ValueError("direction must be 'decreasing' or 'increasing'.")

        self.M_inf_min = float(M_inf_min)
        self.M_inf_max = float(M_inf_max)
        self.beta_deg = float(beta_deg)
        self.height = float(height)
        self.width = float(width)
        self.gamma = float(gamma)
        self.direction = direction
        self.n_planes = n_planes
        self.n_streamwise = n_streamwise
        self.X1, self.X2, self.X3, self.X4 = [float(v) for v in dp]

        # ------------------------------------------------------------------ #
        #  Step 1: Instantiate OC generator for planform geometry              #
        #                                                                      #
        #  Use M_inf_max so that the OC shockwave/LE geometry is defined for  #
        #  the highest Mach (largest cone angle).  The upper surface and all  #
        #  planform Bezier curves come from this single OC instance.           #
        # ------------------------------------------------------------------ #
        from waverider_generator.generator import waverider as OCWaverider

        print(f"[VMOF] Building OC base geometry at M={self.M_inf_max:.3f}...")
        self._oc = OCWaverider(
            self.M_inf_max,
            self.beta_deg,
            self.height,
            self.width,
            dp,
            n_upper_surface=100,
            n_shockwave=100,
            n_planes=n_planes,
            n_streamwise=n_streamwise,
        )

        self.length = self._oc.length
        self.beta = self._oc.beta          # degrees
        self.theta = self._oc.theta        # deflection angle at M_inf_max, degrees

        # delta_streamwise controls max ODE step (default 5% of vehicle length)
        self._delta_streamwise = 0.05

        # ------------------------------------------------------------------ #
        #  Step 2: Copy LE and upper surface from OC                          #
        # ------------------------------------------------------------------ #
        self.leading_edge = self._oc.leading_edge.copy()
        self.upper_surface_streams = list(self._oc.upper_surface_streams)

        # ------------------------------------------------------------------ #
        #  Step 3: Mach distribution and per-plane cone angles                #
        # ------------------------------------------------------------------ #
        self._compute_mach_distribution()
        self._compute_cone_angles()

        # ------------------------------------------------------------------ #
        #  Step 4: Trace lower surface with per-plane T-M                     #
        # ------------------------------------------------------------------ #
        self.lower_surface_streams = []
        print(f"[VMOF] Tracing lower surface streamlines "
              f"(Ma {self.M_inf_min:.2f}-{self.M_inf_max:.2f})...")
        self._trace_lower_surface()

        # ------------------------------------------------------------------ #
        #  Step 5: Spanwise smoothing                                         #
        # ------------------------------------------------------------------ #
        self._smooth_surfaces()

        print(
            f"[VMOF] Done. {n_planes} interior planes, "
            f"Ma {self.M_inf_min:.2f}-{self.M_inf_max:.2f}, "
            f"beta={self.beta_deg:.1f} deg, "
            f"length={self.length:.4f} m, "
            f"cone angles {self.cone_angles_deg.min():.2f}-"
            f"{self.cone_angles_deg.max():.2f} deg."
        )

    # ====================================================================== #
    #  Mach distribution                                                       #
    # ====================================================================== #

    def _compute_mach_distribution(self):
        """
        Assign a design Mach number to every augmented spanwise station.

        Stations: [0 (nose/symmetry plane), 1..n_planes (interior), n_planes+1 (tip)].
        Uses a linear distribution in z normalised by half-span.

        direction='decreasing' : Ma(0) = M_inf_max, Ma(tip) = M_inf_min
        direction='increasing' : Ma(0) = M_inf_min, Ma(tip) = M_inf_max
        """
        oc = self._oc

        # Interior shockwave z-stations (shape: n_planes,)
        z_interior = oc.z_local_shockwave   # already excludes tip/symmetry

        # Augmented z including symmetry plane (0) and tip (width)
        z_all = np.concatenate([[0.0], z_interior, [self.width]])
        self._z_all = z_all          # shape: (n_planes + 2,)

        n_sp = len(z_all)            # n_planes + 2
        self._mach_per_station = np.zeros(n_sp)

        for i in range(n_sp):
            t = z_all[i] / max(self.width, 1e-10)
            t = np.clip(t, 0.0, 1.0)
            if self.direction == 'decreasing':
                self._mach_per_station[i] = (
                    self.M_inf_max * (1.0 - t) + self.M_inf_min * t)
            else:
                self._mach_per_station[i] = (
                    self.M_inf_min * (1.0 - t) + self.M_inf_max * t)

    # ====================================================================== #
    #  Per-plane cone angles via Taylor-Maccoll                               #
    # ====================================================================== #

    def _compute_cone_angles(self):
        """
        Find the cone half-angle delta_j [deg] for each augmented station j
        using Taylor-Maccoll at shock angle beta_deg and local Mach Ma_j.

        The cone angle is where the tangential velocity Vt reaches zero.
        Falls back to the flow deflection angle if integration fails.
        """
        from waverider_generator.flowfield import Taylor_Maccoll

        beta_rad = np.radians(self.beta_deg)
        n_sp = len(self._mach_per_station)
        self.cone_angles_deg = np.zeros(n_sp)

        def _vt_zero(t, y, gamma):
            return y[1]

        _vt_zero.terminal = True

        for j in range(n_sp):
            Ma_j = self._mach_per_station[j]
            g = self.gamma

            # Flow deflection angle from oblique shock relations
            sin_b = np.sin(beta_rad)
            cos_b = np.cos(beta_rad)
            d = np.arctan(
                2.0 / np.tan(beta_rad)
                * (Ma_j**2 * sin_b**2 - 1.0)
                / (Ma_j**2 * (g + np.cos(2.0 * beta_rad)) + 2.0)
            )

            # Post-shock Mach
            Ma2 = (
                1.0 / np.sin(beta_rad - d)
                * np.sqrt(
                    (1.0 + (g - 1.0) / 2.0 * Ma_j**2 * sin_b**2)
                    / (g * Ma_j**2 * sin_b**2 - (g - 1.0) / 2.0)
                )
            )

            # Post-shock velocity magnitude (normalised)
            V = 1.0 / np.sqrt(2.0 / ((g - 1.0) * Ma2**2) + 1.0)
            Vr0 = V * np.cos(beta_rad - d)
            Vt0 = -(V * np.sin(beta_rad - d))

            try:
                sol = solve_ivp(
                    Taylor_Maccoll,
                    (beta_rad, 0.0),
                    [Vr0, Vt0],
                    events=_vt_zero,
                    args=(self.gamma,),
                    dense_output=False,
                )
                if sol.t_events[0].size > 0:
                    self.cone_angles_deg[j] = float(
                        np.degrees(sol.t_events[0][0])
                    )
                else:
                    # Event not triggered — use deflection angle as fallback
                    self.cone_angles_deg[j] = float(np.degrees(d))
            except Exception:
                self.cone_angles_deg[j] = float(np.degrees(d))

    # ====================================================================== #
    #  Per-plane lower surface streamline tracing                              #
    # ====================================================================== #

    def _trace_lower_surface(self):
        """
        Trace lower surface streamlines with per-plane T-M velocity fields.

        Replicates the OC generator's Streamline_Tracing method but calls
        cone_field(Ma_j, delta_j, beta, gamma) independently for each
        osculating plane rather than once for a single shared cone angle.

        The OC generator's 2D cone-local coordinate system is used:
            x_loc  — axial position from cone apex (aligned with freestream)
            eta    — radial distance from cone apex in the osculating plane

        The coordinate transform at each plane uses:
            alpha  = arctan(m)  where m = dY_shockwave / dZ at station i
            GUI stream = [-eta*cos(alpha), eta*sin(alpha)] + cone_center offset
        """
        from waverider_generator.flowfield import cone_field
        from waverider_generator.generator import Euclidean_Distance

        oc = self._oc
        le = self.leading_edge

        beta_rad = np.radians(self.beta_deg)

        # Build augmented arrays of size (n_planes + 2) — same layout as OC generator
        # y_local_shockwave: local y-bar coordinate at each shockwave station
        y_sw = np.vstack([
            np.array([[0.0]]),
            oc.y_local_shockwave,
            np.array([[self.X2 * self.height]])
        ])  # shape: (n_planes + 2, 1)

        # z_local_shockwave: z coordinate at each shockwave station
        z_sw = np.vstack([
            np.array([[0.0]]),
            oc.z_local_shockwave[:, None],
            np.array([[self.width]])
        ])  # shape: (n_planes + 2, 1)

        # cone_centers: (x, y, z) of each cone apex in global coordinates
        cc = np.vstack([
            np.array([[0.0, 0.0, 0.0]]),
            oc.cone_centers,
            np.array([[self.length,
                        oc.Local_to_Global(self.X2 * self.height),
                        self.width]])
        ])  # shape: (n_planes + 2, 3)

        # local_intersections_us: (z, y_bar) intersection of each osculating plane
        # with the upper surface Bezier curve
        us_int = np.vstack([
            np.array([[0.0, self.height]]),
            oc.local_intersections_us,
            np.array([[self.width, self.X2 * self.height]])
        ])  # shape: (n_planes + 2, 2)

        for i, le_point in enumerate(le):

            # ---- Tip: degenerate — two coincident points ---- #
            if i == len(le) - 1:
                stream = np.vstack([le_point, le_point])
                self.lower_surface_streams.append(stream)
                continue

            Ma_i = self._mach_per_station[i]
            delta_i = np.radians(self.cone_angles_deg[i])
            theta_i_deg = float(np.degrees(np.arctan(
                2.0 / np.tan(beta_rad)
                * (Ma_i**2 * np.sin(beta_rad)**2 - 1.0)
                / (Ma_i**2 * (self.gamma + np.cos(2.0 * beta_rad)) + 2.0)
            )))

            # ---- Flat region (z <= X1*width) or X2==0: straight deflected line ---- #
            if z_sw[i, 0] <= self.X1 * self.width or self.X2 == 0:
                bottom_y = (le_point[1]
                            - np.tan(np.radians(theta_i_deg))
                            * (self.length - le_point[0]))
                x = np.linspace(le_point[0], self.length, self.n_streamwise)[:, None]
                y = np.linspace(le_point[1], bottom_y, self.n_streamwise)[:, None]
                z = np.full(y.shape, le_point[2])
                self.lower_surface_streams.append(np.column_stack([x, y, z]))
                continue

            # ---- Curved region: per-plane T-M streamline ---- #

            # T-M velocity splines for this station's Mach and cone angle
            Vr_fn, Vt_fn = cone_field(Ma_i, delta_i, beta_rad, self.gamma)

            # Radial distance from cone apex to the upper surface intersection
            # (this is eta at the leading edge, i.e. the initial condition)
            eta_le = Euclidean_Distance(
                us_int[i, 0],
                oc.Local_to_Global(us_int[i, 1]),
                cc[i, 2],
                cc[i, 1],
            )

            # Radial distance from cone apex to the shockwave at this station
            # (this sets the axial extent x_base = r / tan(beta), the "base plane"
            #  in cone-local coordinates)
            r = Euclidean_Distance(
                z_sw[i, 0],
                oc.Local_to_Global(y_sw[i, 0]),
                cc[i, 2],
                cc[i, 1],
            )

            # Shockwave first derivative dY/dZ at this z-station
            # alpha is the tilt angle of the osculating plane normal
            m_sw, _, _ = oc.Get_First_Derivative(z_sw[i, 0])
            alpha = np.arctan(m_sw)

            # Initial axial position in cone-local frame
            x_le = eta_le / np.tan(beta_rad)
            # Base plane in cone-local frame (integration terminates here)
            x_base = r / np.tan(beta_rad)

            # ODE: propagates (x_loc, eta) under T-M velocity field
            def stode(t, state, Vr=Vr_fn, Vt=Vt_fn,
                      d_lo=delta_i, b_hi=beta_rad):
                th = np.arctan(state[1] / max(state[0], 1e-12))
                th = np.clip(th, d_lo + 1e-6, b_hi - 1e-6)
                vr = float(Vr(th))
                vt = float(Vt(th))
                dxdt  = vr * np.cos(th) - np.sin(th) * vt
                detat = vr * np.sin(th) + np.cos(th) * vt
                return [dxdt, detat]

            # Terminate when x_loc reaches x_base (base plane)
            def back_event(t, state, xb=x_base):
                return state[0] - xb

            back_event.terminal = True

            t_max = (20.0 * max(x_base - x_le, 0.1))

            try:
                sol = solve_ivp(
                    stode,
                    (0.0, t_max),
                    [x_le, eta_le],
                    events=back_event,
                    max_step=self._delta_streamwise * self.length,
                    method='RK45',
                    rtol=1e-5,
                    atol=1e-8,
                )

                if sol.y.shape[1] < 2:
                    raise RuntimeError("Integration returned fewer than 2 points.")

                # Collect all integrated points plus the terminal event point if any
                x_arr   = sol.y[0].copy()
                eta_arr = sol.y[1].copy()

                # Append the event point if the terminal was triggered
                if sol.t_events[0].size > 0 and sol.y_events[0].shape[0] > 0:
                    x_arr   = np.append(x_arr,   sol.y_events[0][0, 0])
                    eta_arr = np.append(eta_arr,  sol.y_events[0][0, 1])

                # Resample to n_streamwise uniform points along x_loc
                # (guarantees consistent grid for surface lofting)
                x_target = np.linspace(x_le, x_base, self.n_streamwise)

                # Sort by x for interpolation
                sort_idx = np.argsort(x_arr)
                x_sorted   = x_arr[sort_idx]
                eta_sorted = eta_arr[sort_idx]

                # Remove duplicate x values
                _, uniq = np.unique(x_sorted, return_index=True)
                x_u   = x_sorted[uniq]
                eta_u = eta_sorted[uniq]

                if x_u.size >= 2:
                    f_eta = interp1d(
                        x_u, eta_u, kind='linear',
                        bounds_error=False,
                        fill_value=(eta_u[0], eta_u[-1]),
                    )
                    eta_resampled = f_eta(x_target)
                else:
                    eta_resampled = np.full(self.n_streamwise, eta_le)

                eta_resampled = np.maximum(eta_resampled, 0.0)

                # Transform from cone-local (x_loc, eta) to GUI (X, Y, Z)
                # The osculating plane has a normal tilted by alpha from the Y-axis:
                #     delta_Y = -eta * cos(alpha)
                #     delta_Z =  eta * sin(alpha)
                # x_loc is along the freestream (X-axis in GUI coords)
                stream = np.column_stack([
                    x_target,
                    -eta_resampled * np.cos(alpha),
                     eta_resampled * np.sin(alpha),
                ])

                # Apply cone-centre offset (transform cone-local -> global GUI)
                stream[:, 0] += cc[i, 0]
                stream[:, 1] += cc[i, 1]
                stream[:, 2] += cc[i, 2]

            except Exception:
                # Fallback: straight deflected line using per-plane theta
                bottom_y = (le_point[1]
                            - np.tan(np.radians(theta_i_deg))
                            * (self.length - le_point[0]))
                stream = np.column_stack([
                    np.linspace(le_point[0], self.length, self.n_streamwise),
                    np.linspace(le_point[1], bottom_y,   self.n_streamwise),
                    np.full(self.n_streamwise, le_point[2]),
                ])

            # Pin first point to leading edge (avoids visual/CAD gaps)
            stream[0] = le_point
            # Pin last point x to vehicle length and z to match upper TE
            stream[-1, 0] = self.length
            stream[-1, 2] = le_point[2]  # keep z constant along span
            self.lower_surface_streams.append(stream)

        # Align upper and lower stream endpoints
        for i in range(min(len(self.upper_surface_streams),
                          len(self.lower_surface_streams))):
            us = self.upper_surface_streams[i]
            ls = self.lower_surface_streams[i]
            if us.shape[0] >= 2 and ls.shape[0] >= 2:
                # Pin LE to leading edge
                if i < len(self.leading_edge):
                    us[0] = self.leading_edge[i]
                    ls[0] = self.leading_edge[i]
                # Pin TE x to vehicle length
                us[-1, 0] = self.length
                ls[-1, 0] = self.length

    # ====================================================================== #
    #  Spanwise smoothing                                                      #
    # ====================================================================== #

    def _smooth_surfaces(self, n_passes=5):
        """
        Apply spanwise smoothing to both upper and lower surface stream lists.

        Uses a 5-point weighted kernel [1, 4, 6, 4, 1] / 16 for each pass.
        The symmetry-plane stream (index 0), its neighbour (index 1), and the
        tip streams (last two indices) are pinned to preserve boundary geometry.

        Only Y-coordinates are smoothed; X and Z are not altered.
        """
        for streams in (self.upper_surface_streams, self.lower_surface_streams):
            N = len(streams)
            if N < 5:
                continue

            kernel = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0

            for _ in range(n_passes):
                smoothed = [s.copy() for s in streams]
                for i in range(2, N - 2):
                    s_prev2 = streams[i - 2]
                    s_prev1 = streams[i - 1]
                    s_curr  = streams[i]
                    s_next1 = streams[i + 1]
                    s_next2 = streams[i + 2]

                    # Only smooth Y; X and Z kept from current stream
                    n_pts = s_curr.shape[0]
                    n2 = min(n_pts, s_prev2.shape[0], s_prev1.shape[0],
                             s_next1.shape[0], s_next2.shape[0])

                    smoothed[i] = s_curr.copy()
                    smoothed[i][:n2, 1] = (
                        kernel[0] * s_prev2[:n2, 1]
                        + kernel[1] * s_prev1[:n2, 1]
                        + kernel[2] * s_curr[:n2,  1]
                        + kernel[3] * s_next1[:n2, 1]
                        + kernel[4] * s_next2[:n2, 1]
                    )

                # Pin boundary streams
                smoothed[0]  = streams[0].copy()
                smoothed[1]  = streams[1].copy()
                smoothed[-2] = streams[-2].copy()
                smoothed[-1] = streams[-1].copy()

                streams[:] = smoothed

    # ====================================================================== #
    #  Body / geometry info                                                    #
    # ====================================================================== #

    def get_body_geometry(self):
        """
        Return a dict of key geometric and aerodynamic summary information.

        Returns
        -------
        dict with keys:
            'cone_angle_min_deg'  : float
            'cone_angle_max_deg'  : float
            'mach_min'            : float
            'mach_max'            : float
            'mach_distribution'   : ndarray  (n_planes + 2,)
            'cone_angle_deg'      : ndarray  (n_planes + 2,)
            'z_stations'          : ndarray  (n_planes + 2,)
            'length'              : float
            'height'              : float
            'width'               : float
            'beta_deg'            : float
            'direction'           : str
        """
        return {
            'cone_angle_min_deg':  float(self.cone_angles_deg.min()),
            'cone_angle_max_deg':  float(self.cone_angles_deg.max()),
            'mach_min':            self.M_inf_min,
            'mach_max':            self.M_inf_max,
            'mach_distribution':   self._mach_per_station.copy(),
            'cone_angle_deg':      self.cone_angles_deg.copy(),
            'z_stations':          self._z_all.copy(),
            'length':              self.length,
            'height':              self.height,
            'width':               self.width,
            'beta_deg':            self.beta_deg,
            'direction':           self.direction,
        }

    # ====================================================================== #
    #  Mesh construction                                                       #
    # ====================================================================== #

    def _build_mesh(self):
        """
        Build a triangulated mesh from upper and lower surface streams.

        Returns
        -------
        vertices : ndarray (n_verts, 3)
        triangles : ndarray (n_tri, 3)  — 0-based vertex indices
        """
        us = self.upper_surface_streams
        ls = self.lower_surface_streams

        n_span = len(us)

        # Streams may have different lengths (tip stream has only 2 points in OC).
        # Use the most common stream length for the mesh; degenerate tip stream
        # is handled by clamping to its last valid point.
        n_stream_vals = [s.shape[0] for s in us]
        n_stream = max(n_stream_vals)

        def _pad_stream(s, n_target):
            """Extend a short stream to n_target by repeating its last point."""
            if s.shape[0] >= n_target:
                return s[:n_target]
            pad = np.tile(s[-1:, :], (n_target - s.shape[0], 1))
            return np.vstack([s, pad])

        us_padded = [_pad_stream(s, n_stream) for s in us]
        ls_padded = [_pad_stream(s, n_stream) for s in ls]

        verts_upper = np.vstack(us_padded)   # (n_span * n_stream, 3)
        verts_lower = np.vstack(ls_padded)   # (n_span * n_stream, 3)
        vertices = np.vstack([verts_upper, verts_lower])
        lower_start = n_span * n_stream

        triangles = []

        # Upper surface quads
        for i in range(n_span - 1):
            for j in range(n_stream - 1):
                v00 = i       * n_stream + j
                v01 = i       * n_stream + j + 1
                v10 = (i + 1) * n_stream + j
                v11 = (i + 1) * n_stream + j + 1
                triangles.append([v00, v10, v01])
                triangles.append([v01, v10, v11])

        # Lower surface quads (reversed winding)
        for i in range(n_span - 1):
            for j in range(n_stream - 1):
                v00 = lower_start + i       * n_stream + j
                v01 = lower_start + i       * n_stream + j + 1
                v10 = lower_start + (i + 1) * n_stream + j
                v11 = lower_start + (i + 1) * n_stream + j + 1
                triangles.append([v00, v01, v10])
                triangles.append([v01, v11, v10])

        # Leading edge cap: upper LE row to lower LE row
        for i in range(n_span - 1):
            u0 = i       * n_stream
            u1 = (i + 1) * n_stream
            l0 = lower_start + i       * n_stream
            l1 = lower_start + (i + 1) * n_stream
            triangles.append([u0, l0, u1])
            triangles.append([u1, l0, l1])

        # Trailing edge cap (base plane)
        for i in range(n_span - 1):
            u0 = i       * n_stream + (n_stream - 1)
            u1 = (i + 1) * n_stream + (n_stream - 1)
            l0 = lower_start + i       * n_stream + (n_stream - 1)
            l1 = lower_start + (i + 1) * n_stream + (n_stream - 1)
            triangles.append([u0, u1, l0])
            triangles.append([u1, l1, l0])

        # Symmetry-plane cap (i = 0, Z = 0)
        for j in range(n_stream - 1):
            u0 = j
            u1 = j + 1
            l0 = lower_start + j
            l1 = lower_start + j + 1
            triangles.append([u0, u1, l0])
            triangles.append([u1, l1, l0])

        # Wingtip cap (i = n_span - 1)
        i_tip = n_span - 1
        for j in range(n_stream - 1):
            u0 = i_tip * n_stream + j
            u1 = i_tip * n_stream + j + 1
            l0 = lower_start + i_tip * n_stream + j
            l1 = lower_start + i_tip * n_stream + j + 1
            triangles.append([u0, l0, u1])
            triangles.append([u1, l0, l1])

        return vertices, np.array(triangles, dtype=int)

    # ====================================================================== #
    #  File export                                                             #
    # ====================================================================== #

    def export_stl(self, filename):
        """
        Export the waverider mesh as an ASCII STL file.

        Parameters
        ----------
        filename : str
            Output file path (should end in .stl).
        """
        vertices, triangles = self._build_mesh()

        with open(filename, 'w') as fh:
            fh.write("solid vmof_waverider\n")
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
                fh.write(
                    f"  facet normal "
                    f"{normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n"
                )
                fh.write("    outer loop\n")
                fh.write(
                    f"      vertex "
                    f"{v0[0]:.8f} {v0[1]:.8f} {v0[2]:.8f}\n"
                )
                fh.write(
                    f"      vertex "
                    f"{v1[0]:.8f} {v1[1]:.8f} {v1[2]:.8f}\n"
                )
                fh.write(
                    f"      vertex "
                    f"{v2[0]:.8f} {v2[1]:.8f} {v2[2]:.8f}\n"
                )
                fh.write("    endloop\n")
                fh.write("  endfacet\n")
            fh.write("endsolid vmof_waverider\n")

        print(f"[VMOF] STL exported to {filename}")

    def export_tri(self, filename):
        """
        Export the waverider mesh in NASA Cart3D TRI format.

        Parameters
        ----------
        filename : str
            Output file path (should end in .tri).
        """
        vertices, triangles = self._build_mesh()
        n_v = len(vertices)
        n_t = len(triangles)

        with open(filename, 'w') as fh:
            fh.write(f"{n_v}\n")
            fh.write(f"{n_t}\n")
            for v in vertices:
                fh.write(f"{v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
            for tri in triangles:
                # TRI format: 1-based vertex indices
                fh.write(f"{tri[0]+1} {tri[1]+1} {tri[2]+1}\n")

        print(f"[VMOF] TRI exported to {filename}")

    # ====================================================================== #
    #  CAD export                                                              #
    # ====================================================================== #

    def to_CAD(self, sides='both', export=False,
               filename='vmof_waverider.step', **kwargs):
        """
        Export the VMOF waverider to STEP via the shared interpPlate pipeline.

        The streams are aligned at LE and TE during generation so that
        interpPlate boundaries are consistent.
        """
        scale = float(kwargs.get('scale', 1000.0))
        from waverider_generator.cad_export import to_CAD as _to_CAD
        return _to_CAD(self, sides=sides, export=export,
                       filename=filename, scale=scale)
