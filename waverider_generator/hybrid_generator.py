"""
goc_generator.py
================
Generalized Osculating-Conical (GOC) waverider generator.

Generates a hybrid waverider that blends cone-derived (CD) and
osculating-cone (OC) flowfields continuously via a blend function h(z, x):

    h = 0  →  pure cone-derived  (high volume, uniform shock, good for nose/payload)
    h = 1  →  pure OC            (spatially-varying curvature, high L/D)

Design variables
----------------
From existing OC generator (X1–X4):
    X1  — shock-curve flat-region fraction  (spanwise)
    X2  — shock-curve wingtip height        (fraction of waverider height)
    X3  — upper surface control point 1
    X4  — upper surface control point 2

New GOC parameters:
    beta_CD  — shock angle [deg] for the cone-derived generating cone
               (independent of OC shock angle beta_OC)
    h0       — blend value at symmetry plane  z = 0    (0..1)
    h1       — blend value at wingtip          z = W    (0..1)
    x_t      — streamwise transition centre, as fraction of length (0..1)
    dx_t     — streamwise transition width,  as fraction of length (0..1, >0)

Coordinate convention (same as generator.py)
---------------------------------------------
    x  — streamwise  (0 = nose tip, L = base plane)
    y  — vertical    (0 = tip height, negative below)
    z  — spanwise    (0 = symmetry plane, W = wingtip)

Architecture
------------
Step 1 — Precompute T-M flowfields
    For each osculating plane i:
        TM_OC[i]  →  Vr_OC(θ), Vt_OC(θ)  using per-plane β_OC(z_i)
        TM_CD     →  Vr_CD(θ), Vt_CD(θ)  using uniform β_CD  (same for all planes)

Step 2 — Construct 3D shock surface S(z, x)
    At base plane (x = L):  S = blend of OC Bézier shock and CD cone surface
    At each x station:      back-project both shocks along their cone generators,
                            blend with h(z, x)
    Shock attachment is enforced exactly: streamlines start ON S by construction.

Step 3 — Streamline tracing with interpolated flowfield
    At each integration step (z_i, x_k):
        V_blend = (1 - h) * V_CD  +  h * V_OC_i
    The streamline follows the blended velocity field.

Step 4 — Upper surface (free-stream surface, unchanged from OC)

Limiting behaviour
------------------
    h(z, x) = 0 everywhere  →  identical to cone-derived waverider
    h(z, x) = 1 everywhere  →  identical to OC waverider (X1–X4)
    h(z)    constant in x   →  spanwise-only blend (intermediate step)
"""

import numpy as np
# NumPy 2.x removed np.trapz; use np.trapezoid with a shim for older versions
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz  # type: ignore[attr-defined]
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
from scipy.integrate import solve_ivp
from typing import Union

# --------------------------------------------------------------------------- #
#  Re-use flowfield primitives from the existing package if available,        #
#  otherwise fall back to inline copies (for standalone testing).             #
# --------------------------------------------------------------------------- #
try:
    from waverider_generator.flowfield import cone_field, cone_angle, shock_angle
    _FLOWFIELD_FROM_PACKAGE = True
except ImportError:
    _FLOWFIELD_FROM_PACKAGE = False

    def Taylor_Maccoll(t, x, gamma):
        A = (gamma - 1.0) / 2.0 * (1.0 - x[0]**2 - x[1]**2)
        dxdt = np.zeros(2)
        dxdt[0] = x[1]
        dxdt[1] = (x[1] * x[0] * x[1] - A * (2.0 * x[0] + x[1] / np.tan(t))) / (A - x[1]**2)
        return dxdt

    from scipy.interpolate import UnivariateSpline as _US
    from scipy.optimize import fsolve as _fsolve

    def cone_field(Mach, theta_rad, beta_rad, gamma):
        d = np.arctan(2.0 / np.tan(beta_rad) * (Mach**2 * np.sin(beta_rad)**2 - 1.0) /
                      (Mach**2 * (gamma + np.cos(2 * beta_rad)) + 2.0))
        Ma2 = (1.0 / np.sin(beta_rad - d) *
               np.sqrt((1.0 + (gamma-1.0)/2.0 * Mach**2 * np.sin(beta_rad)**2) /
                       (gamma * Mach**2 * np.sin(beta_rad)**2 - (gamma-1.0)/2.0)))
        V = 1.0 / np.sqrt(2.0 / ((gamma-1.0) * Ma2**2) + 1.0)
        Vr0 = V * np.cos(beta_rad - d)
        Vt0 = -(V * np.sin(beta_rad - d))
        sol = solve_ivp(Taylor_Maccoll, (beta_rad, theta_rad), [Vr0, Vt0], args=(gamma,))
        Vrf = _US(sol.t[::-1], sol.y[0, ::-1], k=min(3, sol.t.size - 1))
        Vtf = _US(sol.t[::-1], sol.y[1, ::-1], k=min(3, sol.t.size - 1))
        return [Vrf, Vtf]

    def cone_angle(Mach, shock_angle_deg, gamma):
        beta_rad = np.radians(shock_angle_deg)
        d = np.arctan(2.0 / np.tan(beta_rad) * (Mach**2 * np.sin(beta_rad)**2 - 1.0) /
                      (Mach**2 * (gamma + np.cos(2 * beta_rad)) + 2.0))
        Ma2 = (1.0 / np.sin(beta_rad - d) *
               np.sqrt((1.0 + (gamma-1.0)/2.0 * Mach**2 * np.sin(beta_rad)**2) /
                       (gamma * Mach**2 * np.sin(beta_rad)**2 - (gamma-1.0)/2.0)))
        V = 1.0 / np.sqrt(2.0 / ((gamma-1.0) * Ma2**2) + 1.0)
        Vr0 = V * np.cos(beta_rad - d)
        Vt0_v = -(V * np.sin(beta_rad - d))
        def ev(t, y): return y[1]
        ev.terminal = True
        sol = solve_ivp(Taylor_Maccoll, (beta_rad, 0.0), [Vr0, Vt0_v], events=ev, args=(gamma,))
        return float(np.degrees(sol.t_events[0][0]))

    def shock_angle(Mach, theta_deg, gamma):
        theta_rad = np.radians(theta_deg)
        def residual(beta):
            d = np.arctan(2.0 / np.tan(beta) * (Mach**2 * np.sin(beta)**2 - 1.0) /
                          (Mach**2 * (gamma + np.cos(2 * beta)) + 2.0))
            Ma2 = (1.0 / np.sin(beta - d) *
                   np.sqrt((1.0 + (gamma-1.0)/2.0 * Mach**2 * np.sin(beta)**2) /
                           (gamma * Mach**2 * np.sin(beta)**2 - (gamma-1.0)/2.0)))
            V = 1.0 / np.sqrt(2.0 / ((gamma-1.0) * Ma2**2) + 1.0)
            Vr0 = V * np.cos(beta - d)
            Vt0_v = -(V * np.sin(beta - d))
            def ev(t, y): return y[1]
            ev.terminal = True
            sol = solve_ivp(Taylor_Maccoll, (beta, 0.0), [Vr0, Vt0_v], events=ev, args=(gamma,))
            return sol.t_events[0][0] - theta_rad
        beta0 = _fsolve(residual, theta_rad)
        return float(np.degrees(beta0[0]))


# --------------------------------------------------------------------------- #
#  Small helpers                                                               #
# --------------------------------------------------------------------------- #

def _euclidean_2d(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def _cot(a):
    return 1.0 / np.tan(a)

def _line(z, m, c):
    return m * z + c


# =========================================================================== #
#                        GOCWaverider class                                   #
# =========================================================================== #

class GOCWaverider:
    """
    Generalized Osculating-Conical waverider.

    Parameters
    ----------
    M_inf : float
        Freestream Mach number.
    beta_OC : float
        Shock angle [deg] used by the OC generator at the base plane.
        This drives X1–X4 control points exactly as in the original waverider class.
    beta_CD : float
        Shock angle [deg] for the cone-derived generating cone.
        When h=0 everywhere the full vehicle is derived from this cone.
    height : float
        Waverider height (vertical extent of base plane cross-section) [m].
    width : float
        Half-span at base plane [m].
    dp : list[float]
        [X1, X2, X3, X4] — identical semantics to the original OC generator.
    h0 : float
        Blend value at symmetry plane (z=0).  0=CD, 1=OC.
    h1 : float
        Blend value at wingtip (z=width).     0=CD, 1=OC.
    x_t : float
        Streamwise transition centre as fraction of length (0–1).
        x_t=0 → transition starts at nose; x_t=1 → at base.
    dx_t : float
        Streamwise transition width as fraction of length (>0).
        Large dx_t → gradual.  Small dx_t → sharp step.
    n_planes : int
        Number of osculating planes (spanwise resolution).
    n_streamwise : int
        Number of points along each streamline.
    delta_streamwise : float
        ODE max-step as fraction of length (0 < δ ≤ 0.2).
    n_upper_surface : int
        Points used to interpolate upper-surface Bézier curve.
    n_shockwave : int
        Points used to interpolate shockwave Bézier curve.
    gamma : float
        Ratio of specific heats (default 1.4).
    """

    def __init__(
        self,
        M_inf: Union[float, int],
        beta_OC: Union[float, int],
        beta_CD: Union[float, int],
        height: Union[float, int],
        width: Union[float, int],
        dp: list,
        h0: float = 0.0,
        h1: float = 1.0,
        x_t: float = 0.5,
        dx_t: float = 0.2,
        n_planes: int = 20,
        n_streamwise: int = 30,
        delta_streamwise: float = 0.05,
        n_upper_surface: int = 100,
        n_shockwave: int = 100,
        gamma: float = 1.4,
        use_per_plane_beta: bool = False,
        **kwargs,
    ):
        # ---- validate & store ----
        if M_inf <= 0:
            raise ValueError("Mach number must be positive.")
        self.M_inf = float(M_inf)

        if not (0 < beta_OC < 90):
            raise ValueError("beta_OC must be between 0 and 90 degrees.")
        self.beta_OC = float(beta_OC)

        if not (0 < beta_CD < 90):
            raise ValueError("beta_CD must be between 0 and 90 degrees.")
        self.beta_CD = float(beta_CD)

        if height <= 0:
            raise ValueError("height must be positive.")
        self.height = float(height)

        if width <= 0:
            raise ValueError("width must be positive.")
        self.width = float(width)

        if len(dp) != 4:
            raise ValueError("dp must have exactly 4 elements [X1, X2, X3, X4].")
        self.X1, self.X2, self.X3, self.X4 = [float(v) for v in dp]

        if not (0 <= self.X1 < 1 and 0 <= self.X2 <= 1):
            raise ValueError("X1 must be in [0,1) and X2 in [0,1].")
        if not (self.X2 / (1 - self.X1)**4 < (7/64) * (self.width / self.height)**4):
            raise ValueError("OC inverse-design condition violated (X1/X2 constraint).")
        if not (0 <= self.X3 <= 1 and 0 <= self.X4 <= 1):
            raise ValueError("X3 and X4 must be in [0, 1].")

        if not (0.0 <= h0 <= 1.0 and 0.0 <= h1 <= 1.0):
            raise ValueError("h0 and h1 must be in [0, 1].")
        self.h0, self.h1 = float(h0), float(h1)
        self.blend_exp = float(kwargs.get('blend_exp', 1.0))

        # Dome (volume loft): parabolic bump on upper surface interior
        # dome_height: max y-offset at mid-chord centerline [m]
        # dome_taper:  spanwise decay exponent (1=linear, 2=parabolic)
        self.dome_height = float(kwargs.get('dome_height', 0.0))
        self.dome_taper = float(kwargs.get('dome_taper', 2.0))

        if not (0.0 <= x_t <= 1.0):
            raise ValueError("x_t must be in [0, 1].")
        if dx_t <= 0:
            raise ValueError("dx_t must be positive.")
        self.x_t, self.dx_t = float(x_t), float(dx_t)

        if n_planes < 10:
            raise ValueError("n_planes must be >= 10.")
        self.n_planes = int(n_planes)

        if n_streamwise < 10:
            raise ValueError("n_streamwise must be >= 10.")
        self.n_streamwise = int(n_streamwise)

        if not (0 < delta_streamwise <= 0.2):
            raise ValueError("delta_streamwise must be in (0, 0.2].")
        self.delta_streamwise = float(delta_streamwise)
        self.gamma = float(gamma)
        self._use_per_plane_beta = bool(use_per_plane_beta)

        # Independent-profiles blend mode: if h0_nose/h1_nose are passed as
        # keyword arguments, activate independent nose/base spanwise profiles.
        # Otherwise use the default multiplicative mode.
        # These are not formal constructor params to keep the signature clean;
        # pass them as: GOCWaverider(..., h0_nose=0.2, h1_nose=0.8)
        self.h0_nose = float(kwargs.get('h0_nose', 0.0))
        self.h1_nose = float(kwargs.get('h1_nose', 0.0))
        self._independent_profiles = ('h0_nose' in kwargs or 'h1_nose' in kwargs)

        # ---- derived geometry ----
        self.length = self.height / np.tan(np.radians(self.beta_OC))
        self.theta_OC = self._oblique_deflection(self.M_inf, self.beta_OC, self.gamma)
        self.theta_CD = self._oblique_deflection(self.M_inf, self.beta_CD, self.gamma)
        self.cone_angle_OC = cone_angle(self.M_inf, self.beta_OC, self.gamma)
        self.cone_angle_CD = cone_angle(self.M_inf, self.beta_CD, self.gamma)

        # Interior osculating plane z positions
        self.z_planes = np.linspace(0, self.width, self.n_planes + 2)[1:-1]

        # ---- build geometry objects ----
        self._build_oc_bezier_curves()
        self._create_interpolated_shockwave(n_shockwave)
        self._create_interpolated_upper_surface(n_upper_surface)

        # OC shape arrays — must come BEFORE Step 1 so curvature radii are available
        self._y_local_sw = np.zeros((self.n_planes, 1))
        self._get_shockwave_curve()
        self._local_us_intersections = np.zeros((self.n_planes, 2))
        self._find_us_intersections()
        self._leading_edge = np.zeros((self.n_planes + 2, 3))
        self._cone_centers = np.zeros((self.n_planes, 3))
        self._leading_edge[-1, :] = [
            self.length,
            self._local_to_global(self.X2 * self.height),
            self.width,
        ]
        self._compute_leading_edge_and_cone_centers()

        # STEP 1 — Precompute T-M flowfields
        # Runs after geometry so _oc_shock_angle_at_z() can use curvature radii
        self._precompute_tm_fields()

        # STEP 2 — 3D shock surface
        self._build_shock_surface()

        # STEP 4 — Upper surface
        self._compute_upper_surface()

        # STEP 3 — Streamline tracing
        self.lower_surface_streams = []
        self._streamline_tracing()
        self._regularize_lower_streams()

        # Public aliases for to_CAD() and GUI compatibility
        self.leading_edge = self._leading_edge
        self.beta = self.beta_OC  # default beta for info panels

        # Build (n_streams, n_pts, 3) surface arrays for plot_surface()
        self.upper_surface = self._streams_to_array(self.upper_surface_streams)
        self.lower_surface = self._streams_to_array(self.lower_surface_streams)

    @staticmethod
    def _streams_to_array(streams):
        """Convert list of variable-length (n,3) arrays to uniform (N, M, 3) array."""
        if not streams:
            return np.zeros((0, 0, 3))
        n_pts = max(s.shape[0] for s in streams)
        out = np.zeros((len(streams), n_pts, 3))
        for i, s in enumerate(streams):
            if s.shape[0] == n_pts:
                out[i] = s
            elif s.shape[0] < 2:
                out[i] = np.tile(s[0], (n_pts, 1))
            else:
                # Resample via linear interpolation
                t_old = np.linspace(0, 1, s.shape[0])
                t_new = np.linspace(0, 1, n_pts)
                for c in range(3):
                    out[i, :, c] = np.interp(t_new, t_old, s[:, c])
        return out

    # ===================================================================== #
    #  STEP 1 — Precompute T-M flowfields                                   #
    # ===================================================================== #

    def _precompute_tm_fields(self):
        """
        Precompute T-M velocity splines using a beta lookup table.

        Strategy
        --------
        Rather than calling cone_field() N_planes times (once per osculating
        plane), we:
          1. Collect the actual per-plane beta values from _oc_shock_angle_at_z().
          2. Find the unique beta range and build a lookup table of N_beta_lut
             evenly-spaced T-M solutions spanning [beta_min, beta_max].
          3. At query time, bilinearly interpolate between adjacent table entries.

        This costs N_beta_lut T-M integrations (~20) instead of N_planes (~20+),
        but gives O(1) per-streamline-step evaluation and generalises cleanly to
        off-design Mach sweeps (add a Mach axis to the table later).

        Attributes set
        --------------
        self._tm_cd          : (Vr_spline, Vt_spline) for the CD cone (single)
        self._tm_lut_betas   : ndarray (N_lut,) of beta values [rad]
        self._tm_lut         : list of N_lut (Vr_spline, Vt_spline) entries
        self._beta_per_plane : ndarray (n_planes,) local beta [deg] per plane
        self._tm_oc          : list of (Vr_callable, Vt_callable) per plane,
                               each backed by LUT interpolation — same interface
                               as before so _streamline_tracing() needs no change
        """
        # ---- Step 1a: collect per-plane beta values ----
        self._beta_per_plane = np.array(
            [self._oc_shock_angle_at_z(z) for z in self.z_planes]
        )

        # ---- Step 1b: CD field (single, uniform) ----
        self._tm_cd = cone_field(
            self.M_inf,
            self.cone_angle_CD * np.pi / 180.0,
            self.beta_CD * np.pi / 180.0,
            self.gamma,
        )

        # ---- Step 1c: build beta lookup table ----
        beta_min = float(self._beta_per_plane.min())
        beta_max = float(self._beta_per_plane.max())

        # Always span at least a 1-degree bracket so the LUT is well-conditioned
        # even when all planes share the same beta (pure OC uniform case).
        beta_span = max(beta_max - beta_min, 1.0)
        # Guard: pull min/max slightly inside weak-shock/detachment limits
        beta_lo = max(beta_min - 0.1 * beta_span, beta_min - 1.0)
        beta_hi = min(beta_max + 0.1 * beta_span, beta_max + 1.0)

        N_lut = max(8, int(np.ceil(beta_span / 0.5)) + 2)   # ≥1 entry per 0.5 deg
        self._tm_lut_betas = np.linspace(beta_lo, beta_hi, N_lut)   # deg
        self._tm_lut = []

        for b_deg in self._tm_lut_betas:
            ca_deg = cone_angle(self.M_inf, b_deg, self.gamma)
            tm = cone_field(
                self.M_inf,
                ca_deg * np.pi / 180.0,
                b_deg  * np.pi / 180.0,
                self.gamma,
            )
            self._tm_lut.append(tm)

        # ---- Step 1d: build per-plane OC callables from LUT ----
        # Each entry is a (Vr_fn, Vt_fn) that interpolates the LUT at the
        # plane's local beta.  We capture beta_i by value in the closure.
        self._tm_oc = []
        for beta_i in self._beta_per_plane:
            vr_fn, vt_fn = self._make_lut_interpolant(float(beta_i))
            self._tm_oc.append((vr_fn, vt_fn))

    def _make_lut_interpolant(self, beta_deg: float):
        """
        Return (Vr_callable, Vt_callable) that evaluate the T-M velocity
        components at a given polar angle θ [rad] by linearly interpolating
        between the two nearest LUT entries for beta_deg.

        The returned callables have the same signature as the UnivariateSpline
        objects returned by cone_field(), so _streamline_tracing() needs no
        changes.
        """
        betas = self._tm_lut_betas          # deg, sorted ascending
        lut   = self._tm_lut                # list of (Vr_spline, Vt_spline)

        # Find bracketing indices (clip to valid range)
        idx = np.searchsorted(betas, beta_deg) - 1
        idx = int(np.clip(idx, 0, len(betas) - 2))
        j0, j1 = idx, idx + 1

        b0, b1 = betas[j0], betas[j1]
        alpha = (beta_deg - b0) / (b1 - b0) if (b1 - b0) > 1e-12 else 0.0
        alpha = float(np.clip(alpha, 0.0, 1.0))

        Vr0, Vt0 = lut[j0]
        Vr1, Vt1 = lut[j1]

        def Vr_interp(th):
            return (1.0 - alpha) * float(Vr0(th)) + alpha * float(Vr1(th))

        def Vt_interp(th):
            return (1.0 - alpha) * float(Vt0(th)) + alpha * float(Vt1(th))

        return Vr_interp, Vt_interp

    def _oc_shock_angle_at_z(self, z: float) -> float:
        """
        Return the local OC shock angle [deg] for the osculating plane at z.

        Physical derivation
        -------------------
        In the OC method each osculating plane contains a cone whose half-angle
        θ_cone is determined by the local radius of curvature R of the shock
        profile curve.  The relationship is:

            x_apex = L - R / tan(β_global)

        where L is vehicle length and β_global is the global OC shock angle.
        The *effective* shock angle in the plane is therefore the same β_global
        — the OC method prescribes the shock angle, not the cone half-angle,
        as the design parameter.

        However, the local shock curvature does affect the 3D pressure
        distribution through the transverse pressure gradient (neglected in the
        standard OC approximation).  A refined estimate of the effective local
        shock angle can be derived from the local curvature radius R_i:

            Δβ_i ≈ -arctan(height / R_i) * correction_factor

        This is a first-order correction.  For typical waveriders with moderate
        curvature (R >> height) the correction is < 1 deg and well within the
        OC cross-flow approximation error.

        Implementation
        --------------
        We provide two modes controlled by self._use_per_plane_beta:
          False (default) : return beta_OC uniformly — exact OC behaviour
          True            : apply first-order curvature correction

        The mode is set by the `use_per_plane_beta` constructor argument.
        """
        if not getattr(self, '_use_per_plane_beta', False):
            return self.beta_OC

        # Per-plane mode: read curvature radius for this z from pre-built table.
        # The table self._curvature_radii is built by _build_curvature_table()
        # which is called from _compute_leading_edge_and_cone_centers().
        if not hasattr(self, '_curvature_table'):
            return self.beta_OC

        # Interpolate curvature radius at z
        z_tab, R_tab = self._curvature_table
        R_i = float(np.interp(z, z_tab, R_tab,
                              left=R_tab[0], right=R_tab[-1]))

        # First-order correction: Δβ ≈ -arctan(h / R_i)
        # Sign: tighter curvature (smaller R) → cone compresses more → slightly
        # higher effective shock angle.
        if R_i < 1e-6:
            return self.beta_OC
        delta_beta = np.degrees(np.arctan(self.height / R_i))
        beta_local = np.clip(self.beta_OC + delta_beta,
                             self.beta_OC * 0.8,
                             self.beta_OC * 1.4)
        return float(beta_local)

    # ===================================================================== #
    #  STEP 2 — 3D shock surface construction                               #
    # ===================================================================== #

    def _build_shock_surface(self):
        """
        Build self.shock_surface : ndarray (n_planes+2, n_streamwise, 3)

        At each (z_i, x_k) point:
            shock_OC(z_i, x_k) — OC shock back-projected along its cone generator
            shock_CD(z_i, x_k) — CD cone surface (r = x * tan(beta_CD), circular arc)
            S(z_i, x_k) = (1 - h) * shock_CD  +  h * shock_OC

        This enforces exact shock attachment at every streamwise station.
        Streamlines seeded from S will always lie on the intended shock surface.
        """
        n_sp = self.n_planes + 2
        x_stations = np.linspace(0.0, self.length, self.n_streamwise)
        self.shock_surface = np.zeros((n_sp, self.n_streamwise, 3))

        z_all, y_local_all = self._get_augmented_sw_arrays()

        for i in range(n_sp):
            z_i = z_all[i]
            y_global_i = self._local_to_global(float(y_local_all[i]))

            for k, x_k in enumerate(x_stations):

                # ---- OC shock: back-project from base plane ----
                if i == 0 or i == n_sp - 1:
                    # Symmetry plane or wingtip edge: simple linear projection
                    shock_oc = np.array([x_k, y_global_i, z_i])
                else:
                    cc = self._cone_centers[i - 1]       # global (x,y,z)
                    denom = cc[0] - self.length
                    t_p = 0.0 if abs(denom) < 1e-12 else np.clip(
                        (x_k - self.length) / denom, 0.0, 1.0
                    )
                    shock_oc = np.array([
                        x_k,
                        y_global_i + t_p * (cc[1] - y_global_i),
                        z_i      + t_p * (cc[2] - z_i),
                    ])

                # ---- CD shock: circular cone cross-section ----
                # r(x) = x * tan(beta_CD); map z_i to azimuth angle phi
                r_cd = x_k * np.tan(np.radians(self.beta_CD))
                phi = np.clip(z_i / self.width, 0.0, 1.0) * (np.pi / 2.0)
                shock_cd = np.array([
                    x_k,
                    -r_cd * np.cos(phi),    # below axis
                     r_cd * np.sin(phi),
                ])

                # ---- Blend ----
                h = self._blend(z_i, x_k)
                self.shock_surface[i, k, :] = (1 - h) * shock_cd + h * shock_oc

    # ===================================================================== #
    #  STEP 3 — Streamline tracing with blended T-M fields                  #
    # ===================================================================== #

    def _streamline_tracing(self):
        """
        Trace lower-surface streamlines using the blended T-M velocity field.

        For each osculating plane i, at each ODE integration step with current
        global-x = x_curr:
            h        = _blend(z_i, x_curr)
            Vr(θ)    = (1-h) * Vr_CD(θ) + h * Vr_OC_i(θ)
            Vθ(θ)    = (1-h) * Vt_CD(θ) + h * Vt_OC_i(θ)

        Limiting cases:
            h=0 everywhere  →  ODE is identical to original generator (CD)
            h=1 everywhere  →  ODE is identical to original generator (OC)

        Blending approximation note:
            The blended velocity field is a geometric interpolation of two
            valid T-M solutions.  It is not irrotational.  This is the same
            class of approximation as cross-flow neglect in standard OC.
            Shock attachment is exact (enforced by Step 2).
        """
        z_all, y_local_all = self._get_augmented_sw_arrays()
        cc_all = np.vstack([
            [0.0, 0.0, 0.0],
            self._cone_centers,
            [self.length, self._local_to_global(self.X2 * self.height), self.width],
        ])
        us_all = np.vstack([
            [0.0, self.height],
            self._local_us_intersections,
            [self.width, self.X2 * self.height],
        ])

        n_sp = self.n_planes + 2

        for i in range(n_sp):
            z_i = float(z_all[i])
            y_local_i = float(y_local_all[i])
            le = self._leading_edge[i, :]

            # ---- Tip ----
            if i == n_sp - 1:
                tip = self._leading_edge[-1, :]
                self.lower_surface_streams.append(np.vstack([tip, tip]))
                continue

            # ---- Flat / symmetry-plane region ----
            if z_i <= self.X1 * self.width or self.X2 == 0:
                h_mid = self._blend(z_i, self.length / 2.0)
                theta_eff = (1 - h_mid) * self.theta_CD + h_mid * self.theta_OC
                bottom_y = le[1] - np.tan(np.radians(theta_eff)) * (self.length - le[0])
                x_arr = np.linspace(le[0], self.length, self.n_streamwise)[:, None]
                y_arr = np.linspace(le[1], bottom_y,   self.n_streamwise)[:, None]
                z_arr = np.full_like(y_arr, le[2])
                self.lower_surface_streams.append(np.column_stack([x_arr, y_arr, z_arr]))
                continue

            # ---- Curved region ----
            cc_i = cc_all[i]

            eta_le = _euclidean_2d(
                us_all[i, 0], self._local_to_global(us_all[i, 1]),
                cc_i[2],      cc_i[1],
            )
            r_shock = _euclidean_2d(
                z_i,              self._local_to_global(y_local_i),
                cc_i[2],          cc_i[1],
            )

            m_sw, _, _ = self._get_first_derivative(z_i)
            alpha = np.arctan(m_sw)

            x_le_local = eta_le / np.tan(np.radians(self.beta_OC))
            x_terminate = r_shock / np.tan(np.radians(self.beta_OC))

            # T-M splines for this plane (index offset: plane 0 = z_planes[0])
            tm_oc_idx = max(0, min(i - 1, len(self._tm_oc) - 1))
            Vr_OC, Vt_OC = self._tm_oc[tm_oc_idx]
            Vr_CD, Vt_CD = self._tm_cd

            def stode_blended(t, state,
                              _z_i=z_i, _cc_i=cc_i,
                              _Vr_OC=Vr_OC, _Vt_OC=Vt_OC,
                              _Vr_CD=Vr_CD, _Vt_CD=Vt_CD):
                x_loc, eta_loc = state
                x_global = np.clip(x_loc + _cc_i[0], 0.0, self.length)

                # Polar angle in osculating cone
                if abs(x_loc) < 1e-12:
                    th = np.radians(self.beta_OC)
                else:
                    th = np.arctan(eta_loc / x_loc)
                th = np.clip(th, 1e-6, np.radians(self.beta_OC) * 1.5)

                h = self._blend(_z_i, x_global)
                vr = (1 - h) * float(_Vr_CD(th)) + h * float(_Vr_OC(th))
                vt = (1 - h) * float(_Vt_CD(th)) + h * float(_Vt_OC(th))

                return [
                    vr * np.cos(th) - np.sin(th) * vt,
                    vr * np.sin(th) + np.cos(th) * vt,
                ]

            def back_event(t, state):
                return state[0] - x_terminate
            back_event.terminal = True

            try:
                sol = solve_ivp(
                    stode_blended,
                    (0, 1e4),
                    [x_le_local, eta_le],
                    events=back_event,
                    max_step=self.delta_streamwise * self.length,
                    method='RK45',
                )
                x_s =  sol.y[0] + cc_i[0]
                y_s = -sol.y[1] * np.cos(alpha) + cc_i[1]
                z_s =  sol.y[1] * np.sin(alpha) + cc_i[2]
                stream = np.column_stack([x_s, y_s, z_s])
            except Exception:
                # Graceful fallback to straight line
                x_arr = np.linspace(le[0], self.length, self.n_streamwise)
                y_arr = np.linspace(le[1], self._local_to_global(y_local_i), self.n_streamwise)
                z_arr = np.full_like(x_arr, z_i)
                stream = np.column_stack([x_arr, y_arr, z_arr])

            self.lower_surface_streams.append(stream)

    # ===================================================================== #
    #  Post-processing: regularize lower surface streams                    #
    # ===================================================================== #

    def _regularize_lower_streams(self):
        """
        Post-process lower_surface_streams to fix surface inconsistencies.

        The blended T-M ODE produces variable-length streams whose trailing-
        edge positions can drift in y and z, causing folds/overlaps in the
        surface.  This method:
          1. Extends / trims each stream so that it ends exactly at x = L.
          2. Enforces that each stream's z-coordinate stays at its osculating-
             plane z value (prevents cross-plane drift from alpha rotation).
          3. Resamples every stream to exactly self.n_streamwise points with
             uniform arc-length spacing.
          4. Smooths the trailing-edge y-curve across spans to eliminate
             discontinuities between adjacent streams.
        """
        n_sp = len(self.lower_surface_streams)
        if n_sp < 3:
            return

        L = self.length
        n_pts = self.n_streamwise

        # --- Step 1–3: extend to base, pin z, resample ----------------
        regularized = []
        for i, stream in enumerate(self.lower_surface_streams):
            # Skip degenerate tip stream
            if stream.shape[0] < 3:
                # Align degenerate tip to upper surface tip
                if i < len(self.upper_surface_streams):
                    tip = self.upper_surface_streams[i][0].copy()
                    regularized.append(np.vstack([tip, tip]))
                else:
                    regularized.append(stream)
                continue

            # Use upper surface LE z to match both surfaces exactly.
            # The upper surface z comes from the Bézier shockwave profile
            # (_leading_edge), while the ODE-traced lower surface may use
            # the uniform z_planes grid — causing a z-mismatch that makes
            # the CAD solid non-watertight.
            if i < len(self.upper_surface_streams):
                z_plane = float(self.upper_surface_streams[i][0, 2])
            else:
                z_plane = float(stream[0, 2])

            # Pin z to upper surface's osculating-plane z
            stream = stream.copy()
            stream[:, 2] = z_plane

            # Align LE position (x, y) to upper surface LE
            if i < len(self.upper_surface_streams):
                us_le = self.upper_surface_streams[i][0]
                stream[0, 0] = us_le[0]
                stream[0, 1] = us_le[1]

            # Ensure the stream reaches x = L
            if stream[-1, 0] < L - 1e-10:
                # Extrapolate last two points to x = L
                dx = stream[-1, 0] - stream[-2, 0]
                dy = stream[-1, 1] - stream[-2, 1]
                if abs(dx) > 1e-14:
                    slope = dy / dx
                    y_end = stream[-1, 1] + slope * (L - stream[-1, 0])
                else:
                    y_end = stream[-1, 1]
                end_pt = np.array([[L, y_end, z_plane]])
                stream = np.vstack([stream, end_pt])
            elif stream[-1, 0] > L + 1e-10:
                # Trim: find last point before L and interpolate
                mask = stream[:, 0] <= L + 1e-10
                if mask.sum() >= 2:
                    stream = stream[mask]
                stream[-1, 0] = L

            # Resample to uniform n_pts using arc-length parameterization
            dx = np.diff(stream[:, 0])
            dy = np.diff(stream[:, 1])
            ds = np.sqrt(dx**2 + dy**2)
            s = np.concatenate([[0.0], np.cumsum(ds)])
            if s[-1] < 1e-14:
                regularized.append(stream[:2])
                continue
            s /= s[-1]  # normalize to [0, 1]

            t_new = np.linspace(0, 1, n_pts)
            resampled = np.zeros((n_pts, 3))
            resampled[:, 0] = np.interp(t_new, s, stream[:, 0])
            resampled[:, 1] = np.interp(t_new, s, stream[:, 1])
            resampled[:, 2] = z_plane
            # Enforce exact endpoints matching upper surface
            if i < len(self.upper_surface_streams):
                resampled[0, :] = self.upper_surface_streams[i][0]
            else:
                resampled[0] = stream[0]
            resampled[0, 2] = z_plane
            resampled[-1, 0] = L
            resampled[-1, 2] = z_plane

            regularized.append(resampled)

        # --- Step 4: smooth trailing-edge y across spans ---------------
        # Collect TE y-values (skip degenerate tip)
        te_y = np.array([s[-1, 1] for s in regularized])
        n_real = n_sp - 1  # exclude tip
        if n_real >= 5:
            # Apply light Gaussian-like smoothing (3-point moving average, 2 passes)
            for _ in range(2):
                smoothed = te_y[:n_real].copy()
                for j in range(1, n_real - 1):
                    smoothed[j] = 0.25 * te_y[j-1] + 0.5 * te_y[j] + 0.25 * te_y[j+1]
                # Pin endpoints (symmetry plane and last real stream)
                smoothed[0] = te_y[0]
                smoothed[-1] = te_y[n_real - 1]
                te_y[:n_real] = smoothed

            # Apply smoothed TE y to streams (scale entire y profile)
            for i in range(n_real):
                s = regularized[i]
                if s.shape[0] < 3:
                    continue
                old_te_y = s[-1, 1]
                new_te_y = te_y[i]
                if abs(old_te_y - s[0, 1]) > 1e-14:
                    # Scale y linearly: LE stays fixed, TE shifts
                    t_frac = np.linspace(0, 1, s.shape[0])
                    s[:, 1] += t_frac * (new_te_y - old_te_y)

        self.lower_surface_streams = regularized

    # ===================================================================== #
    #  Blend function h(z, x)                                               #
    # ===================================================================== #

    def _blend(self, z: float, x: float) -> float:
        """
        Compute h(z, x) ∈ [0, 1].

        Two modes, selected by self._independent_profiles (default False):

        Multiplicative mode (default)
        ------------------------------
            h(z, x) = h_span(z) * h_stream(x)

            h_span(z)   = h0*(1-t)^p + h1*t^p              power-law ramp (t=z/W)
            h_stream(x) = sigmoid((x/L - x_t) / dx_t)   sigmoid ramp

        At x=0 (nose):  h_stream≈0  →  h≈0 (pure CD everywhere at nose)
        At x=L (base):  h_stream≈1  →  h = h_span(z)

        This is the correct formulation for the mission direction nose=CD,
        base=OC.  The nose is always cone-derived regardless of spanwise
        position; the base plane carries the full spanwise blend.

        Independent profiles mode (use_per_plane_beta=True users may want this)
        -------------------------------------------------------------------------
            h(z, x) = h_nose(z) * (1 - h_stream(x))
                    + h_base(z) * h_stream(x)

            h_nose(z) = h0_nose + (h1_nose - h0_nose) * z/W
            h_base(z) = h0      + (h1      - h0     ) * z/W

        Requires self.h0_nose, self.h1_nose to be set (via constructor kwargs).
        At x=0: h = h_nose(z).   At x=L: h = h_base(z).  Fully independent.

        Parameters
        ----------
        z : float   spanwise position [m]
        x : float   streamwise position [m]
        """
        # Streamwise sigmoid (shared by both modes)
        x_frac = x / self.length
        exp_arg = np.clip((x_frac - self.x_t) / max(self.dx_t, 1e-12), -50, 50)
        h_stream = 1.0 / (1.0 + np.exp(-exp_arg))

        z_frac = np.clip(z / self.width, 0.0, 1.0)

        p = self.blend_exp
        if getattr(self, '_independent_profiles', False):
            h_nose = self.h0_nose * (1.0 - z_frac)**p + self.h1_nose * z_frac**p
            h_base = self.h0      * (1.0 - z_frac)**p + self.h1      * z_frac**p
            h = h_nose * (1.0 - h_stream) + h_base * h_stream
        else:
            h_span = self.h0 * (1.0 - z_frac)**p + self.h1 * z_frac**p
            h = h_span * h_stream

        return float(np.clip(h, 0.0, 1.0))

    # ===================================================================== #
    #  OC Bézier geometry  (mirrors generator.py, with _-prefix)            #
    # ===================================================================== #

    def _build_oc_bezier_curves(self):
        self._s_cp = np.zeros((5, 2))
        self._s_cp[:, 0] = np.linspace(self.X1 * self.width, self.width, 5)
        self._s_cp[-1, 1] = self.X2 * self.height
        self._s_P = [self._s_cp[k, :] for k in range(5)]

        self._us_cp = np.zeros((4, 2))
        self._us_cp[:, 0] = np.linspace(0, self.width, 4)
        self._us_cp[0, 1] = self.height
        self._us_cp[1, 1] = self.height - (1 - self.X2) * self.X3 * self.height
        self._us_cp[2, 1] = self.height - (1 - self.X2) * self.X4 * self.height
        self._us_cp[3, :] = self._s_P[4]
        self._us_P = [self._us_cp[k, :] for k in range(4)]

    def _bezier_shockwave(self, t):
        P = self._s_P
        return ((1-t)**4*P[0] + 4*(1-t)**3*t*P[1] +
                6*(1-t)**2*t**2*P[2] + 4*(1-t)*t**3*P[3] + t**4*P[4])

    def _bezier_upper_surface(self, t):
        P = self._us_P
        return ((1-t)**3*P[0] + 3*(1-t)**2*t*P[1] +
                3*(1-t)*t**2*P[2] + t**3*P[3])

    def _bezier_sw_first_derivative(self, t):
        P = self._s_P
        d = (4*(1-t)**3*(P[1]-P[0]) + 12*(1-t)**2*t*(P[2]-P[1]) +
             12*(1-t)*t**2*(P[3]-P[2]) + 4*t**3*(P[4]-P[3]))
        return d[1]/d[0], d[0], d[1]

    def _bezier_sw_second_derivative(self, t):
        P = self._s_P
        d2 = (12*(1-t)**2*(P[2]-2*P[1]+P[0]) + 24*(1-t)*t*(P[3]-2*P[2]+P[1]) +
              12*t**2*(P[4]-2*P[3]+P[2]))
        return d2[0], d2[1]

    def _create_interpolated_shockwave(self, n):
        t_vals = np.linspace(0, 1, n)
        pts = np.array([self._bezier_shockwave(t) for t in t_vals])
        self._interp_sw = interp1d(pts[:, 0], pts[:, 1], kind='linear')

    def _create_interpolated_upper_surface(self, n):
        t_vals = np.linspace(0, 1, n)
        pts = np.array([self._bezier_upper_surface(t) for t in t_vals])
        self._interp_us = interp1d(pts[:, 0], pts[:, 1], kind='linear')

    def _find_t_value(self, z):
        def f(t): return self._bezier_shockwave(t)[0] - z
        return root_scalar(f, bracket=[0, 1]).root

    def _calculate_radius_curvature(self, t):
        _, dzdt, dydt = self._bezier_sw_first_derivative(t)
        dzdt2, dydt2 = self._bezier_sw_second_derivative(t)
        return 1.0 / (abs(dzdt*dydt2 - dydt*dzdt2) / (dzdt**2 + dydt**2)**1.5)

    def _get_first_derivative(self, z):
        t = self._find_t_value(z)
        return self._bezier_sw_first_derivative(t)

    def _get_shockwave_curve(self):
        for i, z in enumerate(self.z_planes):
            self._y_local_sw[i, 0] = (
                0.0 if z <= self.width * self.X1
                else float(self._interp_sw(float(z)))
            )

    def _find_us_intersections(self):
        for i, z in enumerate(self.z_planes):
            if z <= self.X1 * self.width or self.X2 == 0:
                self._local_us_intersections[i, 0] = z
                self._local_us_intersections[i, 1] = self._interp_us(z)
            else:
                m_sw, _, _ = self._get_first_derivative(z)
                y_s = float(self._y_local_sw[i, 0])
                c = y_s + (1.0 / m_sw) * z
                m = -1.0 / m_sw
                def f_int(zz): return _line(zz, m, c) - self._interp_us(zz)
                res = root_scalar(f_int, bracket=[0, self.width])
                z_int = res.root
                self._local_us_intersections[i, :] = [z_int, _line(z_int, m, c)]

    def _compute_leading_edge_and_cone_centers(self):
        for i, z in enumerate(self.z_planes):
            if z <= self.X1 * self.width or self.X2 == 0:
                self._cone_centers[i, 0] = (
                    self.length -
                    (self._local_us_intersections[i, 1] - self._y_local_sw[i, 0]) /
                    np.tan(np.radians(self.beta_OC))
                )
                self._cone_centers[i, 1] = self._local_to_global(
                    self._local_us_intersections[i, 1]
                )
                self._cone_centers[i, 2] = float(z)
                self._leading_edge[i + 1, :] = self._cone_centers[i, :]
            else:
                t = self._find_t_value(z)
                m_sw, _, _ = self._bezier_sw_first_derivative(t)
                radius = self._calculate_radius_curvature(t)
                theta_a = np.arctan(m_sw)

                self._cone_centers[i, 0] = float(
                    self.length - radius / np.tan(np.radians(self.beta_OC))
                )
                self._cone_centers[i, 1] = float(
                    self._local_to_global(self._y_local_sw[i, 0]) +
                    np.cos(theta_a) * radius
                )
                self._cone_centers[i, 2] = float(z - radius * np.sin(theta_a))

                self._leading_edge[i + 1, :] = self._intersection_freestream_plane(
                    self._cone_centers[i, 0], self._cone_centers[i, 1], self._cone_centers[i, 2],
                    self.length,
                    self._local_to_global(self._y_local_sw[i, 0]),
                    z,
                    self._local_to_global(self._local_us_intersections[i, 1]),
                )

        # Build curvature table now that all radii are available
        self._build_curvature_table()

    def _build_curvature_table(self):
        """
        Build self._curvature_table = (z_array, R_array) mapping each
        osculating plane's z position to its shock-profile radius of curvature.

        Used by _oc_shock_angle_at_z() when use_per_plane_beta=True.

        For the flat region (z ≤ X1·W) the curvature is zero (infinite radius),
        so we assign R = 1e6 * self.length as a practical infinity sentinel.
        """
        z_vals = []
        R_vals = []

        # Add symmetry-plane sentinel
        z_vals.append(0.0)
        R_vals.append(1e6 * self.length)

        for i, z in enumerate(self.z_planes):
            z_vals.append(float(z))
            if z <= self.X1 * self.width or self.X2 == 0:
                R_vals.append(1e6 * self.length)
            else:
                t = self._find_t_value(z)
                R = self._calculate_radius_curvature(t)
                # Guard against degenerate curvature at the flat-to-curved transition
                R_vals.append(max(float(R), 1e-3 * self.length))

        # Add wingtip sentinel
        z_vals.append(self.width)
        # Curvature at wingtip: use last plane value
        R_vals.append(R_vals[-1])

        self._curvature_table = (np.array(z_vals), np.array(R_vals))

    def _compute_upper_surface(self):
        n_sp = self.n_planes + 2
        self.upper_surface_x = np.zeros((n_sp, self.n_streamwise))
        self.upper_surface_y = np.zeros((n_sp, self.n_streamwise))
        self.upper_surface_z = np.zeros((n_sp, self.n_streamwise))

        # Symmetry plane (x-axis strip along centreline)
        self.upper_surface_x[0, :] = np.linspace(0, self.length, self.n_streamwise)
        if self.dome_height > 0:
            t = np.linspace(0, 1, self.n_streamwise)
            # sin²(π·t^0.6): forward-peaked (~32% chord), tangent at LE and TE
            self.upper_surface_y[0, :] = self.dome_height * np.sin(np.pi * t ** 0.6) ** 2
        else:
            self.upper_surface_y[0, :] = 0.0
        self.upper_surface_z[0, :] = 0.0

        for i in range(self.n_planes):
            le = self._leading_edge[i + 1, :]
            us_y = self._local_to_global(self._local_us_intersections[i, 1])
            us_z = self._local_us_intersections[i, 0]
            self.upper_surface_x[i + 1, :] = np.linspace(le[0], self.length, self.n_streamwise)
            y_line = np.linspace(le[1], us_y, self.n_streamwise)

            # Dome: parabolic bump pinned to zero at LE (t=0) and TE (t=1)
            if self.dome_height > 0:
                z_frac = np.clip(le[2] / self.width, 0.0, 1.0)
                # Spanwise taper: cosine — zero slope at both centerline and wingtip
                # dome_taper controls concentration: >1 shifts dome toward center,
                # <1 extends dome further outboard
                span_factor = 0.5 * (1.0 + np.cos(np.pi * z_frac ** self.dome_taper))
                # Chordwise bump: sin²(π·t^0.6), peaks at ~32% chord, tangent at TE
                t = np.linspace(0, 1, self.n_streamwise)
                bump = np.sin(np.pi * t ** 0.6) ** 2
                y_line += self.dome_height * span_factor * bump

            self.upper_surface_y[i + 1, :] = y_line
            self.upper_surface_z[i + 1, :] = np.linspace(le[2], us_z, self.n_streamwise)

        self.upper_surface_x[-1, :] = self.length
        self.upper_surface_y[-1, :] = self._local_to_global(self.X2 * self.height)
        self.upper_surface_z[-1, :] = self.width

        self.upper_surface_streams = []
        for i in range(n_sp):
            row = np.vstack([
                self.upper_surface_x[i, :],
                self.upper_surface_y[i, :],
                self.upper_surface_z[i, :],
            ]).T
            if i == n_sp - 1:
                row = row[:2, :]
            self.upper_surface_streams.append(row)

    # ===================================================================== #
    #  Utility                                                               #
    # ===================================================================== #

    def _local_to_global(self, y_local: float) -> float:
        return float(y_local) - self.height

    def _oblique_deflection(self, M, beta_deg, gamma) -> float:
        b = np.radians(beta_deg)
        tan_t = (2.0 * _cot(b) * (M**2 * np.sin(b)**2 - 1.0) /
                 (M**2 * (gamma + np.cos(2 * b)) + 2.0))
        return float(np.degrees(np.arctan(tan_t)))

    def _intersection_freestream_plane(self, xC, yC, zC, xS, yS, zS, y_target):
        k = (y_target - yS) / (yC - yS)
        return np.array([xS + k*(xC - xS), y_target, zS + k*(zC - zS)])

    def _get_augmented_sw_arrays(self):
        z_all = np.concatenate([[0.0], self.z_planes, [self.width]])
        y_local_all = np.concatenate([
            [0.0],
            self._y_local_sw[:, 0],
            [self.X2 * self.height],
        ])
        return z_all, y_local_all

    # ===================================================================== #
    #  Blend statistics (for GUI display)                                    #
    # ===================================================================== #

    def get_blend_stats(self) -> dict:
        """Area-weighted blend statistics for GUI display."""
        z_vals = self._leading_edge[:, 2]
        # Evaluate h at mid-length (representative streamwise position)
        x_mid = self.length * 0.5
        h_vals = np.array([self._blend(float(z), x_mid) for z in z_vals])
        if len(z_vals) < 2:
            mean_h = float(h_vals[0])
        else:
            denom = np.trapezoid(z_vals, z_vals)
            mean_h = float(np.trapezoid(h_vals * z_vals, z_vals) / denom) if abs(denom) > 1e-12 else float(h_vals.mean())
        return {
            'mean_h': mean_h,
            'oc_fraction': mean_h * 100,
            'cd_fraction': (1 - mean_h) * 100,
        }

    def get_blend_profile(self, n_points: int = 100):
        """Returns (z_vals, h_vals_at_base) for blend profile plotting."""
        z_vals = np.linspace(0, self.width, n_points)
        h_vals = np.array([self._blend(float(z), self.length) for z in z_vals])
        return z_vals, h_vals

    # ===================================================================== #
    #  Küchemann τ                                                           #
    # ===================================================================== #

    def volumetric_efficiency(self, A_ref: float = None) -> float:
        """
        Compute Küchemann τ = V^(2/3) / A_ref.

        Parameters
        ----------
        A_ref : float, optional
            Reference area [m²].  Defaults to planform area estimated from
            leading-edge x vs z curve.

        Returns
        -------
        float
        """
        vol = self._estimate_volume()
        if A_ref is None:
            A_ref = self._estimate_planform_area()
        if A_ref < 1e-20:
            return 0.0
        return vol**(2.0/3.0) / A_ref

    def _estimate_volume(self) -> float:
        """
        Trapezoidal integration of cross-sectional area along x (×2 for symmetry).
        Samples the lower-surface streams at n_x streamwise stations.
        """
        if not self.lower_surface_streams:
            return 0.0
        n_x = min(self.n_streamwise, 25)
        x_stations = np.linspace(0.01 * self.length, self.length, n_x)
        areas = []
        for x_k in x_stations:
            yz = []
            for stream in self.lower_surface_streams:
                if stream.shape[0] < 2:
                    continue
                xi, yi, zi = stream[:, 0], stream[:, 1], stream[:, 2]
                if x_k < xi.min() or x_k > xi.max():
                    continue
                yz.append((float(np.interp(x_k, xi, zi)),
                            float(np.interp(x_k, xi, yi))))
            if len(yz) < 3:
                areas.append(0.0)
                continue
            yz.sort(key=lambda p: p[0])
            zz = [p[0] for p in yz]
            yy = [p[1] for p in yz]
            areas.append(2.0 * abs(np.trapezoid(yy, zz)))
        return abs(np.trapezoid(areas, x_stations))

    def _estimate_planform_area(self) -> float:
        """Estimate planform area from leading-edge curve (×2 for symmetry)."""
        le_x = self._leading_edge[:, 0]
        le_z = self._leading_edge[:, 2]
        return 2.0 * abs(np.trapezoid(self.length - le_x, le_z))
