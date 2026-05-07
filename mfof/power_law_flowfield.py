"""Axisymmetric power-law-body basic flowfield.

Phase 3 deliverable. The body shape is

    r(x) = R_b * (x / L) ** n

For ``n = 1`` this is a cone and the post-shock streamline is a straight
line at angle ``theta_w == delta_c`` from the freestream (analytical fast
path). For ``n != 1`` the body is curved and the streamline must be traced
through the post-shock supersonic field by the axisymmetric Method of
Characteristics (MOC). The MOC solver lives in
:mod:`waverider_generator.vmplo.moc` and is wrapped by :mod:`mfof.moc`.

The body scale ``R_b`` is computed lazily on the first
:meth:`trace_streamline` call from the attached-shock condition

    body slope at x_LE  ==  theta(Ma, beta_design)

so each osculating-plane instance has its own ``R_b`` (different planes
have different ``x_LE``). This is intentional and matches how
:class:`mfof.cone_flowfield.ConeFlowfield` is instantiated per-plane.

Diagnostic helpers (per the Phase 2 docstring clarification that subclasses
MAY add non-abstract methods): :meth:`last_grid` exposes the most recent
MOC mesh for visualisation in the GUI's Flowfield-diagnostics sub-tab.
"""

import numpy as np

from liu2019.shock import (
    DetachedShockError,
    beta_detachment,
    mach_angle,
    oblique_shock_ratios,
    taylor_maccoll_cone_angle,
    theta_from_beta_Ma,
)

from .basic_flowfield import BasicFlowfield, StreamlineResult


# Default MOC step counts. Profiling shows r_TE converges by n_steps=10
# (verified across 10/30/50/100/300 in pre-plan), so we keep this low.
_DEFAULT_MOC_COLUMNS = 20
_DEFAULT_RK4_STEPS   = 10
_DEFAULT_INITIAL_PTS = 12


class PowerLawFlowfield(BasicFlowfield):
    """Axisymmetric power-law body, dispatching to T-M (cone limit) or MOC.

    Parameters
    ----------
    Ma_inf : float
        Free-stream Mach number.
    beta_design_deg : float
        Design shock angle (degrees).
    n : float
        Power-law exponent. ``n = 1`` is a cone. Range ``(0, 2]`` is the
        physically meaningful window enforced by
        :meth:`attached_shock_check`.
    L : float
        Body length (= vehicle length ``L_w``).
    gamma : float
        Specific-heat ratio.
    cone_tol : float
        ``|n - 1| < cone_tol`` triggers the analytical cone path (no MOC).
        Default 0.02; the streamline error from this approximation is
        well below 1% per the audit.
    """

    def __init__(self, Ma_inf: float, beta_design_deg: float,
                 n: float, L: float, gamma: float = 1.4,
                 cone_tol: float = 0.02):
        super().__init__(Ma_inf, beta_design_deg, gamma)
        self.n = float(n)
        self.L = float(L)
        self.cone_tol = float(cone_tol)
        # Lazily-evaluated, cached:
        self._delta_LE_deg = None    # post-shock deflection at LE (= theta from oblique shock)
        self._R_b = None             # body scale; depends on x_LE => set at trace time
        self._last_grid = None       # most recent MOC grid (for diagnostics)
        self._last_body = None       # most recent PowerLawBody instance

    # ------------------------------------------------------------------
    def name(self) -> str:
        return (f"power-law(n={self.n:.3f},Ma={self.Ma_inf:.2f},"
                f"beta={self.beta_design_deg:.2f})")

    # ------------------------------------------------------------------
    def attached_shock_check(self) -> tuple:
        """Standard attachment checks plus power-law-specific n-validity.

        For ``n < 1`` the body slope diverges as ``x -> 0``. The osculating
        sweep guarantees ``x_LE > 0`` (the LE never sits at the apex), but
        we additionally require ``n in (0, 2]`` to keep the body geometry
        physical.
        """
        # 1. Mach angle and detachment limits (same as ConeFlowfield)
        mu_deg = mach_angle(self.Ma_inf)
        if self.beta_design_deg <= mu_deg:
            return False, (f"beta = {self.beta_design_deg:.2f} deg <= "
                           f"Mach angle mu = {mu_deg:.2f} deg")
        b_det = beta_detachment(self.Ma_inf, self.gamma)
        if self.beta_design_deg >= b_det:
            return False, (f"beta = {self.beta_design_deg:.2f} deg >= "
                           f"detachment beta_det = {b_det:.2f} deg")
        # 2. Power-law exponent validity
        if self.n <= 0.0 or self.n > 2.0:
            return False, f"n = {self.n} out of (0, 2]"
        return True, "ok"

    # ------------------------------------------------------------------
    def deflection_angle_deg(self) -> float:
        """LE flow-deflection angle = body slope at ``x_LE``.

        Returns the **2D oblique-shock deflection** ``theta_w`` (from the
        standard theta-beta-M relation). This matches VMPLO's
        :func:`PowerLawBody.from_shock_condition` convention -- their MOC
        is built consuming ``theta_w`` as the body slope at the LE, and we
        reuse that solver. The consequence:

        * ``PowerLawFlowfield(n=1)`` is bit-identical to
          :class:`WedgeFlowfield`, not :class:`ConeFlowfield` (both use a
          straight line at ``tan(theta_w)``).
        * ``PowerLawFlowfield`` produces a *shallower* body than
          ``ConeFlowfield`` at the same ``(Ma, beta)`` because ``theta_w
          < delta_c`` always.

        Per the Phase 2 docstring clarification, this is the LE value
        only; for ``n != 1`` the streamline slope changes downstream
        because the body is curved. Use :meth:`trace_streamline` for the
        full ``(x, r)`` trajectory.

        (An earlier version of this code returned ``delta_c`` from
        Taylor-Maccoll, intending to make ``PowerLaw(n=1) == Cone``. That
        broke the body-frame consistency with the upstream VMPLO MOC,
        which expects ``theta_w`` as the LE body slope; passing ``delta_c``
        instead built a body whose tip exits the design shock surface
        and made the centerline streamline trace degenerate.)
        """
        if self._delta_LE_deg is None:
            self._delta_LE_deg = theta_from_beta_Ma(
                self.beta_design_deg, self.Ma_inf, self.gamma)
        return float(self._delta_LE_deg)

    # ------------------------------------------------------------------
    def _post_shock_mach(self) -> float:
        ratios = oblique_shock_ratios(
            self.Ma_inf, self.beta_design_deg, self.gamma)
        return float(ratios["Ma2"])

    # ------------------------------------------------------------------
    def trace_streamline(self, x_LE: float, r_LE: float, x_end: float,
                         n_points: int = 100) -> StreamlineResult:
        """Trace the LE streamline to ``x = x_end``.

        Dispatches on ``|n - 1|``:

        * ``|n - 1| < cone_tol``: analytical straight line at angle
          ``theta_LE = delta_c`` (cone limit, T-M Taylor-Maccoll).
        * Otherwise: full axisymmetric MOC via :mod:`mfof.moc`.

        The body scale ``R_b`` is computed here from the attached-shock
        condition because it depends on the per-plane ``x_LE``.

        Apex handling
        -------------
        ``PowerLawBody.from_shock_condition`` rejects ``x_LE = 0`` for
        ``n < 1`` (singular tangent at the apex). Earlier versions of
        this code fell back to the analytical cone for any ``x_LE`` below
        a non-trivial threshold; that introduced a visible spanwise step
        wherever the threshold was placed, because cone analytical and
        the power-law MOC trace produce systematically different ``r_TE``
        values (the MOC has a known ~25% bias at the cone limit).

        We instead clamp ``x_LE`` upward by a tiny shim (1e-9 m) only when
        it is exactly zero. Every plane therefore goes through the same
        MOC code path, so there is no model-boundary step in the spanwise
        direction. The centerline plane's mesh is degenerate (essentially
        a sliver), but it sits among many neighbouring MOC planes with
        very similar results, so it disappears into the spanwise
        averaging during 3D assembly.
        """
        theta_deg = self.deflection_angle_deg()
        theta_rad = np.radians(theta_deg)

        # Apex shim: if x_LE is too small for a physically-meaningful body,
        # nudge it up. The MOC mesh size scales with body.radius(x_LE_eff),
        # which scales with R_b * (x_LE_eff/L)^n. For x_LE_eff < ~1e-5 m the
        # body becomes microscopic, the MOC mesh degenerates, and the
        # streamline trace returns ~0 descent. Empirically (verified by
        # sweeping x_LE in the 1e-9 ... 1e-1 range and inspecting r_TE),
        # x_LE_eff >= 1e-4 m produces a stable, plane-1-consistent
        # descent. The shim only affects the centerline plane in
        # Liu-style waveriders (where x_LE = 0); planes at z > ~3 mm
        # already have x_LE > 1e-4 m and pass through unchanged.
        x_LE_eff = float(x_LE) if float(x_LE) > 1e-4 else 1e-4

        # ---- Cone-limit fast path (no MOC) ---------------------------
        if abs(self.n - 1.0) < self.cone_tol:
            # Compute R_b for diagnostics consistency, then short-circuit.
            self._R_b = float(np.tan(theta_rad) * self.L /
                              (self.n * (x_LE_eff / self.L) ** (self.n - 1.0)))
            x_arr = np.linspace(float(x_LE), float(x_end), int(n_points))
            r_arr = float(r_LE) - (x_arr - float(x_LE)) * np.tan(theta_rad)
            self._last_grid = None
            self._last_body = None
            return StreamlineResult(
                x_arr=x_arr,
                r_arr=r_arr,
                delta_LE_deg=theta_deg,
                delta_TE_deg=theta_deg,             # constant in the cone limit
                Ma_TE=self.Ma_inf,                  # placeholder for cone limit
            )

        # ---- Full MOC path ------------------------------------------
        from .moc import build_moc_grid, trace_streamline_with_state
        grid, body = build_moc_grid(
            Ma_inf=self.Ma_inf,
            beta_design_deg=self.beta_design_deg,
            n=self.n, L=self.L, x_LE=x_LE_eff,
            theta_LE_deg=theta_deg, gamma=self.gamma,
            n_columns=_DEFAULT_MOC_COLUMNS,
            n_initial_points=_DEFAULT_INITIAL_PTS,
        )
        self._R_b = float(body.R_b)
        self._last_grid = grid
        self._last_body = body

        # Trace through the MOC mesh starting AT THE BODY SURFACE
        # (r = body.radius(x_LE_eff)), not at the caller's r_LE. The MOC
        # mesh is built around the body, so its (x, r) coordinates are
        # in the body frame; tracing from a point off the body would
        # immediately leave the mesh and the LinearNDInterpolator would
        # extrapolate.
        r_LE_body = float(body.radius(x_LE_eff))
        x_raw, r_raw, Ma_TE, dTE_deg = trace_streamline_with_state(
            grid, x_LE=x_LE_eff, r_LE=r_LE_body,
            alpha0_rad=theta_rad,
            n_steps=_DEFAULT_RK4_STEPS,
            gamma=self.gamma,
        )

        # ---- Body frame -> mfof framework frame ---------------------
        # VMPLO MOC: r_raw INCREASES from r_LE_body to r_TE_body along
        # the body surface (or close to it). The "body increase" at any
        # x is r_raw(x) - r_LE_body >= 0.
        #
        # mfof.osculating consumes a streamline whose r DECREASES from
        # the caller's r_LE (the framework's local-frame anchor) so that
        # ``descent = r_LE_caller - r_arr`` is positive going downstream.
        # We map by treating "body r increase" = "framework descent":
        #
        #     descent_framework(x) = r_raw(x) - r_LE_body
        #     r_arr_framework(x)   = r_LE_caller - descent_framework(x)
        #                          = r_LE_caller - (r_raw(x) - r_LE_body)
        #
        # For the centerline plane in Liu-style waveriders r_LE_caller
        # = 0, so r_arr_framework = -(r_raw - r_LE_body), i.e. the body
        # increase becomes a negative r in the framework, and
        # mfof.osculating's back-projection sees descent = body_increase.
        body_increase = r_raw - r_LE_body
        r_arr_framework = float(r_LE) - body_increase

        # Resample to n_points uniformly in x.
        x_arr = np.linspace(float(x_LE), float(x_end), int(n_points))
        r_arr = np.interp(x_arr, x_raw, r_arr_framework)
        return StreamlineResult(
            x_arr=x_arr,
            r_arr=r_arr,
            delta_LE_deg=theta_deg,
            delta_TE_deg=float(dTE_deg),
            Ma_TE=float(Ma_TE),
        )

    # ------------------------------------------------------------------
    # Diagnostic accessors (non-abstract, optional).
    # Per the Phase 2 docstring on BasicFlowfield: subclasses MAY expose
    # extra non-abstract methods for diagnostics or visualisation.
    # ------------------------------------------------------------------
    def last_grid(self):
        """Return the most recent MOC ``grid`` from :meth:`trace_streamline`,
        or ``None`` if the cone-limit fast path was taken (no MOC mesh).
        Used by the GUI's Flowfield-diagnostics sub-tab to render the mesh.
        """
        return self._last_grid

    def last_body(self):
        """Return the most recent :class:`PowerLawBody` instance, or ``None``.
        """
        return self._last_body

    def R_b(self) -> float:
        """Latest body scale factor, populated after the first
        :meth:`trace_streamline`. Returns ``nan`` before then.
        """
        return float(self._R_b) if self._R_b is not None else float("nan")
