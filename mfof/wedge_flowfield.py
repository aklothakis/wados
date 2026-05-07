"""2D oblique-shock wedge basic flowfield.

Phase 3 deliverable. Unlike the cone (axisymmetric Taylor-Maccoll) and the
power-law body (axisymmetric MOC), the wedge has *no axis of revolution* --
the post-shock flow is uniform in 2D. The streamline downstream of the shock
is therefore a straight line at angle ``theta_wedge`` below the freestream.

Useful in osculating planes where the local ICC curvature radius
``R_osc -> infinity`` (the centre of a flat-shock waverider), where the
"axisymmetric" assumption degenerates anyway. Phase 5 will let an
:func:`mfof.osculating.build_all_osculating_planes` factory return a
``WedgeFlowfield`` for the flat region and a ``ConeFlowfield`` /
``PowerLawFlowfield`` for the curved region within the same waverider.
"""

import numpy as np

from liu2019.shock import (
    DetachedShockError,
    beta_detachment,
    mach_angle,
    oblique_shock_ratios,
    theta_from_beta_Ma,
)

from .basic_flowfield import BasicFlowfield, StreamlineResult


class WedgeFlowfield(BasicFlowfield):
    """2D oblique-shock wedge basic flowfield.

    The wedge half-angle is fully determined by ``(Ma_inf, beta_design)``
    via the standard theta-beta-M relation; no extra parameters. Both the
    leading-edge and trailing-edge deflection angles equal ``theta_wedge``
    (the post-shock flow is uniform).
    """

    def __init__(self, Ma_inf: float, beta_design_deg: float,
                 gamma: float = 1.4):
        super().__init__(Ma_inf, beta_design_deg, gamma)
        self._theta_w_deg = None     # lazy
        self._Ma2 = None             # lazy

    # ------------------------------------------------------------------
    def name(self) -> str:
        return f"wedge(Ma={self.Ma_inf:.2f},beta={self.beta_design_deg:.2f})"

    # ------------------------------------------------------------------
    def attached_shock_check(self) -> tuple:
        """Same Mach-angle and detachment limits as a 2D oblique shock."""
        mu_deg = mach_angle(self.Ma_inf)
        if self.beta_design_deg <= mu_deg:
            return False, (f"beta = {self.beta_design_deg:.2f} deg <= Mach "
                           f"angle mu = {mu_deg:.2f} deg")
        b_det = beta_detachment(self.Ma_inf, self.gamma)
        if self.beta_design_deg >= b_det:
            return False, (f"beta = {self.beta_design_deg:.2f} deg >= "
                           f"detachment beta_det = {b_det:.2f} deg")
        return True, "ok"

    # ------------------------------------------------------------------
    def deflection_angle_deg(self) -> float:
        """Wedge half-angle from the theta-beta-M relation.

        For a wedge this is constant throughout the post-shock flow (the
        oblique shock has no expansion fan downstream). The
        :class:`~mfof.basic_flowfield.BasicFlowfield` contract permits this:
        wedge is one of the two flowfield types whose LE deflection equals
        the everywhere-constant deflection.
        """
        if self._theta_w_deg is None:
            self._theta_w_deg = theta_from_beta_Ma(
                self.beta_design_deg, self.Ma_inf, self.gamma)
        return float(self._theta_w_deg)

    # ------------------------------------------------------------------
    def _post_shock_mach(self) -> float:
        """Post-shock Mach number from oblique-shock relations."""
        if self._Ma2 is None:
            ratios = oblique_shock_ratios(
                self.Ma_inf, self.beta_design_deg, self.gamma)
            self._Ma2 = float(ratios["Ma2"])
        return self._Ma2

    # ------------------------------------------------------------------
    def trace_streamline(self, x_LE: float, r_LE: float, x_end: float,
                         n_points: int = 100) -> StreamlineResult:
        """Straight-line streamline at angle ``theta_wedge`` below freestream.

        Identical *form* to :meth:`mfof.cone_flowfield.ConeFlowfield.trace_streamline`,
        but with ``theta_wedge`` (oblique-shock deflection) in place of
        ``delta_c`` (Taylor-Maccoll cone half-angle). Numerically these are
        DIFFERENT angles: at the same ``(Ma, beta)`` the wedge gives a
        smaller deflection than the cone (cone flow is a weaker compression
        because the post-shock flow can also expand around the cone in 3D).
        """
        theta_deg = self.deflection_angle_deg()
        theta_rad = np.radians(theta_deg)

        x_arr = np.linspace(float(x_LE), float(x_end), int(n_points))
        r_arr = float(r_LE) - (x_arr - float(x_LE)) * np.tan(theta_rad)

        return StreamlineResult(
            x_arr=x_arr,
            r_arr=r_arr,
            delta_LE_deg=theta_deg,
            delta_TE_deg=theta_deg,            # constant on a wedge
            Ma_TE=self._post_shock_mach(),
        )
