"""Cone basic flowfield using Taylor-Maccoll.

The initial (and currently only) concrete :class:`BasicFlowfield` subclass.
It wraps the validated Taylor-Maccoll solver from :mod:`liu2019.shock`, so the
cone half-angle and resulting compression-surface streamline are byte-for-byte
identical to what the Liu 2019 package produces -- the foundation of the
Phase 2 numerical-equivalence guarantee.

The compression-surface streamline on the cone is approximated as a straight
line at angle ``delta_c`` from the freestream (Sobieczky / Liu legacy
prescription), matching the production state of ``liu2019.osculating``.
"""

import numpy as np

from liu2019.shock import (
    DetachedShockError,
    beta_detachment,
    mach_angle,
    taylor_maccoll_cone_angle,
)

from .basic_flowfield import BasicFlowfield, StreamlineResult


class ConeFlowfield(BasicFlowfield):
    """Cone (Taylor-Maccoll) basic flowfield.

    The cone half-angle ``delta_c`` is derived from ``(Ma_inf, beta_design)``
    by integrating Taylor-Maccoll inward from the shock; the value is
    lazy-cached on the first call to :meth:`deflection_angle_deg`. The
    compression-surface streamline is a straight line at angle ``delta_c``
    in the local osculating-plane ``(x, r)`` frame.
    """

    def __init__(self, Ma_inf: float, beta_design_deg: float,
                 gamma: float = 1.4):
        super().__init__(Ma_inf, beta_design_deg, gamma)
        self._delta_c_deg = None    # lazy

    # --------------------------------------------------------------
    def name(self) -> str:
        return f"cone(Ma={self.Ma_inf:.2f},beta={self.beta_design_deg:.2f})"

    # --------------------------------------------------------------
    def attached_shock_check(self) -> tuple:
        """Confirm beta is between the Mach angle and the detachment angle."""
        mu_deg = mach_angle(self.Ma_inf)
        if self.beta_design_deg <= mu_deg:
            return False, (f"beta = {self.beta_design_deg:.2f} deg <= Mach "
                           f"angle mu = {mu_deg:.2f} deg")
        b_det = beta_detachment(self.Ma_inf, self.gamma)
        if self.beta_design_deg >= b_det:
            return False, (f"beta = {self.beta_design_deg:.2f} deg >= "
                           f"detachment beta_det = {b_det:.2f} deg")
        return True, "ok"

    # --------------------------------------------------------------
    def deflection_angle_deg(self) -> float:
        """Cone half-angle ``delta_c`` (degrees) from Taylor-Maccoll.

        Lazily evaluated, then cached. Identical to
        ``liu2019.shock.taylor_maccoll_cone_angle(Ma, beta, gamma)``.
        """
        if self._delta_c_deg is None:
            self._delta_c_deg = taylor_maccoll_cone_angle(
                self.Ma_inf, self.beta_design_deg, self.gamma)
        return float(self._delta_c_deg)

    # --------------------------------------------------------------
    def trace_streamline(self, x_LE: float, r_LE: float, x_end: float,
                         n_points: int = 100) -> StreamlineResult:
        """Cone-surface streamline: straight line at angle ``delta_c`` from LE.

        Reproduces the legacy Liu-2019 ``tan(delta_c)`` formula used in
        ``liu2019.osculating``'s production path. The osculating sweep
        consumes ``r_LE - r_arr = (x_arr - x_LE) * tan(delta_c)`` as the
        in-plane descent and back-projects to 3D via the local inward normal.
        """
        delta_c_deg = self.deflection_angle_deg()
        delta_c_rad = np.radians(delta_c_deg)

        x_arr = np.linspace(float(x_LE), float(x_end), int(n_points))
        r_arr = float(r_LE) - (x_arr - float(x_LE)) * np.tan(delta_c_rad)

        return StreamlineResult(
            x_arr=x_arr,
            r_arr=r_arr,
            delta_LE_deg=delta_c_deg,
            delta_TE_deg=delta_c_deg,         # constant on a cone
            Ma_TE=self.Ma_inf,                # placeholder; refine in future phases
        )
