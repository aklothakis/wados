"""Oblique shock and Taylor-Maccoll cone-flow utilities for Liu 2019.

Contents
--------
- theta_from_beta_Ma, beta_from_theta_Ma : theta-beta-M relations (2D wedge)
- beta_detachment                         : detachment shock angle
- mach_angle                              : Mach angle arcsin(1/Ma)
- oblique_shock_ratios                    : full post-shock state (2D)
- taylor_maccoll_cone_angle               : cone half-angle for given shock angle

The Taylor-Maccoll solver re-uses the implementation already shipped in
``waverider_generator.flowfield`` when available, so that the Liu 2019 module
stays numerically consistent with the rest of the project. A local fallback
is provided if that import fails.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


class DetachedShockError(Exception):
    """Raised when no attached oblique shock exists for the requested (Ma, theta)."""


# ---------------------------------------------------------------------------
# 2D oblique-shock relations
# ---------------------------------------------------------------------------

def theta_from_beta_Ma(beta_deg, Ma, gamma=1.4):
    """Flow-deflection theta (degrees) for freestream Ma and shock angle beta.

    Standard theta-beta-M relation; returns radians-> degrees."""
    beta = np.radians(beta_deg)
    num  = Ma ** 2 * np.sin(beta) ** 2 - 1.0
    den  = Ma ** 2 * (gamma + np.cos(2.0 * beta)) + 2.0
    if den == 0.0:
        return 0.0
    tan_theta = 2.0 / np.tan(beta) * num / den
    return float(np.degrees(np.arctan(tan_theta)))


def mach_angle(Ma):
    """Mach wave angle mu = arcsin(1/Ma), degrees."""
    if Ma <= 1.0:
        return 90.0
    return float(np.degrees(np.arcsin(1.0 / Ma)))


def beta_detachment(Ma, gamma=1.4):
    """Shock angle at which theta_max is achieved (degrees).

    Found numerically by maximising theta(beta) over beta in (mu, 90)."""
    mu = mach_angle(Ma)
    beta_grid = np.linspace(mu + 1e-3, 90.0 - 1e-3, 2001)
    thetas = np.array([theta_from_beta_Ma(b, Ma, gamma) for b in beta_grid])
    return float(beta_grid[int(np.argmax(thetas))])


def beta_from_theta_Ma(theta_deg, Ma, gamma=1.4, weak=True):
    """Invert theta-beta-M (weak or strong branch).

    Raises DetachedShockError if theta exceeds theta_max for this Ma.
    """
    mu = mach_angle(Ma)
    beta_det = beta_detachment(Ma, gamma)
    theta_max = theta_from_beta_Ma(beta_det, Ma, gamma)
    if theta_deg > theta_max + 1e-9:
        raise DetachedShockError(
            f"theta = {theta_deg:.3f} deg exceeds theta_max = {theta_max:.3f} "
            f"deg at Ma = {Ma:.3f}; shock would detach."
        )
    if theta_deg <= 0.0:
        return mu

    def residual(b):
        return theta_from_beta_Ma(b, Ma, gamma) - theta_deg

    if weak:
        return float(brentq(residual, mu + 1e-6, beta_det - 1e-6))
    return float(brentq(residual, beta_det + 1e-6, 90.0 - 1e-6))


def oblique_shock_ratios(Ma, beta_deg, gamma=1.4):
    """Full post-shock state across an oblique shock at angle beta_deg.

    Returns dict with keys: Ma2, p2_p1, rho2_rho1, T2_T1, theta_deg, T02_T01.
    """
    beta = np.radians(beta_deg)
    M1n2 = (Ma * np.sin(beta)) ** 2
    if M1n2 < 1.0:
        raise DetachedShockError(
            f"Normal Mach M1n = {np.sqrt(M1n2):.3f} < 1 at beta = {beta_deg:.3f} deg."
        )
    theta_deg = theta_from_beta_Ma(beta_deg, Ma, gamma)
    p_ratio   = 1.0 + 2.0 * gamma / (gamma + 1.0) * (M1n2 - 1.0)
    rho_ratio = (gamma + 1.0) * M1n2 / ((gamma - 1.0) * M1n2 + 2.0)
    T_ratio   = p_ratio / rho_ratio
    M2n2 = ((gamma - 1.0) * M1n2 + 2.0) / (2.0 * gamma * M1n2 - (gamma - 1.0))
    Ma2 = np.sqrt(M2n2) / np.sin(beta - np.radians(theta_deg))
    return {
        "Ma2": float(Ma2),
        "p2_p1": float(p_ratio),
        "rho2_rho1": float(rho_ratio),
        "T2_T1": float(T_ratio),
        "theta_deg": float(theta_deg),
        "T02_T01": 1.0,   # adiabatic shock — total temperature is conserved
    }


# ---------------------------------------------------------------------------
# Taylor-Maccoll cone flow
# ---------------------------------------------------------------------------

def _taylor_maccoll_rhs(t, x, gamma):
    A = (gamma - 1.0) / 2.0 * (1.0 - x[0] ** 2 - x[1] ** 2)
    return [x[1],
            (x[1] * x[0] * x[1] - A * (2.0 * x[0] + x[1] / np.tan(t)))
            / (A - x[1] ** 2)]


def _vt_event(t, y, gamma):
    return y[1]
_vt_event.terminal = True


def _taylor_maccoll_cone_angle_local(Ma, beta_deg, gamma=1.4):
    """Local implementation of cone-angle solver (fallback)."""
    beta = np.radians(beta_deg)
    d = np.arctan(
        2.0 / np.tan(beta)
        * (Ma ** 2 * np.sin(beta) ** 2 - 1.0)
        / (Ma ** 2 * (gamma + np.cos(2.0 * beta)) + 2.0)
    )
    if d <= 0.0:
        raise DetachedShockError(
            f"Ma = {Ma:.3f}, beta = {beta_deg:.3f} deg yields non-positive "
            "deflection; shock is below Mach angle."
        )
    Ma2 = 1.0 / np.sin(beta - d) * np.sqrt(
        (1.0 + (gamma - 1.0) / 2.0 * Ma ** 2 * np.sin(beta) ** 2)
        / (gamma * Ma ** 2 * np.sin(beta) ** 2 - (gamma - 1.0) / 2.0)
    )
    V  = 1.0 / np.sqrt(2.0 / ((gamma - 1.0) * Ma2 ** 2) + 1.0)
    Vr =  V * np.cos(beta - d)
    Vt = -V * np.sin(beta - d)
    event = lambda t, y: _vt_event(t, y, gamma)
    event.terminal = True
    sol = solve_ivp(
        lambda t, y: _taylor_maccoll_rhs(t, y, gamma),
        (beta, 1e-4),
        [Vr, Vt],
        events=event,
        method="RK45",
        rtol=1e-5, atol=1e-8,
    )
    if not sol.t_events[0].size:
        raise DetachedShockError(
            f"Taylor-Maccoll integration did not reach Vt = 0 for "
            f"Ma = {Ma:.3f}, beta = {beta_deg:.3f} deg."
        )
    return float(np.degrees(sol.t_events[0][0]))


def taylor_maccoll_cone_angle(Ma, beta_deg, gamma=1.4):
    """Cone half-angle delta_c (degrees) whose attached shock sits at beta_deg
    for freestream Ma.

    Uses ``waverider_generator.flowfield.cone_angle`` if importable, else the
    local fallback above. Both solve the Taylor-Maccoll ODE with a Vt=0 event.
    """
    try:
        from waverider_generator.flowfield import cone_angle as _ca
        return float(_ca(Ma, beta_deg, gamma))
    except Exception:
        return _taylor_maccoll_cone_angle_local(Ma, beta_deg, gamma)


def taylor_maccoll_cone_field(Ma, beta_deg, gamma=1.4):
    """Solve the T-M ODE once and return splines V_r(theta), V_theta(theta).

    Integrates from theta = beta (just behind the conical shock) inward,
    event-terminating at V_theta = 0 (cone body surface). The recorded
    trajectory is wrapped in scipy splines valid on theta in [delta_c, beta].

    Returns
    -------
    Vr_spline : callable theta_rad -> non-dim radial velocity component V_r
    Vt_spline : callable theta_rad -> non-dim tangential velocity component V_theta
    delta_c_rad : float, cone half-angle in radians (where V_theta = 0)
    beta_rad   : float, the input beta in radians (handy for callers)

    Notes
    -----
    V_r and V_theta are dimensionless (normalised by V_max as in Anderson §10).
    Only their *direction* is needed for streamline tracing in the osculating
    plane; the magnitude cancels in dr/dx = V_R/V_x.
    """
    from scipy.interpolate import UnivariateSpline

    beta = np.radians(beta_deg)
    # Post-shock initial conditions (same as _taylor_maccoll_cone_angle_local).
    d = np.arctan(
        2.0 / np.tan(beta)
        * (Ma ** 2 * np.sin(beta) ** 2 - 1.0)
        / (Ma ** 2 * (gamma + np.cos(2.0 * beta)) + 2.0)
    )
    if d <= 0.0:
        raise DetachedShockError(
            f"Ma = {Ma:.3f}, beta = {beta_deg:.3f} deg yields non-positive "
            "deflection; shock is below Mach angle."
        )
    Ma2 = 1.0 / np.sin(beta - d) * np.sqrt(
        (1.0 + (gamma - 1.0) / 2.0 * Ma ** 2 * np.sin(beta) ** 2)
        / (gamma * Ma ** 2 * np.sin(beta) ** 2 - (gamma - 1.0) / 2.0)
    )
    V0  = 1.0 / np.sqrt(2.0 / ((gamma - 1.0) * Ma2 ** 2) + 1.0)
    Vr0 =  V0 * np.cos(beta - d)
    Vt0 = -V0 * np.sin(beta - d)

    event = lambda t, y: _vt_event(t, y, gamma)
    event.terminal = True
    sol = solve_ivp(
        lambda t, y: _taylor_maccoll_rhs(t, y, gamma),
        (beta, 1e-4),
        [Vr0, Vt0],
        events=event,
        method="RK45",
        rtol=1e-6, atol=1e-9,
        dense_output=False,
    )
    if not sol.t_events[0].size:
        raise DetachedShockError(
            f"Taylor-Maccoll did not reach Vt = 0 for Ma = {Ma:.3f}, "
            f"beta = {beta_deg:.3f} deg."
        )
    delta_c_rad = float(sol.t_events[0][0])

    # solve_ivp with terminal event stops AT the event but does not always
    # include it as the last sample. Append the cone-surface point explicitly
    # (V_theta = 0 there) so the spline domain reaches all the way to delta_c.
    theta_pts = np.concatenate([sol.t, sol.t_events[0]])
    Vr_pts    = np.concatenate([sol.y[0], sol.y_events[0][:, 0]])
    Vt_pts    = np.concatenate([sol.y[1], sol.y_events[0][:, 1]])
    # Sort ascending in theta for the spline.
    order = np.argsort(theta_pts)
    theta_pts = theta_pts[order]
    Vr_pts    = Vr_pts[order]
    Vt_pts    = Vt_pts[order]
    # De-duplicate (UnivariateSpline requires strictly increasing x).
    keep = np.concatenate([[True], np.diff(theta_pts) > 1e-12])
    theta_pts = theta_pts[keep]
    Vr_pts    = Vr_pts[keep]
    Vt_pts    = Vt_pts[keep]

    k = min(3, theta_pts.size - 1)
    Vr_spline = UnivariateSpline(theta_pts, Vr_pts, k=k, s=0.0)
    Vt_spline = UnivariateSpline(theta_pts, Vt_pts, k=k, s=0.0)
    return Vr_spline, Vt_spline, delta_c_rad, float(beta)
