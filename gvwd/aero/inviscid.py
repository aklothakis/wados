"""Closed-form inviscid aerodynamic coefficients for GVWD reference modes
(spec §4.6, §4.8). Reference area is the full-vehicle planform area.

For the caret and flat-delta on-design (alpha=0, M=M_design), Cp on the
lower surface is uniform at the wedge value and Cp on the upper surface
is zero. For the multi-wedge, the lower surface has piecewise-uniform Cp
per ramp.
"""

from __future__ import annotations

import math
from typing import Dict

from gvwd.geometry.caret import Caret
from gvwd.geometry.flat_delta import FlatDelta
from gvwd.geometry.multi_wedge import MultiWedge
from gvwd.thermo.oblique_shock import (
    cp_attached_wedge, swept_oblique_shock,
)


# ----------------------------------------------------------------------
#  Caret (§4.6)
# ----------------------------------------------------------------------

def caret_inviscid_coefficients(c: Caret) -> Dict[str, float]:
    """On-design (alpha=0, M=M_design) coefficients for a Nonweiler caret.

    Both lower-surface panels are wedges at angle theta_d; each carries
    uniform Cp = 4(M^2 sin^2 beta_d - 1) / ((gamma+1) M^2). The upper
    surface is freestream (Cp = 0).

    Force decomposition (per spec §4.8):
        Lift   per unit q_inf   = Cp * S_planform
        Drag (wave) per unit q  = Cp * S_planform * tan(theta_d)
        L/D    = 1 / tan(theta_d)

    Reference area S_ref = S_planform (full vehicle).
    """
    Cp = cp_attached_wedge(c.M_design, c.beta_d, c.gamma)
    # Planform area: the caret has a triangular planform from apex to
    # wingtips at (L, +-y_tip).  S_planform = L * (2 * y_tip) / 2 = L * y_tip.
    S_planform = c.L * c.y_tip
    L_over_q = Cp * S_planform
    D_over_q = Cp * S_planform * math.tan(c.theta_d)

    CL = L_over_q / S_planform                          # = Cp
    CD = D_over_q / S_planform                          # = Cp tan(theta_d)
    LD = 1.0 / math.tan(c.theta_d) if c.theta_d > 0 else math.inf

    # Pitching moment about x = L/2, normalised by S * L
    # Lower-surface pressure acts at chord midpoint at each spanwise
    # station; integral reduces to 0 by symmetry of the caret about x=L/2.
    Cm = 0.0   # caret on-design Cm about midchord is zero by symmetry

    return {
        "CL": CL, "CD": CD, "Cm": Cm, "LD": LD,
        "Cp_lower": Cp, "S_ref": S_planform,
        "regime": "on_design_attached",
    }


# ----------------------------------------------------------------------
#  Flat-bottomed delta (§4.6)
# ----------------------------------------------------------------------

def flat_delta_inviscid_coefficients(fd: FlatDelta) -> Dict[str, float]:
    """On-design coefficients for a flat-bottomed delta-planform wedge.

    The lower surface is a single inclined plane at theta_d. Surface Cp is
    the standard 2-D body-frame attached oblique shock at full M_design:
        Cp = 4 (M^2 sin^2 beta - 1) / ((gamma+1) M^2)
    The LE sweep affects only LE-attachment validity (M cos Lambda > 1)
    and LE heating, not surface pressure on a flat panel.
    """
    Cp = cp_attached_wedge(fd.M_design, fd.beta_body, fd.gamma)
    S_planform = fd.L * fd.y_tip
    L_over_q = Cp * S_planform
    D_over_q = Cp * S_planform * math.tan(fd.theta_d)

    CL = L_over_q / S_planform
    CD = D_over_q / S_planform
    LD = 1.0 / math.tan(fd.theta_d) if fd.theta_d > 0 else math.inf
    Cm = 0.0

    return {
        "CL": CL, "CD": CD, "Cm": Cm, "LD": LD,
        "Cp_lower": Cp, "S_ref": S_planform,
        "M_perp": fd.M_design * math.cos(fd.Lambda),
        "beta_body_deg": math.degrees(fd.beta_body),
        "regime": "on_design_attached_body",
    }


# ----------------------------------------------------------------------
#  Multi-wedge (§4.6)
# ----------------------------------------------------------------------

def multi_wedge_inviscid_coefficients(mw: MultiWedge) -> Dict[str, float]:
    """On-design coefficients for an Oswatitsch n-ramp wedge.

    Each ramp i has a uniform Cp_i derived from the LOCAL Mach M_{i-1}
    entering that ramp and the LOCAL deflection delta_inc_i. We integrate
    Cp_i over the streamwise extent of each ramp and project on lift /
    drag axes.

    Reference area: full body planform.
    """
    # Per-ramp Cp on the lower-surface ramps. The i-th ramp sees M_{i-1}
    # and turns by delta_inc_i; the SURFACE pressure coefficient is the
    # standard wedge result evaluated at the local Mach.
    Cps = []
    for i in range(mw.n):
        M_in = mw.osw.machs_after[i]
        beta_i = math.radians(mw.osw.betas_deg[i])
        Cps.append(cp_attached_wedge(M_in, beta_i, mw.gamma))

    # Each ramp has streamwise length L * frac_i and is inclined at the
    # cumulative angle delta_cum_i to freestream. The surface area of
    # ramp i (rectangular extrusion, full vehicle) is:
    #   A_i = (L * frac_i / cos(delta_cum_i)) * (2 * half_span)
    # Lift contribution = -Cp_i * A_i * cos(delta_cum_i)
    # Drag contribution = +Cp_i * A_i * sin(delta_cum_i)
    # which simplifies to Cp_i * (L frac_i) * (2 b) for lift,
    # and Cp_i * (L frac_i) * (2 b) tan(delta_cum_i) for wave drag.
    if mw.extrusion == "rectangular":
        half_span_eff = mw.half_span
    else:
        # For the delta extrusion, ramp width varies linearly along x.
        # Use the average width over each ramp segment.
        half_span_eff = mw.half_span * 0.5   # rough average

    L_q = 0.0; D_q = 0.0
    for i in range(mw.n):
        frac = mw.ramp_lengths_frac[i]
        delta_cum = math.radians(mw.osw.deltas_cum_deg[i])
        chord_i = mw.L * frac
        width_i = 2.0 * half_span_eff
        L_q += Cps[i] * chord_i * width_i                   # vertical force
        D_q += Cps[i] * chord_i * width_i * math.tan(delta_cum)

    S_planform = mw.L * 2.0 * half_span_eff   # rectangular projection
    CL = L_q / S_planform
    CD = D_q / S_planform
    LD = CL / CD if CD > 1e-12 else math.inf

    return {
        "CL": CL, "CD": CD, "Cm": 0.0, "LD": LD,
        "Cp_per_ramp": Cps,
        "S_ref": S_planform,
        "regime": "on_design_multi_ramp",
    }
