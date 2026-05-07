"""Osculating-plane sweep for the MFOF framework.

Structurally identical to ``liu2019.osculating`` (legacy ``tan(delta_c)`` mode)
but takes a *flowfield factory* -- a callable ``(z, Ma_z) -> BasicFlowfield``
that decides which basic-flowfield type lives in each osculating plane. For
Phase 2 the factory always returns a :class:`ConeFlowfield`, so behaviour
matches Liu 2019 to machine precision.

Two call sites differ from ``liu2019.osculating``:

1. The cone half-angle is fetched from ``flowfield.deflection_angle_deg()``
   (instead of being looked up from a pre-computed ``delta_c(Ma)`` grid).
2. The compression-surface streamline is sampled by
   ``flowfield.trace_streamline()`` (instead of being inlined as a straight
   line in 3D).

All other geometry -- inward-normal computation, curvature radius, leading-edge
intersection (flat and curved branches), 3D back-projection -- is reused
verbatim from :mod:`liu2019.osculating` via direct import.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np

# Reuse the validated geometric helpers from liu2019. They have no
# cone-specific assumptions; they're pure (z, A, L_s) -> normal/radius/etc.
from liu2019.osculating import (
    _inward_normal,
    _solve_z_LE,
    curvature_radius_ICC,
)
from liu2019.distributions import (
    Ma_distribution,
    shock_curve,
    shock_curve_coefficient,
    upper_surface_coefficients,
    upper_surface_trailing_edge,
)
from liu2019.shock import taylor_maccoll_cone_angle as _tm_cone_angle

from .basic_flowfield import BasicFlowfield
from .cone_flowfield import ConeFlowfield


# ---------------------------------------------------------------------------
# Per-plane records
# ---------------------------------------------------------------------------

@dataclass
class OsculatingPlaneData:
    """One osculating-plane record. Carries both the spec field
    ``delta_deg`` and its ``delta_c`` alias so methods on
    :class:`liu2019.geometry.Liu2019Waverider` (which read ``p.delta_c``)
    work unchanged on MFOF data.
    """
    z:         float
    Ma:        float
    flowfield: BasicFlowfield
    delta_deg: float                                   # spec name
    delta_c:   float                                   # alias = delta_deg, for back-compat
    R_osc:     float
    n_base:    Tuple[float, float, float]
    P_shock:   Tuple[float, float, float]
    P_LE:      Tuple[float, float, float]
    P_TE:      Tuple[float, float, float]
    streamline: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3)))


@dataclass
class OsculatingPlaneSet:
    """Half-span (z >= 0) set of osculating-plane records plus derived coeffs."""
    planes: List[OsculatingPlaneData]
    coeffs: Dict[str, float]

    def __iter__(self):  return iter(self.planes)
    def __len__(self):   return len(self.planes)
    def __getitem__(self, idx): return self.planes[idx]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build_all_osculating_planes(
    params: Dict,
    flowfield_factory: Callable[[float, float], BasicFlowfield],
    n_z: int = 200,
    n_x_per_strip: int = 100,
) -> OsculatingPlaneSet:
    """Sweep z in ``[0, +W/2]`` and build :class:`OsculatingPlaneData` per station.

    Parameters
    ----------
    params : dict
        Liu-style design dict (``beta_deg``, ``L_w``, ``W``, ``L_s``,
        ``y5``, ``z5``, ``y6``, ``z6``, ``delta5``, ``delta6``,
        ``Ma_center``, ``Ma_tip``, ``gamma``).
    flowfield_factory : callable(z, Ma_z) -> BasicFlowfield
        Returns a basic flowfield instance for the osculating plane at
        spanwise station ``z`` with local design Mach ``Ma_z``. For Liu 2019
        reproduction this returns ``ConeFlowfield(Ma_z, beta_deg, gamma)``.
    n_z : int
        Number of spanwise stations (half-span). Mirror later in the
        geometry assembly.
    n_x_per_strip : int
        Streamline-sample count per plane.
    """
    beta_deg  = float(params["beta_deg"])
    L_w       = float(params["L_w"])
    W         = float(params["W"])
    L_s       = float(params["L_s"])
    y5        = float(params["y5"])
    z5        = float(params["z5"])
    y6        = float(params["y6"])
    z6        = float(params["z6"])
    delta5    = float(params["delta5"])
    delta6    = float(params["delta6"])
    Ma_center = float(params["Ma_center"])
    Ma_tip    = float(params["Ma_tip"])
    gamma     = float(params.get("gamma", 1.4))

    a, b, c, d = upper_surface_coefficients(
        y5, z5, y6, z6, delta5, delta6, L_w, beta_deg)
    A = shock_curve_coefficient(y5, z5, L_s)
    coeffs = {"a": a, "b": b, "c": c, "d": d, "A": A}

    z_stations = np.linspace(0.0, W / 2.0, int(n_z))
    planes: List[OsculatingPlaneData] = []

    beta_rad = np.radians(beta_deg)
    tan_beta = np.tan(beta_rad)

    # ---- Liu-compatibility delta_c grid -----------------------------------
    # liu2019.osculating uses a 15-point Ma -> delta_c interpolation table
    # (a perf shortcut over per-plane T-M solves). For the all-cone factory
    # case to match Liu byte-for-byte we mirror that grid here, then
    # override the lazy-cached delta on each ConeFlowfield with the
    # interpolated value before the sweep queries it. Other BasicFlowfield
    # subclasses (future power-law, wedge, ...) are queried as-is.
    Ma_lo = float(min(Ma_center, Ma_tip))
    Ma_hi = float(max(Ma_center, Ma_tip))
    Ma_grid = np.linspace(Ma_lo, Ma_hi, 15)
    delta_grid = np.array(
        [_tm_cone_angle(m, beta_deg, gamma) for m in Ma_grid])

    def _interp_cone(Ma):
        return float(np.interp(Ma, Ma_grid, delta_grid))

    for z in z_stations:
        Ma_local = float(Ma_distribution(z, W, Ma_center, Ma_tip))

        # Per-plane flowfield supplied by the caller (cone for Phase 2).
        flowfield = flowfield_factory(float(z), Ma_local)
        if isinstance(flowfield, ConeFlowfield):
            # Match liu2019's interpolated delta_c for bitwise equivalence.
            flowfield._delta_c_deg = _interp_cone(Ma_local)
        delta_deg = float(flowfield.deflection_angle_deg())

        # Geometric quantities common to all flowfield types -- reused from liu2019.
        n_base = _inward_normal(z, A, L_s)
        R_osc  = curvature_radius_ICC(z, A, L_s)
        flat   = (abs(z) <= L_s) or not np.isfinite(R_osc)

        # --- Leading-edge position (identical to liu2019 dispatcher) -----
        if flat:
            y_upper = float(upper_surface_trailing_edge(
                np.array([z]), a, b, c, d)[0])
            y_s_val = 0.0
            x_LE = L_w - (y_upper - y_s_val) / tan_beta
            y_LE = y_upper
            z_LE = float(z)
            r_LE_local = 0.0          # local-frame anchor; only descent matters
        else:
            y_s_val = float(shock_curve(np.array([z]), A, L_s)[0])
            P_s = np.array([L_w, y_s_val, float(z)])
            C_base = P_s + R_osc * n_base
            y_c, z_c = float(C_base[1]), float(C_base[2])
            n_y_, n_z_ = float(n_base[1]), float(n_base[2])
            z_LE = _solve_z_LE(z, coeffs, y_c, z_c, n_y_, n_z_)
            y_LE = float(upper_surface_trailing_edge(
                np.array([z_LE]), a, b, c, d)[0])
            r_LE_local = float(np.sqrt(
                (y_LE - y_c) ** 2 + (z_LE - z_c) ** 2))
            x_LE = L_w - (R_osc - r_LE_local) / tan_beta

        # --- Streamline via the flowfield ---
        sl = flowfield.trace_streamline(
            x_LE, r_LE_local, L_w, n_points=int(n_x_per_strip))

        # In-plane descent (positive downstream of the LE) and 3D
        # back-projection along the inward normal.
        descent = r_LE_local - sl.r_arr            # shape (n_x_per_strip,)
        n_y = float(n_base[1])
        n_z = float(n_base[2])
        x_3d = sl.x_arr
        y_3d = y_LE - descent * n_y
        z_3d = z_LE - descent * n_z
        stream_3d = np.column_stack([x_3d, y_3d, z_3d])

        x_TE = float(x_3d[-1])
        y_TE = float(y_3d[-1])
        z_TE = float(z_3d[-1])

        # --- Shock footprint (identical to liu2019) ----------------------
        y_shock_val = float(shock_curve(np.array([z]), A, L_s)[0])
        P_shock = (L_w, y_shock_val, float(z))

        planes.append(OsculatingPlaneData(
            z=float(z),
            Ma=Ma_local,
            flowfield=flowfield,
            delta_deg=delta_deg,
            delta_c=delta_deg,                     # back-compat alias
            R_osc=float(R_osc),
            n_base=(float(n_base[0]),
                    float(n_base[1]),
                    float(n_base[2])),
            P_shock=P_shock,
            P_LE=(float(x_LE), float(y_LE), float(z_LE)),
            P_TE=(x_TE, y_TE, z_TE),
            streamline=stream_3d,
        ))

    return OsculatingPlaneSet(planes=planes, coeffs=coeffs)
