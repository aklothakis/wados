"""Per-design evaluator: full geometry -> Saha -> Drude -> Born pipeline.

This module implements ``evaluate_design(x, cfg)`` returning the multi-objective
vector ``F`` and constraint vector ``G`` consumed by the pymoo Problem in
:mod:`pswr.opt.problem`.

Objective vector (minimization form, PSWR-1 §5.7):
    F = ( -L/D,  max_k sigma_b,k [dBsm],  -eta_V )

Constraint vector (g(x) <= 0):
    g1 = (beta_detach(M_inf) - margin) - min beta(y)     # no detachment
    g2 = (mach_angle(M_inf) + margin) - min beta(y)      # supersonic shock
    g3 = q_FR(x) - q_LE_max                              # Fay-Riddell heating
    g4 = max|Re chi(x)| - 0.3                            # Born validity
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

from ..geometry.variable_wedge import VariableWedgeWaverider
from ..geometry.volume import volume_efficiency
from ..thermo.oblique_shock import mach_angle, detachment_beta
from ..aero.inviscid import inviscid_coefficients
from ..aero.viscous import viscous_drag_coefficient
from ..plasma.sheath import build_sheath_grid
from ..plasma.permittivity import susceptibility, born_validity
from ..em.born_rcs import (
    bistatic_rcs,
    bistatic_direction_from_angles,
    rcs_dBsm,
)


# ----------------------------------------------------------------------
#  Fay-Riddell stagnation-line heating (Sutton-Graves form, swept LE)
# ----------------------------------------------------------------------

def fay_riddell_swept(M_inf: float, p_inf: float, T_inf: float,
                       R_LE: float, Lambda_rad: float,
                       gamma: float = 1.4) -> float:
    """Stagnation-line heat flux for a swept cylindrical leading edge [W/m^2].

    Uses Sutton-Graves coefficient (K = 1.7415e-4 in CGS-mixed units;
    in SI the result in W/m^2 is K_SI * sqrt(rho/R) * V^3 with K_SI = 1.7415).
    Swept-cylinder correction: q_swept = q_unswept * sqrt(cos(Lambda)) per
    Beckwith & Cohen (high Re_theta).

        q_FR_swept = K_SI * sqrt(rho_inf / R_LE) * V_inf^3 * sqrt(cos(Lambda))

    Returns 0 if Lambda >= 90 deg (degenerate).
    """
    R = 287.05
    rho_inf = p_inf / (R * T_inf)
    a_inf = math.sqrt(gamma * R * T_inf)
    V_inf = M_inf * a_inf
    cos_L = math.cos(Lambda_rad)
    if cos_L <= 0.0 or R_LE <= 0.0:
        return 0.0
    K_SI = 1.7415   # Sutton-Graves in SI, W m^(-7/2) s^3 kg^(-1/2)
    q_unswept = K_SI * math.sqrt(rho_inf / R_LE) * V_inf ** 3
    return q_unswept * math.sqrt(cos_L)


# ----------------------------------------------------------------------
#  Per-design evaluator
# ----------------------------------------------------------------------

@dataclass
class DesignResult:
    F: np.ndarray            # objective vector (3,)
    G: np.ndarray            # constraint vector (4,)
    raw: dict                # diagnostics: CL, CD, eta_V, sigma_b list, max|Re chi|, ...
    feasible: bool


_BIG = 1.0e6


def evaluate_design(x: np.ndarray, *,
                    M_inf: float,
                    body_length: float,
                    T_w: float,
                    p_inf: float,
                    T_inf: float,
                    f0_Hz: float,
                    R_LE: float,
                    q_LE_max: float,
                    bistatic_angles_deg: Iterable[Tuple[float, float]],
                    k_i_hat: Tuple[float, float, float],
                    n_span_geom: int = 41,
                    n_chord_geom: int = 20,
                    n_span_grid: int = 21,
                    n_chord_grid: int = 30,
                    n_normal: int = 10,
                    margin_detach_rad: float = math.radians(1.0),
                    margin_mach_rad: float = math.radians(1.0),
                    gamma: float = 1.4,
                    ) -> DesignResult:
    """Evaluate one design vector.

    x has length 4 or 5. The first four entries are
    (beta0, beta1, beta2, Lambda) in radians. If x has length 5, x[4] is the
    flat-nose fraction X1 (centerline blunt strip, dimensionless in [0, 1)).
    """
    beta0 = float(x[0]); beta1 = float(x[1]); beta2 = float(x[2])
    Lambda = float(x[3])
    flat_fraction = float(x[4]) if len(x) >= 5 else 0.0
    omega_0 = 2.0 * math.pi * f0_Hz

    raw: dict = {"x": tuple(x)}

    # ---- (1) Geometry: try to build, catch detachment/mach-angle failures
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            wr = VariableWedgeWaverider(
                M_inf=M_inf,
                beta_knots=(beta0, beta1, beta2),
                Lambda=Lambda,
                body_length=body_length,
                flat_fraction=flat_fraction,
                n_span=n_span_geom, n_chord=n_chord_geom,
                gamma=gamma, T_inf=T_inf, p_inf=p_inf,
            )
    except Exception as e:
        raw["error"] = f"geometry: {e}"
        F = np.array([_BIG, _BIG, _BIG])
        G = np.array([_BIG, _BIG, _BIG, _BIG])
        return DesignResult(F=F, G=G, raw=raw, feasible=False)

    # ---- (2) Aero: inviscid + viscous
    try:
        inv = inviscid_coefficients(wr)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            visc = viscous_drag_coefficient(wr, T_w=T_w)
        CL = inv["CL"]; CD_wave = inv["CD"]; CD_fric = visc["CD_friction"]
        CD = CD_wave + CD_fric
        if CD <= 0 or not math.isfinite(CD):
            raise ValueError(f"non-physical CD={CD}")
        LD = CL / CD
        eta_V = volume_efficiency(wr)
    except Exception as e:
        raw["error"] = f"aero: {e}"
        F = np.array([_BIG, _BIG, _BIG]); G = np.array([_BIG]*4)
        return DesignResult(F=F, G=G, raw=raw, feasible=False)

    # ---- (3) Plasma sheath
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            grid = build_sheath_grid(
                wr, T_w=T_w,
                n_chord=n_chord_grid, n_span=n_span_grid, n_normal=n_normal,
            )
        chi = susceptibility(grid.n_e, grid.n_neutral, grid.T, omega_0)
        born_ok, max_re_chi = born_validity(chi)
    except Exception as e:
        raw["error"] = f"plasma: {e}"
        F = np.array([_BIG, _BIG, _BIG]); G = np.array([_BIG]*4)
        return DesignResult(F=F, G=G, raw=raw, feasible=False)

    # ---- (4) Born RCS at three prescribed angles
    try:
        points = np.stack([grid.X, grid.Y, grid.Z], axis=-1)
        volumes = grid.cell_volume
        sigmas = []
        for theta_deg, phi_deg in bistatic_angles_deg:
            k_s = bistatic_direction_from_angles(
                k_i_hat, math.radians(theta_deg), math.radians(phi_deg))
            sigmas.append(bistatic_rcs(k_i_hat, k_s, omega_0,
                                        chi, points, volumes))
        sigmas_dBsm = [rcs_dBsm(s) for s in sigmas]
        max_sigma_dBsm = max(sigmas_dBsm)
    except Exception as e:
        raw["error"] = f"rcs: {e}"
        F = np.array([_BIG, _BIG, _BIG]); G = np.array([_BIG]*4)
        return DesignResult(F=F, G=G, raw=raw, feasible=False)

    # ---- (5) Constraints
    mu = mach_angle(M_inf)
    beta_det = detachment_beta(M_inf, gamma)
    min_beta = min(beta0, beta1, beta2)
    g1 = (beta_det - margin_detach_rad) - min_beta   # < 0 if min_beta below (beta_det - margin)
    # Reformulate: detachment violation = beta - (beta_det - margin) > 0  =>  g1 = beta_max - (beta_det - margin)
    max_beta = max(beta0, beta1, beta2)
    g1 = max_beta - (beta_det - margin_detach_rad)
    g2 = (mu + margin_mach_rad) - min_beta            # > 0 if min_beta < mu + margin
    q_FR = fay_riddell_swept(M_inf, p_inf, T_inf, R_LE, Lambda, gamma=gamma)
    g3 = q_FR - q_LE_max                                # > 0 if heating exceeds limit
    g4 = max_re_chi - 0.3                                # > 0 if Born suspect

    G = np.array([g1, g2, g3, g4], dtype=float)
    feasible = bool(np.all(G <= 0))

    # ---- (6) Objectives
    F = np.array([-LD, max_sigma_dBsm, -eta_V], dtype=float)

    raw.update(dict(
        CL=CL, CD=CD, CD_wave=CD_wave, CD_fric=CD_fric, LD=LD,
        eta_V=eta_V, sigmas_m2=sigmas, sigmas_dBsm=sigmas_dBsm,
        max_sigma_dBsm=max_sigma_dBsm,
        max_re_chi=max_re_chi, born_valid=born_ok,
        q_FR_W_m2=q_FR, q_FR_MW_m2=q_FR/1e6,
        beta_knots_deg=(math.degrees(beta0), math.degrees(beta1),
                         math.degrees(beta2)),
        Lambda_deg=math.degrees(Lambda),
        flat_fraction=flat_fraction,
        beta_det_deg=math.degrees(beta_det),
        mu_deg=math.degrees(mu),
    ))
    return DesignResult(F=F, G=G, raw=raw, feasible=feasible)
