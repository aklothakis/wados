"""Mach-alpha sweep driver (GVWD §4.9, §5.5).

Headline output of the GVWD tab: a 2-D grid sweep over (M_inf, alpha)
producing a Pandas DataFrame of aerodynamic + heating coefficients.

For each (M_i, alpha_j) cell the driver:
  1. Runs the panel-method inviscid aero (panel_aero_coefficients).
  2. Adds the viscous correction (panel_viscous_drag).
  3. Computes peak LE heating (Tauber-Sutton swept) and nose heating.
  4. Records a shock-detachment margin diagnostic
     (min_attached_margin = theta_max - max(theta_local) over windward
     panels).

Default grid per spec: M in [5, 20] (8 points), alpha in [0, 15] deg
(6 points) -> 48 cells, completes in < 30 s on a modern CPU for
~5000-panel meshes.
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd

from gvwd.geometry.mesh import Mesh
from gvwd.aero.panel_method import (
    panel_aero_coefficients, freestream_direction,
)
from gvwd.aero.viscous import panel_viscous_drag, _us_std_1976
from gvwd.heating.fay_riddell import (
    swept_LE_heat_flux, nose_heat_flux,
)
from gvwd.thermo.oblique_shock import theta_max as _theta_max


@dataclass
class SweepConfig:
    """Inputs for :func:`mach_alpha_sweep`."""
    M_grid: tuple = (5.0, 20.0, 8)         # (M_min, M_max, n_M)
    alpha_grid_deg: tuple = (0.0, 15.0, 6)  # (alpha_min, alpha_max, n_alpha)
    altitude_km: float = 30.0
    T_w: float = 1500.0
    Re_x_tr: float = 1.0e6
    gamma: float = 1.4
    # LE / nose radii for heating evaluation (fixed across the sweep —
    # the heating numbers scale as 1/sqrt(R), so changing them is just a
    # post-hoc multiplication).
    r_LE: float = 5e-3
    r_nose: float = 10e-3
    Lambda_LE_rad: float = math.radians(75.0)
    # Reference geometry overrides
    S_ref: Optional[float] = None
    L_ref: Optional[float] = None
    x_ref: Optional[float] = None


def _expand_grid(t: tuple) -> np.ndarray:
    lo, hi, n = t
    return np.linspace(float(lo), float(hi), int(n))


def mach_alpha_sweep(mesh: Mesh, cfg: Optional[SweepConfig] = None,
                       *, on_cell: Optional[Callable] = None,
                       progress: bool = False) -> pd.DataFrame:
    """Run a 2-D (M, alpha) sweep.

    Parameters
    ----------
    mesh        : closed surface mesh
    cfg         : SweepConfig (default constructor used if None)
    on_cell     : optional callback(i, j, M, alpha_deg, row_dict) called
                  after every cell evaluation; useful for GUI progress.
    progress    : if True, print each row to stdout.

    Returns
    -------
    DataFrame with one row per (M, alpha) cell. Columns:
      M_inf, alpha_deg, CL, CD_total, CD_wave, CD_friction, Cm, LD,
      q_LE_swept_W_m2, q_nose_W_m2, beta_attached_margin_deg,
      Re_chord_max, delta_BL_max, regime_share_attached,
      regime_share_newtonian, regime_share_shadow.
    """
    cfg = cfg or SweepConfig()
    M_vals = _expand_grid(cfg.M_grid)
    alpha_vals = _expand_grid(cfg.alpha_grid_deg)

    # Atmosphere is fixed per sweep
    p_inf, T_inf = _us_std_1976(cfg.altitude_km)
    rho_inf = p_inf / (287.05 * T_inf)
    a_inf = math.sqrt(cfg.gamma * 287.05 * T_inf)

    rows = []
    for i, M in enumerate(M_vals):
        V_inf = M * a_inf
        # Heating depends on rho_inf and V_inf (no alpha-dependence in
        # the simple TS form), so compute once per Mach.
        q_LE = swept_LE_heat_flux(rho_inf, V_inf, cfg.r_LE,
                                    cfg.Lambda_LE_rad)
        q_nose = nose_heat_flux(rho_inf, V_inf, cfg.r_nose)
        # theta_max for shock-detachment check
        try:
            th_max = _theta_max(M, cfg.gamma)
        except Exception:
            th_max = math.nan

        for j, alpha_deg in enumerate(alpha_vals):
            alpha = math.radians(alpha_deg)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                inv = panel_aero_coefficients(
                    mesh, M, alpha, gamma=cfg.gamma,
                    S_ref=cfg.S_ref, L_ref=cfg.L_ref, x_ref=cfg.x_ref,
                )
                visc = panel_viscous_drag(
                    mesh, M, alpha,
                    altitude_km=cfg.altitude_km, T_w=cfg.T_w,
                    Re_x_tr=cfg.Re_x_tr,
                    p_inf=p_inf, T_inf=T_inf,
                    gamma=cfg.gamma, S_ref=inv.S_ref,
                )
            CD_total = inv.CD + visc.CD_friction
            LD = inv.CL / CD_total if CD_total > 1e-12 else math.inf

            # Shock-detachment margin = theta_max - max(theta_local)
            # over windward panels. Negative value => any panel with
            # theta_local exceeding theta_max (force fall-back to
            # Newtonian, regime tag = 2).
            v_inf = freestream_direction(alpha)
            n = mesh.face_normals()
            n_dot_v = np.clip(n @ v_inf, -1.0, 1.0)
            theta_local = -np.arcsin(n_dot_v)
            windward = theta_local > 1e-6
            if np.any(windward):
                max_th = float(theta_local[windward].max())
            else:
                max_th = 0.0
            margin_deg = math.degrees(th_max - max_th) if not math.isnan(th_max) \
                else math.nan

            # Regime share over windward panels
            n_win = int(windward.sum())
            n_attached = int(np.sum(inv.regime_code == 1))
            n_newt = int(np.sum(inv.regime_code == 2))
            n_shadow = int(np.sum(inv.regime_code == 0))
            n_total = max(int(inv.regime_code.size), 1)

            row = {
                "M_inf": float(M),
                "alpha_deg": float(alpha_deg),
                "CL": inv.CL,
                "CD_total": CD_total,
                "CD_wave": inv.CD,
                "CD_friction": visc.CD_friction,
                "Cm": inv.Cm,
                "LD": LD,
                "q_LE_swept_W_m2": float(q_LE),
                "q_nose_W_m2": float(q_nose),
                "q_LE_swept_MW_m2": float(q_LE / 1e6),
                "q_nose_MW_m2": float(q_nose / 1e6),
                "beta_attached_margin_deg": float(margin_deg),
                "Re_chord_max": float(visc.Re_chord_max),
                "delta_BL_max": float(visc.delta_BL_max),
                "regime_share_attached": n_attached / n_total,
                "regime_share_newtonian": n_newt / n_total,
                "regime_share_shadow": n_shadow / n_total,
                "S_ref": inv.S_ref,
                "L_ref": inv.L_ref,
            }
            rows.append(row)
            if on_cell is not None:
                on_cell(i, j, float(M), float(alpha_deg), row)
            if progress:
                print(f"M={M:5.2f} alpha={alpha_deg:5.2f}  "
                       f"CL={inv.CL:+.4f} CD_t={CD_total:.4f} "
                       f"L/D={LD:6.3f} q_LE={q_LE/1e6:.1f} MW/m^2  "
                       f"margin={margin_deg:+.2f} deg")

    df = pd.DataFrame(rows)
    df.attrs["altitude_km"] = cfg.altitude_km
    df.attrs["T_w"] = cfg.T_w
    df.attrs["r_LE"] = cfg.r_LE
    df.attrs["r_nose"] = cfg.r_nose
    df.attrs["Lambda_LE_deg"] = math.degrees(cfg.Lambda_LE_rad)
    df.attrs["p_inf"] = p_inf
    df.attrs["T_inf"] = T_inf
    return df


def heatmap_2d(df: pd.DataFrame, value_col: str) -> tuple:
    """Reshape a sweep DataFrame into 2-D arrays (M_grid, alpha_grid,
    Z) suitable for ``contourf`` / ``pcolormesh``."""
    M_vals = sorted(df["M_inf"].unique())
    a_vals = sorted(df["alpha_deg"].unique())
    Z = np.full((len(M_vals), len(a_vals)), np.nan)
    for _, row in df.iterrows():
        i = M_vals.index(float(row["M_inf"]))
        j = a_vals.index(float(row["alpha_deg"]))
        Z[i, j] = row[value_col]
    return np.array(M_vals), np.array(a_vals), Z
