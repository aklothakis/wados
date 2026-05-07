"""Shared run-from-config helper used by examples 01-06."""

from __future__ import annotations

import math
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

from gvwd.io.config import (
    GVWDRunConfig, load_config_yaml, build_geometry, config_sha256,
    EngineeringFlatConfig, EngineeringShallowVConfig,
)
from gvwd.io.results import write_run_artifact
from gvwd.aero.coefficients import aero_coefficients_full
from gvwd.aero.sweep import SweepConfig, mach_alpha_sweep


def run_from_config(cfg_path_or_obj,
                     *,
                     write_step: bool = True,
                     verbose: bool = True) -> dict:
    """Load a YAML config (or accept a GVWDRunConfig directly), build
    geometry, evaluate on-design + optional sweep, persist artifacts.

    Returns ``{"artifact": RunArtifact, "wall_time_s": float}``.
    """
    if isinstance(cfg_path_or_obj, GVWDRunConfig):
        cfg = cfg_path_or_obj
    else:
        cfg = load_config_yaml(cfg_path_or_obj)
    sha = config_sha256(cfg)
    if verbose:
        print(f"[gvwd] sha256 = {sha[:16]}...  tag = {cfg.tag or '(none)'}")

    t0 = time.perf_counter()

    # Geometry
    geom = build_geometry(cfg)
    if verbose:
        kind = cfg.geometry.mode
        print(f"[gvwd] geometry: {kind}, mesh = "
              f"{geom.mesh.n_vertices} verts, {geom.mesh.n_faces} faces")

    # Optional fins (engineering modes only)
    fins_mesh = None
    if cfg.fins.n_fins > 0 and isinstance(
            cfg.geometry, (EngineeringFlatConfig, EngineeringShallowVConfig)):
        from gvwd.geometry import FinParams, generate_fins, merge_meshes
        fp = FinParams(
            n_fins=cfg.fins.n_fins,
            root_chord=cfg.fins.root_chord, tip_chord=cfg.fins.tip_chord,
            span=cfg.fins.span,
            sweep_LE=math.radians(cfg.fins.sweep_LE_deg),
            dihedral=math.radians(cfg.fins.dihedral_deg),
            t_c=cfg.fins.t_c,
            max_thickness_loc=cfg.fins.max_thickness_loc,
            LE_style=cfg.fins.LE_style,
            LE_radius=cfg.fins.LE_radius_mm * 1e-3,
            attach_x_frac=cfg.fins.attach_x_frac,
        )
        fins_mesh = generate_fins(
            fp, attach_xyz=(geom.L_fore + geom.L_center * fp.attach_x_frac,
                              0.0, -0.05),
        )
        if fins_mesh is not None:
            geom_mesh = merge_meshes([geom.mesh, fins_mesh])
            if verbose:
                print(f"[gvwd] fins: {cfg.fins.n_fins}-fin assembly "
                      f"(+{fins_mesh.n_faces} faces)")
        else:
            geom_mesh = geom.mesh
    else:
        geom_mesh = geom.mesh

    # On-design aero (M_design at alpha=0). Skip for multi-wedge (no
    # natural alpha for that geometry).
    on_design = None
    M_design = getattr(cfg.geometry, "M_design", None)
    if M_design is not None and cfg.geometry.mode != "multi_wedge":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            on_design = aero_coefficients_full(
                geom_mesh, M_inf=float(M_design), alpha_rad=0.0,
                altitude_km=cfg.sweep.altitude_km,
                T_w=cfg.sweep.T_w,
                Re_x_tr=cfg.sweep.Re_x_tr,
            )
            on_design_clean = {
                k: v for k, v in on_design.items()
                if k not in ("panel", "viscous")
            }
        if verbose:
            print(f"[gvwd] on-design (M={M_design}, alpha=0): "
                  f"CL={on_design['CL']:.4f}, CD={on_design['CD_total']:.4f}, "
                  f"L/D={on_design['LD']:.3f}")

    # Optional sweep
    sweep_df = None
    if cfg.sweep.enabled:
        if verbose:
            n = cfg.sweep.M_grid[2] * cfg.sweep.alpha_grid_deg[2]
            print(f"[gvwd] sweep: {cfg.sweep.M_grid[2]}x{cfg.sweep.alpha_grid_deg[2]}={n} cells...")
        sw_cfg = SweepConfig(
            M_grid=tuple(cfg.sweep.M_grid),
            alpha_grid_deg=tuple(cfg.sweep.alpha_grid_deg),
            altitude_km=cfg.sweep.altitude_km,
            T_w=cfg.sweep.T_w,
            Re_x_tr=cfg.sweep.Re_x_tr,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sweep_df = mach_alpha_sweep(geom_mesh, sw_cfg)
        if verbose:
            print(f"[gvwd]   L/D range {sweep_df['LD'].min():.2f}-"
                  f"{sweep_df['LD'].max():.2f}, "
                  f"q_LE max {sweep_df['q_LE_swept_MW_m2'].max():.1f} MW/m^2")

    # Persist
    on_design_for_save = (
        {k: (v if not hasattr(v, "tolist") else v.tolist())
          for k, v in (on_design or {}).items()
          if k not in ("panel", "viscous")}
        if on_design is not None else None
    )
    art = write_run_artifact(
        cfg, on_design=on_design_for_save, sweep_df=sweep_df,
        mesh=geom_mesh,
        write_step=write_step,
    )

    # Plots if a sweep ran
    if sweep_df is not None:
        from gvwd.viz import plot_full_sweep_suite
        plot_full_sweep_suite(sweep_df, out_dir=art.base_dir / "plots")

    dt = time.perf_counter() - t0
    if verbose:
        print(f"[gvwd] total wall time: {dt:.2f} s")
        print(f"[gvwd] artifacts -> {art.base_dir}")
    return {"artifact": art, "wall_time_s": dt}
