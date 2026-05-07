#!/usr/bin/env python
"""PSWR-1 Phase-5 production runner.

Usage:
    python examples/03_nsga2_run.py CONFIG.yaml [--pop N --gen N --seed N --tag T]

Loads a YAML config, runs NSGA-II, saves under results/run_<timestamp>_<tag>/,
generates Pareto, geometry-3D and sheath plots, evaluates the caret baseline,
and reports the >=6 dB / <15% L/D-loss DoD gate.
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path

import numpy as np

# Allow running as a script
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from pswr.io import load_config, save_run
from pswr.opt import PSWRProblem, evaluate_design, run_nsga2_pilot
from pswr.viz import (
    apply_style, plot_pareto_full, plot_design_geometry,
    plot_sheath_contour, plot_rcs_polar,
)
from pswr.geometry.variable_wedge import VariableWedgeWaverider
from pswr.plasma.sheath import build_sheath_grid
from pswr.plasma.permittivity import susceptibility
from pswr.em.born_rcs import bistatic_rcs, bistatic_direction_from_angles, rcs_dBsm


def caret_baseline(cfg, beta_deg: float, Lambda_deg: float):
    """Evaluate a constant-beta caret reference at the same M/h/radar."""
    beta = math.radians(beta_deg)
    Lambda_rad = math.radians(Lambda_deg)
    res = evaluate_design(
        np.array([beta, beta, beta, Lambda_rad]),
        M_inf=cfg.M_inf, body_length=cfg.body_length, T_w=cfg.T_w,
        p_inf=cfg.p_inf, T_inf=cfg.T_inf,
        f0_Hz=cfg.f0_Hz, R_LE=cfg.R_LE, q_LE_max=cfg.q_LE_max,
        bistatic_angles_deg=cfg.bistatic_angles_deg, k_i_hat=cfg.k_i_hat,
        n_span_geom=cfg.n_span_geom, n_chord_geom=cfg.n_chord_geom,
        n_span_grid=cfg.n_span_grid, n_chord_grid=cfg.n_chord_grid,
        n_normal=cfg.n_normal,
    )
    return res


def _select_designs(F: np.ndarray):
    """Return indices of (best_LD, best_RCS, best_compromise)."""
    LD = -F[:, 0]; sig = F[:, 1]; eta = -F[:, 2]
    i_ld = int(np.argmax(LD))
    i_rcs = int(np.argmin(sig))
    # Compromise: rank-sum of normalized scores (higher LD + lower sigma + higher eta)
    def _norm(a, invert=False):
        if a.max() == a.min():
            return np.zeros_like(a)
        v = (a - a.min()) / (a.max() - a.min())
        return 1 - v if invert else v
    score = _norm(LD) + _norm(-sig) + _norm(eta)
    i_comp = int(np.argmax(score))
    return i_ld, i_rcs, i_comp


def main():
    ap = argparse.ArgumentParser(description="PSWR-1 NSGA-II runner")
    ap.add_argument("config", help="YAML config path")
    ap.add_argument("--pop", type=int, default=50)
    ap.add_argument("--gen", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260503)
    ap.add_argument("--tag", default="")
    ap.add_argument("--style", default="paper",
                    choices=["paper", "slide", "draft"])
    ap.add_argument("--caret-beta", type=float, default=20.0,
                    help="caret baseline beta in deg (default 20)")
    ap.add_argument("--caret-lambda", type=float, default=70.0,
                    help="caret baseline Lambda in deg (default 70)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f"Loaded config: M_inf={cfg.M_inf}, h_inf={cfg.p_inf:.1f}Pa, "
          f"f0={cfg.f0_Hz/1e9:.2f}GHz")
    print(f"NSGA-II: pop={args.pop} gen={args.gen} seed={args.seed}")

    problem = PSWRProblem(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_nsga2_pilot(problem, pop_size=args.pop, n_gen=args.gen,
                                  seed=args.seed, on_gen=None, verbose=False)
    artifact = save_run(cfg, result, tag=args.tag or Path(args.config).stem)
    print(f"Saved run -> {artifact.base_dir}")
    print(f"  evals={result.n_eval}  feasible={result.n_feasible}  "
          f"pareto={result.X.shape[0]}  t={result.wall_time_s:.1f}s")

    # Caret baseline
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cb = caret_baseline(cfg, args.caret_beta, args.caret_lambda)
    cF = cb.F.reshape(1, 3)
    print(f"Caret baseline (beta={args.caret_beta}, Lambda={args.caret_lambda}):")
    print(f"  L/D={-cb.F[0]:.3f}  sigma={cb.F[1]:+.2f} dBsm  eta_V={-cb.F[2]:.4f}")

    # ---- DoD: >=6 dB reduction at <15% L/D loss ----
    LD = -result.F[:, 0]
    sig_dBsm = result.F[:, 1]
    LD_caret = -cb.F[0]
    sig_caret = cb.F[1]
    LD_min_allowed = 0.85 * LD_caret
    sig_target = sig_caret - 6.0
    qualifying = (LD >= LD_min_allowed) & (sig_dBsm <= sig_target)
    n_qual = int(np.sum(qualifying))
    print(f"DoD gate: >=6 dB sigma_b reduction at <=15% L/D loss vs caret")
    print(f"  caret L/D={LD_caret:.3f}, sigma={sig_caret:+.2f} dBsm")
    print(f"  qualifying solutions: {n_qual}/{len(LD)}")

    # ---- Plots ----
    out_plots = artifact.plots_dir
    apply_style(args.style)

    i_ld, i_rcs, i_comp = _select_designs(result.F)
    highlight = result.F[i_comp].reshape(1, 3)
    cap = (f"M={cfg.M_inf:.1f}, f0={cfg.f0_Hz/1e9:.2f} GHz")
    plot_pareto_full(result.F, caret_F=cF, highlight_F=highlight,
                     out_dir=out_plots, prefix="pareto", caption=cap,
                     style=args.style)

    # Best designs visualisation
    for idx, tag in [(i_ld, "best_LD"), (i_rcs, "best_RCS"), (i_comp, "best_compromise")]:
        x = result.X[idx]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wr = VariableWedgeWaverider(
                M_inf=cfg.M_inf,
                beta_knots=(float(x[0]), float(x[1]), float(x[2])),
                Lambda=float(x[3]),
                body_length=cfg.body_length,
                n_span=cfg.n_span_geom, n_chord=cfg.n_chord_geom,
                T_inf=cfg.T_inf, p_inf=cfg.p_inf, gamma=cfg.gamma)
        fig = plot_design_geometry(wr, title=f"Geometry — {tag.replace('_', ' ')}",
                                    style=args.style)
        fig.savefig(out_plots / f"geometry_{tag}.png", dpi=200)
        fig.savefig(out_plots / f"geometry_{tag}.pdf")
        import matplotlib.pyplot as plt; plt.close(fig)

    # Sheath n_e for compromise design
    x_c = result.X[i_comp]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wr_c = VariableWedgeWaverider(
            M_inf=cfg.M_inf,
            beta_knots=(float(x_c[0]), float(x_c[1]), float(x_c[2])),
            Lambda=float(x_c[3]),
            body_length=cfg.body_length,
            n_span=cfg.n_span_geom, n_chord=cfg.n_chord_geom,
            T_inf=cfg.T_inf, p_inf=cfg.p_inf, gamma=cfg.gamma)
        grid_c = build_sheath_grid(wr_c, T_w=cfg.T_w,
                                    n_chord=cfg.n_chord_grid,
                                    n_span=cfg.n_span_grid,
                                    n_normal=cfg.n_normal)
    fig = plot_sheath_contour(grid_c,
                                title=fr"$n_e$ best-compromise design ({cap})",
                                style=args.style)
    fig.savefig(out_plots / "sheath_ne_best_compromise.pdf")
    fig.savefig(out_plots / "sheath_ne_best_compromise.png", dpi=200)

    # RCS polar for compromise vs caret
    omega_0 = 2.0 * math.pi * cfg.f0_Hz
    chi_c = susceptibility(grid_c.n_e, grid_c.n_neutral, grid_c.T, omega_0)
    pts = np.stack([grid_c.X, grid_c.Y, grid_c.Z], axis=-1)
    vols = grid_c.cell_volume
    theta_sweep = np.linspace(0, 360, 73)
    sigma_design = []
    for th in theta_sweep:
        ks = bistatic_direction_from_angles(cfg.k_i_hat, math.radians(th), 0.0)
        sigma_design.append(bistatic_rcs(cfg.k_i_hat, ks, omega_0, chi_c, pts, vols))
    sigma_design_dB = np.array([rcs_dBsm(s) for s in sigma_design])

    # Caret sheath (regenerate)
    cb_beta = math.radians(args.caret_beta)
    cb_Lambda = math.radians(args.caret_lambda)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wr_caret = VariableWedgeWaverider(
            M_inf=cfg.M_inf,
            beta_knots=(cb_beta, cb_beta, cb_beta), Lambda=cb_Lambda,
            body_length=cfg.body_length,
            n_span=cfg.n_span_geom, n_chord=cfg.n_chord_geom,
            T_inf=cfg.T_inf, p_inf=cfg.p_inf, gamma=cfg.gamma)
        grid_caret = build_sheath_grid(wr_caret, T_w=cfg.T_w,
                                        n_chord=cfg.n_chord_grid,
                                        n_span=cfg.n_span_grid,
                                        n_normal=cfg.n_normal)
    chi_caret = susceptibility(grid_caret.n_e, grid_caret.n_neutral,
                                 grid_caret.T, omega_0)
    pts_c = np.stack([grid_caret.X, grid_caret.Y, grid_caret.Z], axis=-1)
    vols_c = grid_caret.cell_volume
    sigma_caret = []
    for th in theta_sweep:
        ks = bistatic_direction_from_angles(cfg.k_i_hat, math.radians(th), 0.0)
        sigma_caret.append(bistatic_rcs(cfg.k_i_hat, ks, omega_0,
                                         chi_caret, pts_c, vols_c))
    sigma_caret_dB = np.array([rcs_dBsm(s) for s in sigma_caret])

    fig = plot_rcs_polar(theta_sweep, sigma_design_dB,
                         caret_dBsm=sigma_caret_dB,
                         title=rf"$\sigma_b(\theta_s)$ — {cap}",
                         style=args.style)
    fig.savefig(out_plots / "rcs_polar_best_compromise.pdf")
    fig.savefig(out_plots / "rcs_polar_best_compromise.png", dpi=200)

    print(f"Plots saved -> {out_plots}")
    return artifact, n_qual


if __name__ == "__main__":
    main()
