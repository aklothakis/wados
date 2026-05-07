#!/usr/bin/env python
"""PSWR-1 bistability diagnostic.

Run a moderate NSGA-II pilot at M=15 / X-band, capture the *entire* archive
of evaluated designs (dominated + infeasible included), and plot sigma_b
against the projections that could explain a bistable Pareto:

  (a) min(beta knots)            — Saha threshold story
  (b) max(beta knots)            — hot-station-dominates story
  (c) spline-min beta(eta)       — actual coldest streamline
  (d) Lambda                      — sweep / planform shape
  (e) max(beta) - min(beta)       — beta non-monotonicity / spline shape
  (f) max delta_BL                — sheath-depth grid-cell crossing artifact

If sigma_b shows a sharp jump in (a) or (b) the bistability is the Saha
threshold and the right Phase-6 move is to add f_0 as a design variable.
If the jump is in (f) the bistability is a top-hat-profile grid artifact and
the fix is the Crocco-Busemann sheath profile. If the jump is in (d) or (e)
something else is going on and we need to think harder.

Output: results/diagnostic_<timestamp>/sigma_vs_*.{pdf,png} plus a CSV of
the full archive.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from pswr.io import load_config
from pswr.opt import PSWRProblem, run_nsga2_pilot
from pswr.geometry.variable_wedge import BetaSpline, VariableWedgeWaverider
from pswr.aero.viscous import per_station_state
from pswr.thermo.oblique_shock import (
    rankine_hugoniot, mach_angle, detachment_beta,
)
from pswr.viz.style import apply_style


# ----------------------------------------------------------------------
#  Saha-onset threshold beta* at given M_inf, T_inf  (T_post crossing)
# ----------------------------------------------------------------------

def beta_for_T_post(M_inf: float, T_inf: float, T_post_target: float,
                    gamma: float = 1.4) -> float:
    """Inverse of Rankine-Hugoniot temperature rise for a target T_post."""
    from scipy.optimize import brentq
    mu = mach_angle(M_inf)
    bdet = detachment_beta(M_inf, gamma)

    def f(beta):
        rh = rankine_hugoniot(M_inf, beta, p_inf=1.0, T_inf=T_inf, gamma=gamma)
        return rh["T2"] - T_post_target

    a, b = mu + 1e-6, bdet - 1e-6
    if f(a) * f(b) > 0:
        return float("nan")
    return brentq(f, a, b, xtol=1e-9)


# ----------------------------------------------------------------------
#  Per-design diagnostic: spline-min, max delta_BL
# ----------------------------------------------------------------------

def _design_diagnostics(x: np.ndarray, cfg) -> dict:
    """Spline-min beta(eta) and max(delta_BL_base) for one design."""
    b0, b1, b2, Lam = float(x[0]), float(x[1]), float(x[2]), float(x[3])
    spline = BetaSpline(b0, b1, b2)
    eta = np.linspace(0.0, 1.0, 200)
    beta_eta = spline(eta)
    out = {
        "beta_min_spline_deg": math.degrees(float(beta_eta.min())),
        "beta_max_spline_deg": math.degrees(float(beta_eta.max())),
    }
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wr = VariableWedgeWaverider(
                M_inf=cfg.M_inf, beta_knots=(b0, b1, b2), Lambda=Lam,
                body_length=cfg.body_length,
                n_span=cfg.n_span_geom, n_chord=cfg.n_chord_geom,
                T_inf=cfg.T_inf, p_inf=cfg.p_inf, gamma=cfg.gamma,
            )
            state = per_station_state(wr, T_w=cfg.T_w)
        out["delta_BL_max_m"] = float(state["delta_BL_base"].max())
        out["delta_BL_at_max_beta"] = float(state["delta_BL_base"][np.argmax(wr.beta_y)])
    except Exception:
        out["delta_BL_max_m"] = float("nan")
        out["delta_BL_at_max_beta"] = float("nan")
    return out


# ----------------------------------------------------------------------
#  Plot helpers
# ----------------------------------------------------------------------

def _scatter(ax, x, y, feasible, *, xlabel, ylabel, threshold=None,
             threshold_label=""):
    feas = np.asarray(feasible, dtype=bool)
    inf = ~feas
    if np.any(inf):
        ax.scatter(x[inf], y[inf], s=8, c="#999999", alpha=0.5,
                    label=f"infeasible ({inf.sum()})", edgecolors='none')
    if np.any(feas):
        ax.scatter(x[feas], y[feas], s=10, c="#cb4b16", alpha=0.7,
                    label=f"feasible ({feas.sum()})", edgecolors='none')
    if threshold is not None and not np.isnan(threshold):
        ax.axvline(threshold, color="#268bd2", linestyle="--", linewidth=1.2,
                   label=threshold_label)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, frameon=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="examples/configs/m15_h30km_xband.yaml")
    ap.add_argument("--pop", type=int, default=30)
    ap.add_argument("--gen", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260503)
    ap.add_argument("--style", default="paper",
                    choices=["paper", "slide", "draft"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f"Config: M_inf={cfg.M_inf}, h={cfg.p_inf:.0f}Pa, "
          f"f0={cfg.f0_Hz/1e9:.2f} GHz")
    print(f"NSGA-II archive run: pop={args.pop} gen={args.gen} "
          f"-> {args.pop*args.gen} evals")

    problem = PSWRProblem(cfg, capture_archive=True)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_nsga2_pilot(problem, pop_size=args.pop, n_gen=args.gen,
                                  seed=args.seed, on_gen=None, verbose=False)
    dt = time.perf_counter() - t0
    print(f"NSGA-II done in {dt:.1f} s, archive size = {len(problem.archive_X)}")

    X = np.asarray(problem.archive_X)
    F = np.asarray(problem.archive_F)
    G = np.asarray(problem.archive_G)
    feasible = np.all(G <= 0.0, axis=1) & (F[:, 0] < 1e5)

    print(f"  feasible: {int(feasible.sum())}/{len(X)}  "
          f"({100*feasible.mean():.1f} %)")

    # Per-design diagnostics
    print("Computing spline-min and delta_BL diagnostics...")
    diag_min, diag_max, diag_dBL, diag_dBL_atmaxbeta = [], [], [], []
    for x in X:
        d = _design_diagnostics(x, cfg)
        diag_min.append(d["beta_min_spline_deg"])
        diag_max.append(d["beta_max_spline_deg"])
        diag_dBL.append(d["delta_BL_max_m"])
        diag_dBL_atmaxbeta.append(d["delta_BL_at_max_beta"])
    diag_min = np.array(diag_min); diag_max = np.array(diag_max)
    diag_dBL = np.array(diag_dBL); diag_dBL_atmaxbeta = np.array(diag_dBL_atmaxbeta)

    # Knot-based beta features
    beta_knots_deg = np.degrees(X[:, :3])
    beta_knot_min = beta_knots_deg.min(axis=1)
    beta_knot_max = beta_knots_deg.max(axis=1)
    beta_knot_range = beta_knot_max - beta_knot_min
    Lambda_deg = np.degrees(X[:, 3])
    # Flat-nose fraction is the 5th design variable (if present)
    flat_X1 = X[:, 4] if X.shape[1] >= 5 else np.zeros(len(X))
    sigma_dBsm = F[:, 1]
    LD = -F[:, 0]; eta_V = -F[:, 2]

    # Saha-onset thresholds vs T_post = 2500K (onset) and 3500K (strong)
    beta_onset = math.degrees(
        beta_for_T_post(cfg.M_inf, cfg.T_inf, 2500.0))
    beta_strong = math.degrees(
        beta_for_T_post(cfg.M_inf, cfg.T_inf, 3500.0))
    print(f"  Saha onset (T_post=2500K) at beta = {beta_onset:.2f} deg")
    print(f"  Saha strong (T_post=3500K) at beta = {beta_strong:.2f} deg")

    # ---- Plot ------------------------------------------------------
    apply_style(args.style)
    out_root = Path("results") / f"diagnostic_{time.strftime('%Y%m%dT%H%M%S')}"
    out_root.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 3, figsize=(13.0, 11.0))

    _scatter(axes[0, 0], beta_knot_min, sigma_dBsm, feasible,
             xlabel="min(beta knots) [deg]",
             ylabel=r"$\sigma_b$ [dBsm]",
             threshold=beta_onset,
             threshold_label=f"Saha onset T2=2500K (beta={beta_onset:.1f})")
    axes[0, 0].set_title("(a) sigma_b vs min(beta knots)")

    _scatter(axes[0, 1], beta_knot_max, sigma_dBsm, feasible,
             xlabel="max(beta knots) [deg]",
             ylabel=r"$\sigma_b$ [dBsm]",
             threshold=beta_onset,
             threshold_label=f"T2=2500K  ({beta_onset:.1f} deg)")
    axes[0, 1].axvline(beta_strong, color="#dc322f", linestyle=":",
                        linewidth=1.0,
                        label=f"T2=3500K ({beta_strong:.1f})")
    axes[0, 1].legend(loc="best", fontsize=8, frameon=False)
    axes[0, 1].set_title("(b) sigma_b vs max(beta knots)")

    _scatter(axes[0, 2], diag_min, sigma_dBsm, feasible,
             xlabel="spline-min beta(eta) [deg]",
             ylabel=r"$\sigma_b$ [dBsm]",
             threshold=beta_onset,
             threshold_label=f"Saha onset")
    axes[0, 2].set_title("(c) sigma_b vs spline-min beta")

    _scatter(axes[1, 0], Lambda_deg, sigma_dBsm, feasible,
             xlabel="Lambda [deg]",
             ylabel=r"$\sigma_b$ [dBsm]")
    axes[1, 0].set_title("(d) sigma_b vs Lambda")

    _scatter(axes[1, 1], beta_knot_range, sigma_dBsm, feasible,
             xlabel="max-min(beta knots) [deg]  (spline span)",
             ylabel=r"$\sigma_b$ [dBsm]")
    axes[1, 1].set_title("(e) sigma_b vs beta non-monotonicity")

    _scatter(axes[1, 2], diag_dBL_atmaxbeta * 1000, sigma_dBsm, feasible,
             xlabel=r"$\delta_{BL}$ at max-beta station [mm]",
             ylabel=r"$\sigma_b$ [dBsm]")
    axes[1, 2].set_title("(f) sigma_b vs sheath depth")

    # ---- New panel (g): sigma_b vs X1 (flat-nose fraction) -------
    _scatter(axes[2, 0], flat_X1, sigma_dBsm, feasible,
             xlabel="X1 (flat-nose fraction)",
             ylabel=r"$\sigma_b$ [dBsm]")
    axes[2, 0].set_title("(g) sigma_b vs X1 flat-nose")

    # ---- (h): sigma_b heatmap in (X1, max(beta knots)) plane -----
    ax = axes[2, 1]
    sc = ax.scatter(beta_knot_max[feasible], flat_X1[feasible],
                    c=sigma_dBsm[feasible], s=14, cmap="viridis",
                    edgecolor='none')
    if not math.isnan(beta_onset):
        ax.axvline(beta_onset, color="white", linestyle="--", linewidth=0.8)
    fig.colorbar(sc, ax=ax, label=r"$\sigma_b$ [dBsm]")
    ax.set_xlabel("max(beta knots) [deg]"); ax.set_ylabel("X1")
    ax.set_title("(h) joint  (max beta, X1) -> sigma_b")

    # ---- (i): X1 vs max(beta) showing where Pareto-optimal designs sit
    ax = axes[2, 2]
    # Identify non-dominated within feasible set
    feas_idx = np.where(feasible)[0]
    LD_f = LD[feas_idx]; sig_f = sigma_dBsm[feas_idx]
    nd_mask = np.array([
        not np.any((LD_f > LD_f[i]) & (sig_f < sig_f[i]))
        for i in range(len(feas_idx))
    ])
    ax.scatter(beta_knot_max[feasible], flat_X1[feasible],
               c="#999999", s=8, alpha=0.5, label=f"feasible ({feasible.sum()})")
    if nd_mask.any():
        ax.scatter(beta_knot_max[feas_idx[nd_mask]],
                   flat_X1[feas_idx[nd_mask]],
                   c="#cb4b16", s=22, edgecolor='k', linewidth=0.3,
                   label=f"Pareto ({nd_mask.sum()})")
    if not math.isnan(beta_onset):
        ax.axvline(beta_onset, color="#268bd2", linestyle="--", linewidth=1.0,
                    label=f"Saha onset ({beta_onset:.1f} deg)")
    ax.set_xlabel("max(beta knots) [deg]"); ax.set_ylabel("X1")
    ax.set_title("(i) Pareto location in (max beta, X1)")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"PSWR-1 archive diagnostic — M_inf={cfg.M_inf}, "
        f"f0={cfg.f0_Hz/1e9:.1f} GHz   "
        f"({len(X)} designs, {int(feasible.sum())} feasible)",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_root / "sigma_vs_diagnostics.pdf")
    fig.savefig(out_root / "sigma_vs_diagnostics.png", dpi=200)

    # CSV
    import csv
    with open(out_root / "archive.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "beta0_deg", "beta1_deg", "beta2_deg", "Lambda_deg", "X1_flat",
            "beta_min_knots_deg", "beta_max_knots_deg", "beta_range_deg",
            "spline_min_deg", "spline_max_deg",
            "delta_BL_max_mm", "delta_BL_at_maxbeta_mm",
            "LD", "sigma_dBsm", "eta_V", "feasible",
            "g1_detach", "g2_mach", "g3_heat", "g4_born",
        ])
        for i in range(len(X)):
            w.writerow([
                math.degrees(X[i, 0]), math.degrees(X[i, 1]),
                math.degrees(X[i, 2]), math.degrees(X[i, 3]),
                flat_X1[i],
                beta_knot_min[i], beta_knot_max[i], beta_knot_range[i],
                diag_min[i], diag_max[i],
                diag_dBL[i] * 1000, diag_dBL_atmaxbeta[i] * 1000,
                LD[i], sigma_dBsm[i], eta_V[i], int(feasible[i]),
                G[i, 0], G[i, 1], G[i, 2], G[i, 3],
            ])

    print(f"Saved -> {out_root}")
    return out_root


if __name__ == "__main__":
    main()
