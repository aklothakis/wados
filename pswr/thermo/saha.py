"""Saha-Boltzmann LTE solver for 7-species air (PSWR-1 §5.3).

Species: {N2, O2, N, O, NO, NO+, e-}.

Reduction strategy. The 4 mass-action equilibria + charge neutrality express
five species in terms of (n_N, n_O):

    n_N2  = n_N^2 / K_N2_dis
    n_O2  = n_O^2 / K_O2_dis
    n_NO  = n_N * n_O / K_NO_dis
    n_NO+ = n_e          (charge neutrality)
    n_e   = sqrt(K_ion * n_NO)        (Saha for NO -> NO+ + e-)

That leaves a 2-equation residual on (n_N, n_O):
    R1 : nuclei mole-fraction ratio  (n_N_atoms / n_O_atoms = X_N / X_O)
    R2 : total pressure              (sum n_i = p / (k_B T))

Solved with :func:`scipy.optimize.fsolve` on log variables. A bisection
fallback on log10(n_N) is provided for failed convergence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import fsolve, brentq

from .species import (
    K_B, EV,
    SPECIES, REACTIONS, X_N, X_O,
    reaction_K, all_reaction_K,
)


# ----------------------------------------------------------------------
#  Public solver
# ----------------------------------------------------------------------

@dataclass
class SahaResult:
    n: Dict[str, float]   # number densities, m^-3
    T: float
    p: float
    K: Dict[str, float]
    converged: bool
    method: str
    nfev: int

    @property
    def n_e(self) -> float:
        return self.n["e-"]

    @property
    def n_total(self) -> float:
        return sum(self.n.values())

    @property
    def alpha_N2(self) -> float:
        n_N = self.n["N"]; n_N2 = self.n["N2"]
        denom = n_N + 2.0 * n_N2
        return n_N / denom if denom > 0 else 0.0

    @property
    def alpha_O2(self) -> float:
        n_O = self.n["O"]; n_O2 = self.n["O2"]
        denom = n_O + 2.0 * n_O2
        return n_O / denom if denom > 0 else 0.0

    @property
    def x_e(self) -> float:
        n_tot = self.n_total
        return self.n["e-"] / n_tot if n_tot > 0 else 0.0


# ----------------------------------------------------------------------
#  Reduced 2-D system
# ----------------------------------------------------------------------

def _close_from_NO(n_N: float, n_O: float, K: Dict[str, float]) -> Dict[str, float]:
    """Given (n_N, n_O), return all 7 number densities via mass-action + Saha."""
    with np.errstate(over="ignore", invalid="ignore"):
        n_N2 = n_N * n_N / K["N2_dissociation"]
        n_O2 = n_O * n_O / K["O2_dissociation"]
        n_NO = n_N * n_O / K["NO_dissociation"]
    n_e = math.sqrt(max(K["NO_ionization"] * n_NO, 0.0))
    n_NOp = n_e
    return {"N2": n_N2, "O2": n_O2, "N": n_N, "O": n_O,
            "NO": n_NO, "NO+": n_NOp, "e-": n_e}


def _residuals_2d(x: np.ndarray, T: float, p: float,
                  K: Dict[str, float]) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        n_N, n_O = np.exp(np.clip(x, -700.0, 700.0))
        n = _close_from_NO(n_N, n_O, K)
        nN_atoms = n["N"] + 2.0 * n["N2"] + n["NO"] + n["NO+"]
        nO_atoms = n["O"] + 2.0 * n["O2"] + n["NO"] + n["NO+"]
        total = sum(n.values())
    n_target = p / (K_B * T)

    norm_atom = max(nN_atoms + nO_atoms, 1e-30)
    if not (math.isfinite(nN_atoms) and math.isfinite(nO_atoms)
            and math.isfinite(total)):
        return np.array([1e30, 1e30])
    R1 = (nN_atoms * X_O - nO_atoms * X_N) / norm_atom
    R2 = (total - n_target) / max(n_target, 1e-30)
    return np.array([R1, R2])


def _initial_guess_2d(T: float, p: float) -> np.ndarray:
    """Smooth temperature-dependent initial guess for log(n_N), log(n_O)."""
    n_target = p / (K_B * T)
    # Center sigmoids at temperatures where dissociation reaches ~50% at 1 atm
    a_N2 = 1.0 / (1.0 + math.exp(-(T - 7000.0) / 700.0))
    a_O2 = 1.0 / (1.0 + math.exp(-(T - 3700.0) / 500.0))
    n_N0 = max(a_N2 * 2.0 * X_N * n_target, 1.0)
    n_O0 = max(a_O2 * 2.0 * X_O * n_target, 1.0)
    return np.log([n_N0, n_O0])


def solve_saha_lte(T: float, p: float, *,
                   xtol: float = 1e-12,
                   maxfev: int = 5000) -> SahaResult:
    """LTE solve at given (T_K, p_Pa) -> :class:`SahaResult`."""
    if T <= 0 or p <= 0:
        raise ValueError(f"T and p must be positive; got T={T}, p={p}")

    K = all_reaction_K(T)

    # Low-T short circuit: below ~1500 K dissociation/ionization are
    # exponentially suppressed (Boltzmann factors < 1e-30). Skip the
    # nonlinear solve and return frozen-air composition.
    if T < 1500.0:
        n_total = p / (K_B * T)
        TINY = 1e-30 * n_total
        n = {"N2": 0.78 * n_total, "O2": 0.21 * n_total,
             "N": TINY, "O": TINY, "NO": 0.01 * n_total,
             "NO+": TINY, "e-": TINY}
        return SahaResult(n=n, T=T, p=p, K=K, converged=True,
                          method="frozen", nfev=0)

    x0 = _initial_guess_2d(T, p)

    def F(x):
        return _residuals_2d(x, T, p, K)

    x_sol, info, ier, _ = fsolve(F, x0, xtol=xtol, maxfev=maxfev,
                                  full_output=True)
    method = "fsolve"
    nfev = int(info["nfev"])
    # Verify: ier==1 from MINPACK can be a local non-zero — check residual.
    resid = np.max(np.abs(F(x_sol)))
    converged = (ier == 1) and (resid < 1e-6)

    if not converged:
        try:
            x_sol = _bisect_fallback(T, p, K, x0)
            resid = np.max(np.abs(F(x_sol)))
            converged = resid < 1e-6
            method = "fsolve+bisect"
        except Exception:
            pass

    n_N, n_O = np.exp(np.clip(x_sol, -700.0, 700.0))
    n = _close_from_NO(n_N, n_O, K)
    n = {k: float(v) for k, v in n.items()}
    return SahaResult(n=n, T=T, p=p, K=K, converged=converged,
                      method=method, nfev=nfev)


def _bisect_fallback(T: float, p: float, K: Dict[str, float],
                     x0: np.ndarray) -> np.ndarray:
    """Bisect log(n_N) and at each step inner-solve for n_O against pressure."""
    def inner_O(n_N_test: float) -> Optional[float]:
        # Solve pressure constraint for n_O given n_N
        n_target = p / (K_B * T)

        def g(log_nO):
            n_O = math.exp(log_nO)
            n = _close_from_NO(n_N_test, n_O, K)
            return sum(n.values()) - n_target

        try:
            return math.exp(brentq(g, -50.0, 70.0, xtol=1e-10))
        except Exception:
            return None

    def atom_residual(log_nN):
        n_N_t = math.exp(log_nN)
        n_O_t = inner_O(n_N_t)
        if n_O_t is None:
            return 1e30
        n = _close_from_NO(n_N_t, n_O_t, K)
        nN_atoms = n["N"] + 2*n["N2"] + n["NO"] + n["NO+"]
        nO_atoms = n["O"] + 2*n["O2"] + n["NO"] + n["NO+"]
        return nN_atoms * X_O - nO_atoms * X_N

    log_nN_sol = brentq(atom_residual, -30.0, 70.0, xtol=1e-10)
    n_N_sol = math.exp(log_nN_sol)
    n_O_sol = inner_O(n_N_sol)
    return np.log([n_N_sol, n_O_sol])
