"""Phase 2 numerical-equivalence test.

Builds the same waverider with both packages -- ``liu2019`` and ``mfof`` with
the all-cone factory -- and asserts the two agree on every geometric metric
to within ``1e-6`` relative deviation. This is the gate condition for
Phase 2 acceptance: any drift signals that the architectural refactor has
introduced a numerical regression.

Run from the repo root:

    py -3.10 -m mfof.validate

Returns exit code 0 on success, 1 on any tolerance failure.
"""

from __future__ import annotations

import sys


def _build_liu(params, n_z=200, n_x=100):
    from liu2019.geometry import build_liu2019_waverider
    return build_liu2019_waverider(params, n_z=n_z, n_x=n_x)


def _build_mfof_all_cone(params, n_z=200, n_x=100):
    from mfof.cone_flowfield import ConeFlowfield
    from mfof.geometry import build_mfof_waverider

    beta = float(params["beta_deg"])
    gamma = float(params.get("gamma", 1.4))

    def all_cone_factory(z, Ma_z):
        return ConeFlowfield(Ma_z, beta, gamma)

    return build_mfof_waverider(params, all_cone_factory, n_z=n_z, n_x=n_x)


def run_equivalence_test(params=None, n_z: int = 200, n_x: int = 100,
                          tol: float = 1e-6, verbose: bool = True) -> bool:
    """Run the all-cone equivalence test.

    Parameters
    ----------
    params : dict, optional
        Liu-style design dict. Defaults to :data:`mfof.config.PAPER_PARAMS`.
    n_z, n_x : int
        Mesh resolution. Both packages use the same value.
    tol : float
        Relative-deviation acceptance threshold. Phase 2 spec is 1e-6.
    verbose : bool
        If True, print a per-metric table.

    Returns
    -------
    bool
        ``True`` iff every checked metric is within ``tol``.
    """
    if params is None:
        from liu2019.config import PAPER_PARAMS as params

    # ---- Warm-up the T-M solver -----------------------------------------
    # liu2019.shock.taylor_maccoll_cone_angle has a try/except that prefers
    # waverider_generator.flowfield.cone_angle but falls back to a local
    # solver if anything goes wrong. The very first call in a fresh Python
    # process returns the local-fallback value (~8.3852 deg at Ma=6, beta=13);
    # every subsequent call returns the waverider_generator value (~8.3834).
    # Whichever package builds first contaminates its delta_c grid with the
    # one-off first-call value. Warming up here ensures both packages see the
    # same post-warmup behavior.
    from liu2019.shock import taylor_maccoll_cone_angle as _tmca
    _ = _tmca(float(params["Ma_center"]), float(params["beta_deg"]),
              float(params.get("gamma", 1.4)))

    liu_wv  = _build_liu(params, n_z=n_z, n_x=n_x)
    mfof_wv = _build_mfof_all_cone(params, n_z=n_z, n_x=n_x)

    checks = [
        ("volume",       liu_wv.volume(),               mfof_wv.volume()),
        ("wetted_area",  liu_wv.wetted_area(),          mfof_wv.wetted_area()),
        ("planform",     liu_wv.planform_area(),        mfof_wv.planform_area()),
        ("base_area",    liu_wv.base_area(),            mfof_wv.base_area()),
        ("eta",          liu_wv.volumetric_efficiency(),
                         mfof_wv.volumetric_efficiency()),
    ]

    if verbose:
        print(f"{'Metric':<14} {'Liu 2019':>14} {'MFOF':>14} "
              f"{'Δ rel':>12}  status")
        print("-" * 64)

    all_pass = True
    for name, a, b in checks:
        denom = max(abs(a), 1e-30)
        delta = abs(a - b) / denom
        ok = delta < tol
        if not ok:
            all_pass = False
        if verbose:
            status = "PASS" if ok else "FAIL"
            print(f"{name:<14} {a:>14.6f} {b:>14.6f} "
                  f"{delta:>12.2e}  {status}")

    if verbose:
        print()
        print(f"Phase 2 equivalence: "
              f"{'PASS' if all_pass else 'FAIL'} "
              f"(tol = {tol:.0e})")
    return all_pass


def main():
    ok = run_equivalence_test()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
