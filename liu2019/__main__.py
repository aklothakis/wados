"""CLI entry point for the liu2019 package.

Usage:
    python -m liu2019 --validate
    python -m liu2019 --generate --stl out.stl
"""

import argparse

from .config import PAPER_PARAMS
from .geometry import build_liu2019_waverider
from .validate import run_paper_validation


def main():
    parser = argparse.ArgumentParser(
        prog="liu2019",
        description="Liu et al. 2019 variable-Mach osculating flowfield "
                    "waverider reproduction.",
    )
    parser.add_argument("--validate", action="store_true",
                        help="Run paper validation (Tables 4 and Fig. 12).")
    parser.add_argument("--generate", action="store_true",
                        help="Generate waverider geometry only (no aero).")
    parser.add_argument("--stl", metavar="PATH", default=None,
                        help="Export STL to this path.")
    parser.add_argument("--n-z", type=int, default=200,
                        help="Number of spanwise osculating planes.")
    parser.add_argument("--n-x", type=int, default=100,
                        help="Number of chordwise streamline samples.")
    parser.add_argument("--no-aero", action="store_true",
                        help="Skip the aerodynamic sweep.")
    args = parser.parse_args()

    if args.validate:
        run_paper_validation(n_z=args.n_z, n_x=args.n_x,
                             run_aero=not args.no_aero)
    elif args.generate:
        wr = build_liu2019_waverider(PAPER_PARAMS,
                                     n_z=args.n_z, n_x=args.n_x)
        s = wr.summary()
        print(f"Vol   = {s['Vol_m3']:.4f} m3 "
              f"(paper {s['reference']['Vol_m3']:.3f})")
        print(f"S_wet = {s['S_wet_m2']:.4f} m2 "
              f"(paper {s['reference']['S_wet_m2']:.3f})")
        print(f"S_p   = {s['S_p_m2']:.4f} m2 "
              f"(paper {s['reference']['S_p_m2']:.3f})")
        print(f"S_b   = {s['S_b_m2']:.4f} m2 "
              f"(paper {s['reference']['S_b_m2']:.3f})")
        print(f"eta   = {s['eta']:.4f} "
              f"(paper {s['reference']['eta']:.4f})")
        if args.stl:
            wr.export_stl(args.stl)
            print(f"STL written to {args.stl}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
