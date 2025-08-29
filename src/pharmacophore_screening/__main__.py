from __future__ import annotations

import argparse
import sys

from .screener import PharmacophoreScreener


def main(argv=None):
    argv = argv or sys.argv[1:]
    parser = argparse.ArgumentParser(prog="phscreen", description="Pharmacophore screening (2D/3D)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p2d = sub.add_parser("2d", help="Run 2D similarity screening")
    p2d.add_argument("--query", required=True, help="Query SMILES")
    p2d.add_argument("--library", required=True, help="CSV with a 'smiles' column")
    p2d.add_argument("-o", "--output", required=False, help="Output CSV path")
    p2d.add_argument("--threshold", type=float, default=0.0, help="Similarity threshold")
    p2d.add_argument("--radius", type=int, default=2, help="ECFP radius")
    p2d.add_argument("--bits", type=int, default=2048, help="ECFP nBits")
    p2d.add_argument("--workers", type=int, default=1, help="Number of workers (Ray if available)")

    p3d = sub.add_parser("3d", help="Run 3D pharmacophore screening")
    p3d.add_argument("--ph4", required=True, help="Pharmacophore definition CSV")
    p3d.add_argument("--library", required=True, help="CSV with a 'smiles' column")
    p3d.add_argument("-o", "--output", required=False, help="Output CSV path")
    p3d.add_argument("--confs", type=int, default=10, help="Number of conformers")
    p3d.add_argument("--workers", type=int, default=1, help="Number of workers (Ray if available)")

    args = parser.parse_args(argv)

    if args.cmd == "2d":
        sc = PharmacophoreScreener(mode="2D")
        res = sc.screen(
            query_smiles=args.query,
            library_csv=args.library,
            similarity_threshold=args.threshold,
            radius=args.radius,
            n_bits=args.bits,
            n_workers=args.workers,
        )
    else:
        sc = PharmacophoreScreener(mode="3D")
        res = sc.screen(
            pharmacophore_csv=args.ph4,
            library_csv=args.library,
            n_conformers=args.confs,
            n_workers=args.workers,
        )
    if args.output:
        res.to_csv(args.output)
    else:
        # Print head to stdout
        print(res.df.head().to_string(index=False))


if __name__ == "__main__":
    main()

