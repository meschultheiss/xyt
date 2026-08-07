"""Generate a Declic-like synthetic landing dataset.

This script is intentionally small: it calls the package generator, writes the
landing-ready files, and prints the export manifest. It does not read any
sensitive `Data/` directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import xyt_gps as xyt


DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[2] / "Data" / "Output" / "synthetic-declic-testset"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for generated files.",
    )
    parser.add_argument(
        "--users-per-experiment",
        type=int,
        default=50,
        help="Number of generated users for each Declic experiment.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed used to make the generation reproducible.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["parquet"],
        choices=["csv", "parquet"],
        help="Formats for storyline, trips and journeys.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = xyt.generate_synthetic_declic_gps(
        users_per_experiment=args.users_per_experiment,
        random_state=args.random_state,
    )
    manifest = xyt.write_synthetic_gps_dataset(
        dataset,
        args.output,
        formats=args.formats,
    )
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
