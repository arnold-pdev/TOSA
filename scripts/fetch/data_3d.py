#!/usr/bin/env python3
"""
Download the NITO 3D train bundle from Google Drive into nito/Data/3D/.

Source folder:
  https://drive.google.com/drive/folders/1uK_X3-FcCWY9LiiXkVQDI69q0t6Vosgm

Separate from 2D train/test (fetch/data_2d.py). ~106k samples, all 3D.

Usage:

    ./scripts/fetch/data_3d.sh
    ./scripts/fetch/data_3d.sh --metadata-only
    ./scripts/fetch/data_3d.sh --skip-old-bc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from lib.paths import ensure_scripts_on_path

ensure_scripts_on_path()

import gdown

from lib.paths import REPO_ROOT

DEFAULT_DATA_DIR = REPO_ROOT / "nito" / "Data" / "3D"

DATA_3D_IDS: dict[str, str] = {
    "topologies.npy": "1iAOadIAnJQCPbTpb2SdqzIULvXSP47PN",
    "boundary_conditions.npy": "1C3dHNptbhqNU9vBaqMDx8CgeXWa24rbY",
    "boundary_conditions_old.npy": "1Ps6vyEsZAGbtUq1X3Y8kygOLaDiHsjVV",
    "shapes.npy": "1Dotw3guK83l9qkcdYpnA_C9isRRRM7Zw",
    "loads.npy": "1VR5MjJVHcuQVB_swWlFzjq5z4uJZpxsT",
    "vfs.npy": "1jTwsiAy5FjOK6--ohCTmiWoP7OeXPTgT",
}


def download_file(file_id: str, output_path: Path, *, force: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        print(f"Skip (exists): {output_path}")
        return
    print(f"Downloading → {output_path}")
    gdown.download(id=file_id, output=str(output_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download NITO 3D train data into nito/Data/3D/."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Output directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Download shapes, BCs, loads, vfs only (skip topologies.npy, ~4 GB).",
    )
    parser.add_argument(
        "--skip-old-bc",
        action="store_true",
        help="Skip boundary_conditions_old.npy (superseded by boundary_conditions.npy).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files already exist",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()

    skip = set()
    if args.metadata_only:
        skip.add("topologies.npy")
    if args.skip_old_bc:
        skip.add("boundary_conditions_old.npy")

    to_fetch = {k: v for k, v in DATA_3D_IDS.items() if k not in skip}

    print(f"Fetching NITO 3D train bundle into {data_dir}")
    if skip:
        print(f"Skipping: {', '.join(sorted(skip))}")

    for name, file_id in to_fetch.items():
        download_file(file_id, data_dir / name, force=args.force)

    print("Done.")
    if args.metadata_only:
        print(
            "Metadata only. For sensitivity on GT designs, re-run without "
            "--metadata-only to fetch topologies.npy."
        )
    print(
        "Example: python scripts/voxel/inspect_dataset.py --data-dir nito/Data/3D"
    )


if __name__ == "__main__":
    main()
