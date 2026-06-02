#!/usr/bin/env python3
"""
Download NITO-3D .npy bundles into nito/Data/ (TOSA wrapper; submodule stays vanilla).

Drive file IDs match nito/download.py in NITO_Public — update both if upstream changes.

Usage (from repo root, tosa env with gdown installed):

    python scripts/fetch_nito_data.py --test-only      # sprint default (~85 MB)
    python scripts/fetch_nito_data.py --full           # train + test (~4.7 GB)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gdown

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "nito" / "Data"

# Keep in sync with nito/download.py (NITO_Public submodule).
TRAIN_IDS = {
    "topologies.npy": "1VitnaTfJtkEqY5jFIdfyB132-8Gu7s7u",
    "boundary_conditions.npy": "1CYqL9BMR6PiM9PfE81VZIWyPIK1hDzAR",
    "shapes.npy": "1g6A152bwJEQwh0Bvr9xQUHeBboT4gRPT",
    "loads.npy": "1ZMvQk7J_kKaAaTpr2BYkm8B4E-6E3hRB",
    "vfs.npy": "1WFeDNY_qwWeVSCVoSqIh_SrPXmf8Xsym",
}

TEST_IDS = {
    "topologies.npy": "1tFa2twksRnhc67XR47yQ-aeWnwZFfVe7",
    "boundary_conditions.npy": "1tXzECt2Gb2__lXVG799jc1hvWMTqWu2T",
    "shapes.npy": "1bY4SuHoWBJZ2iJRgvRSUdbJL20OTQ_RG",
    "loads.npy": "126ysGYKE9RynNM814QNzQK94knhEtm03",
    "vfs.npy": "1HlKcqpZ78gjVolEFUrwQVJW1vQRdOS88",
}


def download_file(file_id: str, output_path: Path, *, force: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        print(f"Skip (exists): {output_path}")
        return
    print(f"Downloading → {output_path}")
    gdown.download(id=file_id, output=str(output_path))


def download_bundle(
    ids: dict[str, str],
    data_dir: Path,
    *,
    subdir: str | None = None,
    force: bool,
) -> None:
    root = data_dir / subdir if subdir else data_dir
    for name, file_id in ids.items():
        download_file(file_id, root / name, force=force)


def parse_args() -> argparse.Namespace:
    group = argparse.ArgumentParser(
        description="Download NITO data into nito/Data/ without modifying the submodule."
    )
    scope = group.add_mutually_exclusive_group()
    scope.add_argument(
        "--test-only",
        action="store_true",
        help="Download only nito/Data/Test/ (~85 MB, 5k samples). Default if neither flag is set.",
    )
    scope.add_argument(
        "--full",
        action="store_true",
        help="Download train (nito/Data/) and test (nito/Data/Test/). ~4.7 GB.",
    )
    group.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Output directory (default: {DEFAULT_DATA_DIR})",
    )
    group.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files already exist",
    )
    return group.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    test_only = args.test_only or not args.full

    if test_only:
        print(f"Fetching test split into {data_dir / 'Test'}")
        download_bundle(TEST_IDS, data_dir, subdir="Test", force=args.force)
    else:
        print(f"Fetching train + test into {data_dir}")
        download_bundle(TRAIN_IDS, data_dir, force=args.force)
        download_bundle(TEST_IDS, data_dir, subdir="Test", force=args.force)

    print("Done.")


if __name__ == "__main__":
    main()
