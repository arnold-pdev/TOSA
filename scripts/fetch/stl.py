#!/usr/bin/env python3
"""
Download selected NITO-3D STL meshes from Google Drive into public/stl/.

Source folder (STLs_0-499_MarchingCubes+Smoothing):
  https://drive.google.com/drive/folders/1Jsuw7Jscc7aHD8JVHHjdgyeZhvX8sPcD

Usage:

    ./scripts/fetch/stl.sh --indices 0 3 42
    ./scripts/fetch/stl.sh --range 0 49
    ./scripts/fetch/stl.sh --from-file indices.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lib.paths import ensure_scripts_on_path

ensure_scripts_on_path()

from lib.paths import REPO_ROOT
from lib.stl_common import (
    DEFAULT_OUTPUT_DIR,
    DRIVE_FOLDER_URL,
    StlFile,
    add_index_selection_args,
    list_stl_files,
    resolve_requested_indices,
    stl_path,
)


def download_file(file_id: str, output_path: Path, *, force: bool) -> None:
    import gdown

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        print(f"Skip (exists): {output_path.name}")
        return
    print(f"Downloading → {output_path.name}")
    gdown.download(id=file_id, output=str(output_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download selected NITO-3D STL meshes into public/stl/."
    )
    add_index_selection_args(parser)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files already exist",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    requested = resolve_requested_indices(args)

    print("Listing STL files in Drive folder …")
    on_drive = {f.index: f for f in list_stl_files()}
    if not on_drive:
        raise SystemExit(f"No .stl files found in {DRIVE_FOLDER_URL}")

    missing_on_drive = [i for i in requested if i not in on_drive]
    to_fetch: list[StlFile] = [on_drive[i] for i in requested if i in on_drive]

    print(
        f"Requested {len(requested)} index(es); "
        f"{len(to_fetch)} available on Drive now "
        f"(Drive has {len(on_drive)} file(s), "
        f"indices {min(on_drive)}–{max(on_drive)})"
    )
    if missing_on_drive:
        print(
            "Not on Drive yet (skipped): "
            + ", ".join(str(i) for i in missing_on_drive)
        )

    if not to_fetch:
        raise SystemExit("Nothing to download — none of the requested indices are on Drive.")

    for stl in to_fetch:
        download_file(stl.file_id, stl_path(output_dir, stl.index), force=args.force)

    print("Done.")
    print(f"Attribution: {REPO_ROOT / 'public' / 'ATTRIBUTION.md'}")


if __name__ == "__main__":
    main()
