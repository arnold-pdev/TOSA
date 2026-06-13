#!/usr/bin/env python3
"""
Remove locally cached NITO-3D STL meshes from public/stl/.

Usage:

    ./scripts/fetch/clean_stl.sh --indices 0 3 42
    ./scripts/fetch/clean_stl.sh --range 0 49
    ./scripts/fetch/clean_stl.sh --all
    ./scripts/fetch/clean_stl.sh --all --dry-run
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

from lib.stl_common import DEFAULT_OUTPUT_DIR, resolve_requested_indices, stl_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove cached NITO-3D STL meshes from public/stl/."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--indices",
        type=int,
        nargs="+",
        metavar="N",
        help="Dataset indices to remove (e.g. --indices 0 3 42).",
    )
    group.add_argument(
        "--range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="Inclusive index range (e.g. --range 0 49).",
    )
    group.add_argument(
        "--from-file",
        type=Path,
        metavar="PATH",
        help="Text file of indices (whitespace- or comma-separated).",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Remove every *.stl file in the output directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to clean (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be removed without deleting files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()

    if args.all:
        targets = sorted(output_dir.glob("*.stl")) if output_dir.is_dir() else []
    else:
        indices = resolve_requested_indices(args)
        targets = [stl_path(output_dir, i) for i in indices]

    existing = [p for p in targets if p.exists()]
    missing = [p for p in targets if not p.exists()]

    if missing and not args.all:
        print(
            "Not present (skipped): "
            + ", ".join(p.name for p in missing)
        )

    if not existing:
        print("Nothing to remove.")
        return

    verb = "Would remove" if args.dry_run else "Removing"
    for path in existing:
        print(f"{verb}: {path}")
        if not args.dry_run:
            path.unlink()

    print(f"Done. {len(existing)} file(s) {'would be ' if args.dry_run else ''}removed.")


if __name__ == "__main__":
    main()
