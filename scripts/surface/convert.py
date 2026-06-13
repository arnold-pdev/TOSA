#!/usr/bin/env python3
"""
Convert a cached STL to distribution .vtp (geometry + optional z scalar).

One-time or batch conversion from legacy public/stl/ downloads.

Examples:

    python scripts/surface/convert.py --index 0
    python scripts/surface/convert.py --stl-file public/stl/0.stl
    python scripts/surface/convert.py --range 0 49
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from lib.paths import ensure_scripts_on_path

ensure_scripts_on_path()

from lib.stl_common import DEFAULT_OUTPUT_DIR as DEFAULT_STL_DIR, stl_path
from lib.surface_io import DEFAULT_SURFACE_DIR, load_surface, save_vtp, vtp_path


def convert_one(stl_path_in: Path, vtp_path_out: Path, *, attach_z: bool) -> None:
    mesh = load_surface(stl_path_in)
    if attach_z:
        mesh = mesh.copy()
        mesh.point_data["z"] = np.asarray(mesh.points, dtype=float)[:, 2]
    save_vtp(mesh, vtp_path_out)
    print(f"Wrote {vtp_path_out}  ({mesh.n_points:,} points)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert STL → VTP for distribution.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--index", type=int)
    src.add_argument("--stl-file", type=Path)
    src.add_argument("--range", type=int, nargs=2, metavar=("START", "END"))
    p.add_argument("--stl-dir", type=Path, default=DEFAULT_STL_DIR)
    p.add_argument("--surface-dir", type=Path, default=DEFAULT_SURFACE_DIR)
    p.add_argument(
        "--no-z",
        action="store_true",
        help="Do not attach a z coordinate array to point_data",
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing .vtp")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    attach_z = not args.no_z

    if args.range is not None:
        start, end = args.range
        if start > end:
            raise SystemExit("--range START END requires START <= END")
        indices = range(start, end + 1)
    elif args.index is not None:
        indices = [args.index]
    else:
        stl_in = args.stl_file.expanduser().resolve()
        out = args.surface_dir / f"{stl_in.stem}.vtp"
        if out.exists() and not args.force:
            print(f"Skip (exists): {out}")
            return
        convert_one(stl_in, out, attach_z=attach_z)
        return

    for i in indices:
        stl_in = stl_path(args.stl_dir.resolve(), i)
        vtp_out = vtp_path(args.surface_dir.resolve(), i)
        if not stl_in.is_file():
            print(f"Skip (no STL): {stl_in}")
            continue
        if vtp_out.exists() and not args.force:
            print(f"Skip (exists): {vtp_out}")
            continue
        convert_one(stl_in, vtp_out, attach_z=attach_z)


if __name__ == "__main__":
    main()
