#!/usr/bin/env python3
"""Print validation dataset metrics and launch visualize.py for one index."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from lib.paths import REPO_ROOT


def main() -> None:
    p = argparse.ArgumentParser(description="View validation dataset metrics / meshes.")
    p.add_argument("--dataset", type=str, default="compatible-pipelines-v1")
    p.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT / "output" / "surfaces" / "validation",
    )
    p.add_argument("--index", type=int, default=119, help="Canonical test index (default 119)")
    p.add_argument(
        "--open",
        type=str,
        default=None,
        help="Recipe CLI name to open in PyVista (e.g. v3_taubin)",
    )
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--overlay-voxels", action="store_true", default=True)
    args = p.parse_args()

    root = args.root.expanduser().resolve() / args.dataset
    summary = root / "metrics" / "summary.csv"
    if not summary.is_file():
        raise SystemExit(f"Missing {summary}. Run build_validation_dataset.py first.")

    rows = list(csv.DictReader(summary.open(encoding="utf-8")))
    rows = [r for r in rows if int(r["index"]) == args.index]

    if not rows:
        raise SystemExit(f"No rows for index {args.index}.")

    cols = [
        "recipe_cli",
        "tag",
        "status",
        "vf_delta",
        "bc_plane_max_residual",
        "bc_footprint_coverage",
        "bc_labeled_triangles",
        "bc_min_patch_tris",
        "free_dihedral_p95_deg",
        "axis_aligned_edge_frac",
        "vertices",
        "faces",
        "construction_sec",
        "bc_planar_cap_method",
        "bc_assembly",
        "planar_cap_watertight",
        "planar_cap_bodies",
        "load_check_failed",
    ]
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    header_cols = ["index", *cols]
    widths["index"] = max(len("index"), *(len(str(r.get("index", ""))) for r in rows))
    fmt = "  ".join(f"{{:{widths[c]}}}" for c in header_cols)
    print(fmt.format(*header_cols))
    print("  ".join("-" * widths[c] for c in header_cols))
    for r in rows:
        print(fmt.format(r.get("index", ""), *(r.get(c, "") for c in cols)))

    if args.open is not None:
        vtp = root / "vtp" / str(args.index) / f"{args.open}.vtp"
        if not vtp.is_file():
            raise SystemExit(f"Missing {vtp}")
        vis = REPO_ROOT / "scripts" / "surface" / "visualize.py"
        cmd = [
            sys.executable,
            str(vis),
            "--index",
            str(args.index),
            "--surface-file",
            str(vtp),
            "--scalar",
            "boundary",
        ]
        if args.overlay_voxels:
            cmd.append("--overlay-voxels")
        if args.data_dir is not None:
            cmd.extend(["--data-dir", str(args.data_dir)])
        else:
            manifest = root / "manifest.json"
            if manifest.is_file():
                import json

                data_dir = json.loads(manifest.read_text(encoding="utf-8")).get("data_dir")
                if data_dir:
                    cmd.extend(["--data-dir", data_dir])
        subprocess.call(cmd)


if __name__ == "__main__":
    main()
