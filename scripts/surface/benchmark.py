#!/usr/bin/env python3
"""Run voxel2surf on probe indices and write metrics for recipe comparison."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from lib.paths import REPO_ROOT


def _load_indices(path: Path) -> list[int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[int] = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(int(line))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark voxel2surf on probe indices.")
    p.add_argument("--recipe", type=str, default="pyvista_laplacian")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument(
        "--indices-file",
        type=Path,
        default=REPO_ROOT / "batch" / "probe_indices.txt",
    )
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--test", action="store_true")
    p.add_argument("--on-load-fail", choices=("raise", "warn"), default="warn")
    p.add_argument("-q", action="store_true", help="Quiet voxel2surf output")
    args = p.parse_args()

    indices = _load_indices(args.indices_file.expanduser())
    if not indices:
        raise SystemExit(f"No indices in {args.indices_file}")

    run_dir = args.run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    cli = REPO_ROOT / "scripts" / "lib" / "converters" / "voxel2surf.py"
    failures: list[int] = []

    for index in indices:
        cmd = [
            sys.executable,
            str(cli),
            "--index",
            str(index),
            "--recipe",
            args.recipe,
            "--run-dir",
            str(run_dir),
            "--on-load-fail",
            args.on_load_fail,
        ]
        if args.data_dir is not None:
            cmd.extend(["--data-dir", str(args.data_dir)])
        if args.test:
            cmd.append("--test")
        if args.q:
            cmd.append("-q")
        print(f"=== index {index} ===")
        rc = subprocess.call(cmd)
        if rc != 0:
            failures.append(index)

    summary = run_dir / "metrics" / "summary.csv"
    print(f"\nMetrics: {summary}")
    if failures:
        raise SystemExit(f"Failed indices: {failures}")
    print("All probe indices completed.")


if __name__ == "__main__":
    main()
