#!/usr/bin/env python3
"""
Build a multi-recipe validation dataset for offline viewing and comparison.

Writes::

    <output-dir>/<dataset-id>/
      manifest.json
      VIEW.md              # visualize.py commands + metric guide
      metrics/summary.csv  # one row per (index, recipe)
      vtp/<index>/<recipe>.vtp
      logs/<index>_<recipe>.log

Smoothness (stairstepping) proxies are recorded per run:
``axis_aligned_edge_frac``, ``free_dihedral_mean_deg``, ``free_dihedral_p95_deg``.

Examples::

    # Plan only
    python scripts/surface/build_validation_dataset.py --dry-run

    # Build full matrix (requires nito/Data)
    python scripts/surface/build_validation_dataset.py \\
      --dataset-id compatible-pipelines-v1 \\
      --on-load-fail warn --on-bc-fail warn

    # View one index side-by-side (after build)
    python scripts/surface/view_validation.py --dataset compatible-pipelines-v1 --index 119
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from lib.paths import REPO_ROOT

# Core QA metrics (compatible-pipelines-v1) plus extended timing / planar-cap columns.
SUMMARY_FIELDNAMES: list[str] = [
    # identity
    "index",
    "recipe_cli",
    "base_recipe",
    "recipe_id",
    "tag",
    "status",
    # core surface / BC / smoothness (first validation matrix)
    "vf_delta",
    "bc_plane_max_residual",
    "bc_footprint_coverage",
    "bc_labeled_triangles",
    "bc_min_patch_tris",
    "free_dihedral_mean_deg",
    "free_dihedral_p95_deg",
    "axis_aligned_edge_frac",
    "load_check_failed",
    "vertices",
    "faces",
    # timing / planar cap (extended matrices)
    "construction_sec",
    "timing_extract_sec",
    "timing_bc_planar_cap_sec",
    "timing_bc_assembly_sec",
    "timing_smooth_sec",
    "timing_validate_sec",
    "bc_planar_cap_method",
    "bc_assembly",
    "planar_cap_containment_ok",
    "planar_cap_watertight",
    "planar_cap_bodies",
    "shared_seam_n_caps",
    "shared_seam_watertight",
    "shared_seam_bodies",
    "vtp_path",
]


def _recipe_id(entry: dict) -> str:
    from lib.converters.voxel2surf.recipes import get_recipe

    base = entry.get("base_recipe") or entry["recipe_cli"]
    return get_recipe(base).id


def _base_recipe(entry: dict) -> str:
    return entry.get("base_recipe") or entry["recipe_cli"]


def _load_indices(path: Path) -> list[int]:
    out: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(int(line))
    return out


def _load_matrix(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _dataset_root(output_dir: Path, dataset_id: str) -> Path:
    return output_dir.expanduser().resolve() / dataset_id


def _write_view_md(
    root: Path,
    recipes: list[dict],
    indices: list[int],
    *,
    data_dir: str,
    dataset_id: str = "",
) -> None:
    lines = [
        "# Validation dataset viewer",
        "",
        f"Dataset: `{dataset_id or root.name}`",
        "",
        "## Stairstepping / smoothness",
        "",
        "Compare **free boundary** (not BC patches) using metrics in `metrics/summary.csv`:",
        "",
        "| Metric | Interpretation |",
        "|--------|----------------|",
        "| `axis_aligned_edge_frac` | Higher → more axis-aligned edges on free skin (stairstep proxy) |",
        "| `free_dihedral_p95_deg` | Higher → sharper creases between free triangles |",
        "| `free_dihedral_mean_deg` | Mean crease angle on free-free edges |",
        "",
        "**Visual check** (recommended): overlay voxels, rotate free boundary, compare to `extract_only`.",
        "",
        "## PyVista commands",
        "",
        "From repo root:",
        "",
    ]
    for index in indices:
        lines.append(f"### Index {index}")
        lines.append("")
        for entry in recipes:
            recipe_cli = entry["recipe_cli"]
            vtp = root / "vtp" / str(index) / f"{recipe_cli}.vtp"
            lines.append(f"**{recipe_cli}** ({entry.get('tag', '')}) — {entry.get('role', '')}")
            lines.append("```bash")
            lines.append(
                f"python scripts/surface/visualize.py "
                f"--index {index} "
                f"--surface-file {vtp} --scalar boundary "
                f"--overlay-voxels --data-dir {data_dir}"
            )
            lines.append("```")
            lines.append("")
    lines.append("## Quick metric pivot")
    lines.append("")
    lines.append("```bash")
    lines.append(
        f"python scripts/surface/view_validation.py "
        f"--dataset {dataset_id or root.name} --index <index>"
    )
    lines.append("```")
    (root / "VIEW.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _merge_summary_row(summary_path: Path, row: dict, fieldnames: list[str]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[tuple[str, str], dict] = {}
    if summary_path.is_file() and summary_path.stat().st_size > 0:
        with summary_path.open(newline="", encoding="utf-8") as fh:
            for old in csv.DictReader(fh):
                existing[(old["index"], old["recipe_cli"])] = old
    key = (str(row["index"]), row["recipe_cli"])
    existing[key] = row
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for (_, _), r in sorted(existing.items(), key=lambda kv: (int(kv[0][0]), kv[0][1])):
            writer.writerow(r)


def main() -> None:
    p = argparse.ArgumentParser(description="Build multi-recipe validation dataset.")
    p.add_argument(
        "--matrix",
        type=Path,
        default=REPO_ROOT / "batch" / "validation_matrix.json",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "output" / "surfaces" / "validation",
    )
    p.add_argument("--dataset-id", type=str, default=None, help="Defaults to matrix id")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--test", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Print jobs only")
    p.add_argument("--on-load-fail", choices=("raise", "warn"), default="warn")
    p.add_argument("--on-bc-fail", choices=("raise", "warn"), default="warn")
    p.add_argument("-q", action="store_true")
    p.add_argument(
        "--refresh-metrics",
        action="store_true",
        help="Re-parse logs into summary.csv without re-running voxel2surf",
    )
    args = p.parse_args()

    matrix = _load_matrix(args.matrix.expanduser())
    dataset_id = args.dataset_id or matrix["id"]
    root = _dataset_root(args.output_dir, dataset_id)
    indices_path = REPO_ROOT / matrix.get("indices_file", "batch/validation_indices.txt")
    indices = _load_indices(indices_path)
    if not indices:
        raise SystemExit(f"No indices in {indices_path}")

    recipes = [
        {**entry, "recipe_cli": entry.get("recipe_cli", entry.get("cli", ""))}
        for entry in matrix["recipes"]
    ]
    cli = REPO_ROOT / "scripts" / "lib" / "converters" / "voxel2surf.py"
    data_dir = args.data_dir or (REPO_ROOT / "nito" / "Data" / ("Test" if args.test else "3D"))

    jobs: list[tuple[int, dict]] = [(i, r) for i in indices for r in recipes]
    print(f"Dataset {dataset_id}: {len(jobs)} runs ({len(indices)} indices × {len(recipes)} recipes)")
    print(f"Output: {root}")

    if args.dry_run:
        for index, entry in jobs:
            args_s = " ".join(entry.get("cli_args", []))
            print(f"  index={index} recipe={entry['recipe_cli']} base={_base_recipe(entry)} {args_s}")
        return

    root.mkdir(parents=True, exist_ok=True)
    (root / "vtp").mkdir(exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    (root / "metrics").mkdir(exist_ok=True)

    summary_path = root / "metrics" / "summary.csv"
    fieldnames = SUMMARY_FIELDNAMES
    failures: list[str] = []

    if args.refresh_metrics:
        _refresh_summary_from_logs(root, recipes, indices, summary_path, fieldnames)
        _write_view_md(root, recipes, indices, data_dir=str(data_dir), dataset_id=dataset_id)
        print(f"Refreshed: {summary_path}")
        return

    for index, entry in jobs:
        recipe_cli = entry["recipe_cli"]
        base_recipe = _base_recipe(entry)
        recipe_id = _recipe_id(entry)
        out_vtp = root / "vtp" / str(index) / f"{recipe_cli}.vtp"
        out_vtp.parent.mkdir(parents=True, exist_ok=True)
        log_path = root / "logs" / f"{index}_{recipe_cli}.log"

        cmd = [
            sys.executable,
            str(cli),
            "--index",
            str(index),
            "--recipe",
            base_recipe,
            "--output",
            str(out_vtp),
            "--log-file",
            str(log_path),
            "--on-load-fail",
            args.on_load_fail,
            "--on-bc-fail",
            args.on_bc_fail,
        ]
        cmd.extend(entry.get("cli_args", []))
        if args.data_dir is not None:
            cmd.extend(["--data-dir", str(args.data_dir)])
        elif args.test:
            cmd.append("--test")
        if args.q:
            cmd.append("-q")

        print(f"=== index {index} recipe {recipe_cli} (base={base_recipe}) ===")
        t0 = time.perf_counter()
        rc = subprocess.call(cmd)
        wall_sec = time.perf_counter() - t0
        status = "ok" if rc == 0 else "failed"
        if rc != 0:
            failures.append(f"{index}:{recipe_cli}")

        row = {
            "index": index,
            "recipe_cli": recipe_cli,
            "base_recipe": base_recipe,
            "recipe_id": recipe_id,
            "tag": entry.get("tag", ""),
            "status": status,
            "vtp_path": str(out_vtp),
        }
        if rc == 0 and out_vtp.is_file():
            run_metrics = _read_log_metrics(log_path)
            row.update(run_metrics)
            row["construction_sec"] = run_metrics.get(
                "construction_sec",
                round(wall_sec, 3),
            )
        else:
            row["construction_sec"] = round(wall_sec, 3)
        _merge_summary_row(summary_path, row, fieldnames)

    manifest = {
        "id": dataset_id,
        "created": datetime.now(timezone.utc).isoformat(),
        "matrix": matrix,
        "indices": indices,
        "data_dir": str(data_dir),
        "n_runs": len(jobs),
        "n_failed": len(failures),
        "failures": failures,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_view_md(root, recipes, indices, data_dir=str(data_dir), dataset_id=dataset_id)

    print(f"\nSummary: {summary_path}")
    print(f"Viewer guide: {root / 'VIEW.md'}")
    if failures:
        raise SystemExit(f"Failed runs: {failures}")
    print("Validation dataset complete.")


def _refresh_summary_from_logs(
    root: Path,
    recipes: list[dict],
    indices: list[int],
    summary_path: Path,
    fieldnames: list[str],
) -> None:
    """Re-parse every ``logs/<index>_<recipe_cli>.log`` under the dataset root."""
    recipe_by_cli = {entry["recipe_cli"]: entry for entry in recipes}
    logs_dir = root / "logs"
    if not logs_dir.is_dir():
        raise SystemExit(f"Missing logs dir: {logs_dir}")

    # Start fresh so column order matches SUMMARY_FIELDNAMES.
    existing: dict[tuple[str, str], dict] = {}
    for log_path in sorted(logs_dir.glob("*.log")):
        stem = log_path.stem
        if "_" not in stem:
            continue
        index_str, recipe_cli = stem.split("_", 1)
        try:
            index = int(index_str)
        except ValueError:
            continue
        if indices and index not in indices:
            continue

        entry = recipe_by_cli.get(recipe_cli, {})
        out_vtp = root / "vtp" / str(index) / f"{recipe_cli}.vtp"
        row = {
            "index": index,
            "recipe_cli": recipe_cli,
            "base_recipe": _base_recipe(entry) if entry else recipe_cli,
            "recipe_id": _recipe_id(entry) if entry else "",
            "tag": entry.get("tag", ""),
            "status": "ok" if out_vtp.is_file() else "unknown",
            "vtp_path": str(out_vtp),
        }
        row.update(_read_log_metrics(log_path))
        if not row.get("recipe_id") and entry:
            row["recipe_id"] = _recipe_id(entry)
        existing[(str(index), recipe_cli)] = row

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for (_, _), r in sorted(existing.items(), key=lambda kv: (int(kv[0][0]), kv[0][1])):
            writer.writerow(r)


def _read_log_metrics(log_path: Path) -> dict[str, str | float | int]:
    """Parse key metrics from a voxel2surf log file."""
    import re

    text = log_path.read_text(encoding="utf-8") if log_path.is_file() else ""
    out: dict[str, str | float | int] = {}

    recipe_id = re.search(r"voxel2surf\s+recipe=([\w-]+)", text)
    if recipe_id:
        out["recipe_id"] = recipe_id.group(1)

    cli_recipe = re.search(r"index=\d+\s+recipe=(\S+)", text)
    if cli_recipe:
        out["base_recipe"] = cli_recipe.group(1)

    done = re.search(
        r"\(([\d,]+) vertices,.*?([\d,]+) labeled triangles",
        text,
    )
    if done:
        out["vertices"] = int(done.group(1).replace(",", ""))
        out["bc_labeled_triangles"] = int(done.group(2).replace(",", ""))

    vf = re.search(
        r"volume fraction \[[^\]]*\]: mesh=[\d.]+ voxel=[\d.]+ delta=([+-]?[\d.]+)",
        text,
    )
    if vf:
        out["vf_delta"] = float(vf.group(1))
    else:
        vf2 = re.search(
            r"volume_fraction\s+after \(surface\)=([\d.]+)\s+delta=([+-]?[\d.]+)",
            text,
        )
        if vf2:
            out["vf_delta"] = float(vf2.group(2))

    bc = re.search(
        r"vertex plane residual=([\d.eE+-]+), footprint coverage=([\d.]+)%",
        text,
    )
    if bc:
        out["bc_plane_max_residual"] = float(bc.group(1))
        out["bc_footprint_coverage"] = float(bc.group(2)) / 100.0

    min_patch = re.search(r"bc_min_patch_tris=(\d+)", text)
    if min_patch:
        out["bc_min_patch_tris"] = int(min_patch.group(1))

    smooth = re.search(
        r"dihedral_mean=([\d.]+)° p95=([\d.]+)° axis_aligned_edges=([\d.]+)%",
        text,
    )
    if smooth:
        out["free_dihedral_mean_deg"] = float(smooth.group(1))
        out["free_dihedral_p95_deg"] = float(smooth.group(2))
        out["axis_aligned_edge_frac"] = float(smooth.group(3)) / 100.0

    final = re.search(r"\[final\].*?faces=([\d,]+)", text)
    if final:
        out["faces"] = int(final.group(1).replace(",", ""))

    out["load_check_failed"] = 1 if "load_check=WARN" in text else 0

    wall = re.search(r"construction_sec=([\d.]+)", text)
    if wall:
        out["construction_sec"] = float(wall.group(1))

    for stage in ("extract", "bc_planar_cap", "bc_assembly", "smooth", "validate"):
        m = re.search(rf"\[{stage}\] elapsed=([\d.]+)s", text)
        if m:
            out[f"timing_{stage}_sec"] = float(m.group(1))

    cap_method = re.search(r"bc_planar_cap \[([^\]]+)\]", text)
    if cap_method:
        out["bc_planar_cap_method"] = cap_method.group(1).split()[0]
    elif re.search(r"bc_planar_cap=False", text) or "bc_planar_cap=0" in text:
        out["bc_planar_cap_method"] = "disabled"

    cap_stats = re.search(
        r"method=\S+ caps=\d+ cap_tris=[\d,]+ containment_ok=(\w+)",
        text,
    )
    if cap_stats:
        out["planar_cap_containment_ok"] = int(cap_stats.group(1) == "True")

    cap_topo = re.search(
        r"cap mesh: watertight=(\w+) bodies=(-?\d+)",
        text,
    )
    if cap_topo:
        out["planar_cap_watertight"] = int(cap_topo.group(1) == "True")
        out["planar_cap_bodies"] = int(cap_topo.group(2))

    asm = re.search(r"bc_assembly \[(\w+)\]", text)
    if asm:
        out["bc_assembly"] = asm.group(1)

    seam = re.search(
        r"caps=(\d+) seam_segments=\d+ loops=\d+ fallback_caps=\d+ "
        r"merged_verts=\d+ watertight=(\w+) bodies=(-?\d+)",
        text,
    )
    if seam:
        out["shared_seam_n_caps"] = int(seam.group(1))
        out["shared_seam_watertight"] = int(seam.group(2) == "True")
        out["shared_seam_bodies"] = int(seam.group(3))

    return out


if __name__ == "__main__":
    main()
