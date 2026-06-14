"""Run directory layout, logging, manifest, and metrics CSV."""

from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from lib.converters.voxel2surf.types import PipelineOptions, RunMetrics


class PipelineLogger:
    """Capture pipeline messages; optionally echo to stdout and write to a file."""

    def __init__(self, *, echo_stdout: bool, log_path: Path | None) -> None:
        self._echo = echo_stdout
        self._path = log_path.expanduser().resolve() if log_path is not None else None
        self._lines: list[str] = []

    def __call__(self, msg: str) -> None:
        self._lines.append(msg)
        if self._echo:
            print(msg, flush=True)

    def save(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text("\n".join(self._lines) + "\n", encoding="utf-8")


def _git_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _options_dict(opts: PipelineOptions) -> dict:
    data = asdict(opts)
    data["on_load_fail"] = str(opts.on_load_fail)
    return data


def update_manifest(
    run_dir: Path,
    *,
    recipe_id: str,
    options: PipelineOptions,
    index: int,
    metrics: RunMetrics,
) -> None:
    """Append run metadata to ``<run-dir>/manifest.json``."""
    run_dir = run_dir.expanduser().resolve()
    path = run_dir / "manifest.json"
    if path.is_file():
        manifest = json.loads(path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "recipe_id": recipe_id,
            "git_sha": _git_sha(),
            "created": datetime.now(timezone.utc).isoformat(),
            "options": _options_dict(options),
            "indices": [],
            "runs": [],
        }
    if index not in manifest["indices"]:
        manifest["indices"].append(index)
        manifest["indices"].sort()
    manifest["runs"].append(
        {
            "index": index,
            "metrics": metrics.as_csv_row(),
            "updated": datetime.now(timezone.utc).isoformat(),
        }
    )
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


_METRICS_FIELDS = list(RunMetrics(index=0, recipe_id="", vf_voxel=0.0, vf_mesh=None,
    vf_delta=None, vf_nito=None, bc_plane_max_residual=0.0, bc_labeled_triangles=0,
    load_check_failed=False, vertices=0, faces=0, patches=0).as_csv_row().keys())


def append_metrics_csv(run_dir: Path, metrics: RunMetrics) -> None:
    """Append one row to ``<run-dir>/metrics/summary.csv``."""
    run_dir = run_dir.expanduser().resolve()
    path = run_dir / "metrics" / "summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_METRICS_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(metrics.as_csv_row())
