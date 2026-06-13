"""Shared helpers for NITO-3D STL fetch/clean scripts."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from lib.paths import REPO_ROOT

DEFAULT_OUTPUT_DIR = REPO_ROOT / "public" / "stl"

DRIVE_FOLDER_URL = (
    "https://drive.google.com/drive/folders/1Jsuw7Jscc7aHD8JVHHjdgyeZhvX8sPcD"
)

_STL_RE = re.compile(r"^(\d+)\.stl$")


@dataclass(frozen=True)
class StlFile:
    index: int
    name: str
    file_id: str


def list_stl_files() -> list[StlFile]:
    """Return STL entries from the Drive folder, sorted by index."""
    import gdown

    entries = gdown.download_folder(
        DRIVE_FOLDER_URL,
        quiet=True,
        skip_download=True,
    )
    out: list[StlFile] = []
    for entry in entries:
        match = _STL_RE.match(entry.path)
        if not match:
            continue
        out.append(
            StlFile(
                index=int(match.group(1)),
                name=entry.path,
                file_id=entry.id,
            )
        )
    out.sort(key=lambda f: f.index)
    return out


def read_indices_file(path: Path) -> list[int]:
    """Parse indices from a text file (whitespace- or comma-separated)."""
    text = path.read_text()
    tokens = re.split(r"[\s,]+", text.strip())
    if not tokens or tokens == [""]:
        raise ValueError(f"No indices found in {path}")
    return sorted({int(token) for token in tokens})


def add_index_selection_args(parser: argparse.ArgumentParser) -> None:
    """Add mutually exclusive ways to choose STL indices."""
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--indices",
        type=int,
        nargs="+",
        metavar="N",
        help="Dataset indices to select (e.g. --indices 0 3 42).",
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


def resolve_requested_indices(args: argparse.Namespace) -> list[int]:
    if args.indices is not None:
        return sorted(set(args.indices))
    if args.range is not None:
        start, end = args.range
        if start > end:
            raise SystemExit("--range START END requires START <= END.")
        return list(range(start, end + 1))
    return read_indices_file(args.from_file.resolve())


def stl_path(output_dir: Path, index: int) -> Path:
    return output_dir / f"{index}.stl"
