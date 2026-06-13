#!/usr/bin/env python3
"""
Download NITO pre-trained checkpoints into nito/Checkpoints/ (TOSA wrapper).

Usage:

    ./scripts/fetch/checkpoints.sh
    ./scripts/fetch/checkpoints.sh --256x256
    ./scripts/fetch/checkpoints.sh --all
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

DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "nito" / "Checkpoints"
CHECKPOINT_FILENAME = "checkpoint_epoch_50.pth"

CHECKPOINT_IDS: dict[str, str] = {
    "64x64": "13y4UdoxBMwZnO-3Oz3dWCVlgfkwqwLd1",
    "256x256": "1VAnEhX1GTLYQXOTq-f1kZperQJTTbYyC",
    "64x64_256x256": "1JX1M9EOrpWfwEUUwFXrNGeEyObFk9gYP",
    "All": "18ocK4a9zV2v5Zv986z_VdYC0AK-QzZtn",
}

SPRINT_PRESETS = ("64x64",)


def download_checkpoint(
    preset: str,
    checkpoint_dir: Path,
    *,
    force: bool,
) -> None:
    if preset not in CHECKPOINT_IDS:
        raise ValueError(
            f"Unknown preset {preset!r}. Choose from: {', '.join(CHECKPOINT_IDS)}"
        )
    out_dir = checkpoint_dir / preset
    out_path = out_dir / CHECKPOINT_FILENAME
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        print(f"Skip (exists): {out_path}")
        return
    print(f"Downloading {preset} → {out_path}")
    gdown.download(id=CHECKPOINT_IDS[preset], output=str(out_path), quiet=False)


def parse_args() -> argparse.Namespace:
    names = tuple(CHECKPOINT_IDS)
    p = argparse.ArgumentParser(
        description="Download NITO checkpoints into nito/Checkpoints/."
    )
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help=f"Output root (default: {DEFAULT_CHECKPOINT_DIR})",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if checkpoint_epoch_50.pth exists",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help=f"Download every preset: {', '.join(names)}",
    )
    for name in names:
        flag = "--" + name.replace("_", "-")
        p.add_argument(
            flag,
            dest="presets",
            action="append_const",
            const=name,
            help=f"Download nito/Checkpoints/{name}/{CHECKPOINT_FILENAME}",
        )
    p.set_defaults(presets=None)
    return p.parse_args()


def resolve_presets(args: argparse.Namespace) -> tuple[str, ...]:
    if args.all:
        return tuple(CHECKPOINT_IDS)
    if args.presets:
        return tuple(dict.fromkeys(args.presets))
    return SPRINT_PRESETS


def main() -> None:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir.resolve()
    presets = resolve_presets(args)
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Presets: {', '.join(presets)}")
    for preset in presets:
        download_checkpoint(preset, checkpoint_dir, force=args.force)
    print("Done.")


if __name__ == "__main__":
    main()
