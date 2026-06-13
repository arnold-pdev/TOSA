"""Repository paths shared by TOSA scripts."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"


def ensure_scripts_on_path() -> None:
    root = str(SCRIPTS_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def ensure_nito_on_path() -> None:
    """Add nito/ submodule root for ATOMS / NITO imports."""
    ensure_scripts_on_path()
    nito_root = str(REPO_ROOT / "nito")
    if nito_root not in sys.path:
        sys.path.insert(0, nito_root)
