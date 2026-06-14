"""CLI entry: python scripts/lib/converters/voxel2surf.py"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[3]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from lib.converters.voxel2surf.cli import main

if __name__ == "__main__":
    main()
