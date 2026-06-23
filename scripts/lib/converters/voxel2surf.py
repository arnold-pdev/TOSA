#!/usr/bin/env python3
"""CLI entry for voxel2surf surface meshing (cuberille + variational fairing).

    python scripts/lib/converters/voxel2surf.py --index 119 -o out/119.vtp

Thin launcher around :mod:`lib.converters.voxel2surf.cli`; see the package
``README.md`` for the pipeline and options.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from lib.converters.voxel2surf.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
