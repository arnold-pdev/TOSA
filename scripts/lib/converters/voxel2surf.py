#!/usr/bin/env python3
"""
Backward-compatible CLI entry for staged voxel2surf.

    python scripts/lib/converters/voxel2surf.py --index 10
    python scripts/lib/converters/voxel2surf.py --index 119 --recipe v3_taubin
    python scripts/lib/converters/voxel2surf.py --index 119 --run-dir output/surfaces/runs/test
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from lib.converters.voxel2surf.cli import main

# Re-export public API for ``from lib.converters import voxel_to_surface`` when
# the package shadows this module file on import paths.
from lib.converters.voxel2surf import (  # noqa: F401
    PipelineOptions,
    SurfaceOutputFormat,
    SurfaceResult,
    save_surface,
    transfer_patches,
    voxel_to_surface,
)

if __name__ == "__main__":
    main()
