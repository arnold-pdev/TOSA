"""Stage dispatch for the voxel2surf pipeline."""

from __future__ import annotations

from lib.converters.voxel2surf.stages.bc_build import run_bc_build
from lib.converters.voxel2surf.stages.bc_enforce import run_bc_enforce, run_bc_transfer
from lib.converters.voxel2surf.stages.extract import run_extract
from lib.converters.voxel2surf.stages.smooth import run_smooth
from lib.converters.voxel2surf.stages.subdivide import run_subdivide
from lib.converters.voxel2surf.stages.validate import run_validate

STAGE_RUNNERS = {
    "extract": run_extract,
    "bc_build": run_bc_build,
    "bc_transfer": run_bc_transfer,
    "bc_enforce": run_bc_enforce,
    "subdivide": run_subdivide,
    "smooth": run_smooth,
    "validate": run_validate,
}


def run_stage(name: str, state, ctx) -> None:
    runner = STAGE_RUNNERS.get(name)
    if runner is None:
        raise ValueError(f"Unknown pipeline stage {name!r}")
    runner(state, ctx)
