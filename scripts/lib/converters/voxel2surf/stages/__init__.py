"""Stage dispatch for the voxel2surf pipeline."""

from __future__ import annotations

from lib.converters.voxel2surf.stages.bc_assembly import run_bc_assembly
from lib.converters.voxel2surf.stages.bc import (
    run_bc_build,
    run_bc_enforce,
    run_bc_planar_cap,
    run_bc_transfer,
)
from lib.converters.voxel2surf.stages.extract import run_extract
from lib.converters.voxel2surf.stages.refine import (
    run_bc_regrow,
    run_smooth,
    run_snap,
    run_subdivide,
)
from lib.converters.voxel2surf.stages.validate import run_validate

STAGE_RUNNERS = {
    "extract": run_extract,
    "bc_build": run_bc_build,
    "bc_transfer": run_bc_transfer,
    "bc_enforce": run_bc_enforce,
    "bc_assembly": run_bc_assembly,
    "bc_planar_cap": run_bc_planar_cap,
    "bc_regrow": run_bc_regrow,
    "subdivide": run_subdivide,
    "smooth": run_smooth,
    "snap": run_snap,
    "validate": run_validate,
}


def run_stage(name: str, state, ctx) -> None:
    runner = STAGE_RUNNERS.get(name)
    if runner is None:
        raise ValueError(f"Unknown pipeline stage {name!r}")
    runner(state, ctx)
