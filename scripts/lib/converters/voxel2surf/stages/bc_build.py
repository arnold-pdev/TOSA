"""Build NITO BC patches from voxel specs."""

from __future__ import annotations

import numpy as np

from lib.converters.bc_patch import BCSpec, Patch, _exposed_faces, build_patch
from lib.converters.voxel2surf.types import PipelineContext, SurfaceState
from lib.volume_check import mesh_stage_report


def run_bc_build(state: SurfaceState, ctx: PipelineContext) -> None:
    if not state.specs:
        state.patches = []
        state.tri_labels = np.full(len(state.mesh.faces), -1, dtype=np.int32)
        return

    ctx.log(f"  BC patches — {len(state.specs)} spec(s)")
    exposed = _exposed_faces(state.vox)
    patches: list[Patch] = []
    for spec in state.specs:
        patch = build_patch(
            state.vox,
            spec,
            state.origin,
            state.spacing,
            exposed=exposed,
            bc_rows=ctx.bc_rows,
            shape=state.shape,
        )
        if patch.faces:
            patches.append(patch)
            ctx.log(
                f"        patch {len(patches) - 1}: axis={patch.axis} "
                f"side={patch.side} cells={len(patch.faces)}",
            )
    state.patches = patches
    report = mesh_stage_report(
        state.mesh,
        ctx.domain_vol,
        "bc_build",
        vf_target=ctx.vf_voxel,
    )
    state.stage_reports.append(report)
    ctx.log(report.format_line())
