"""Optional mesh subdivision."""

from __future__ import annotations

from lib.converters.voxel2surf.stages.bc_transfer import transfer_patches
from lib.converters.voxel2surf.types import PipelineContext, SurfaceState
from lib.meshing.surface_smooth import subdivide_mesh
from lib.volume_check import mesh_stage_report


def run_subdivide(state: SurfaceState, ctx: PipelineContext) -> None:
    levels = ctx.options.subdivide_levels
    if levels <= 0:
        return
    ctx.log(f"  subdivide — levels={levels}")
    state.mesh = subdivide_mesh(state.mesh, levels)
    if state.patches:
        state.tri_labels = transfer_patches(
            state.mesh,
            state.patches,
            state.origin,
            state.spacing,
            state.shape,
        ).astype(np.int32)
    report = mesh_stage_report(
        state.mesh,
        ctx.domain_vol,
        "subdivide",
        vf_target=ctx.vf_voxel,
    )
    state.stage_reports.append(report)
    ctx.log(report.format_line())
