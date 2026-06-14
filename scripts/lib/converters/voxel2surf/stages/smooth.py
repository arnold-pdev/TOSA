"""Surface smoothing backends with BC vertex freeze."""

from __future__ import annotations

import numpy as np

from lib.converters.voxel2surf.stages.validate import _log_load_surface_checks
from lib.converters.voxel2surf.types import PipelineContext, SurfaceState
from lib.meshing.surface_smooth import (
    bc_patch_freeze_weights,
    constrained_laplacian_smooth,
    masked_taubin_smooth,
)
from lib.volume_check import mesh_stage_report


def _smooth_weights(state: SurfaceState, ctx: PipelineContext):
    if state.patches and state.tri_labels is not None and np.any(state.tri_labels >= 0):
        weights, n_bc_tris, n_frozen = bc_patch_freeze_weights(
            state.mesh,
            state.tri_labels,
            len(state.patches),
            transition_rings=ctx.options.laplacian_freeze_rings,
        )
        ctx.log(
            f"        freeze BC patch vertices: {n_bc_tris:,} triangles, "
            f"{n_frozen:,} vertices",
        )
        return weights
    return np.zeros(len(state.mesh.vertices), dtype=float)


def run_smooth(state: SurfaceState, ctx: PipelineContext) -> None:
    smoother = ctx.options.smoother
    if smoother == "none":
        return

    has_loads = (
        ctx.options.check_loads
        and ctx.load_rows is not None
        and np.atleast_2d(ctx.load_rows).size > 0
    )
    if has_loads:
        _log_load_surface_checks(
            ctx.log,
            state.mesh,
            ctx.load_rows,
            state.shape,
            state.spacing,
            stage="pre-smooth",
            max_distance_cells=ctx.options.load_surface_tol_cells,
        )

    if smoother == "laplacian_bc":
        iters = ctx.options.laplacian_iters
        if iters <= 0:
            return
        ctx.log(
            f"  smooth [laplacian_bc] — iters={iters} "
            f"relax={ctx.options.laplacian_relaxation}",
        )
        weights = _smooth_weights(state, ctx)
        constrained_laplacian_smooth(
            state.mesh,
            weights,
            state.patches,
            state.tri_labels,
            iterations=iters,
            relaxation=ctx.options.laplacian_relaxation,
            constrain_bc_planes=ctx.options.constrain_bc_planes,
        )
        stage = "laplacian"
    elif smoother == "taubin_bc":
        iters = ctx.options.taubin_iters
        if iters <= 0:
            return
        ctx.log(
            f"  smooth [taubin_bc] — iters={iters} "
            f"λ={ctx.options.taubin_lambda} μ={ctx.options.taubin_nu}",
        )
        weights = _smooth_weights(state, ctx)
        masked_taubin_smooth(
            state.mesh,
            weights,
            state.patches,
            state.tri_labels,
            iterations=iters,
            lamb=ctx.options.taubin_lambda,
            nu=ctx.options.taubin_nu,
            constrain_bc_planes=ctx.options.constrain_bc_planes,
        )
        stage = "taubin"
    else:
        raise ValueError(f"Unknown smoother {smoother!r}")

    report = mesh_stage_report(
        state.mesh,
        ctx.domain_vol,
        stage,
        vf_target=ctx.vf_voxel,
    )
    state.stage_reports.append(report)
    ctx.log(report.format_line())
