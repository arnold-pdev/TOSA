"""Subdivide, smooth, snap, and BC regrow post-processing."""

from __future__ import annotations

import numpy as np

from lib.converters.patch_regrow import regrow_patches_on_mesh
from lib.converters.voxel2surf.bc_transfer import transfer_patches
from lib.converters.voxel2surf.stages.validate import _log_load_surface_checks
from lib.converters.voxel2surf.types import PipelineContext, SurfaceState
from lib.meshing.field import signed_distance_field, snap_mesh_vertices_to_sdf
from lib.meshing.surface_smooth import (
    bc_patch_freeze_weights,
    constrained_laplacian_smooth,
    masked_hc_smooth,
    masked_taubin_smooth,
    subdivide_mesh,
    subdivide_mesh_free_only,
    tangential_taubin_smooth,
    taubin_mu_from_passband,
)
from lib.volume_check import mesh_stage_report


def run_subdivide(state: SurfaceState, ctx: PipelineContext) -> None:
    levels = ctx.options.subdivide_levels
    if levels <= 0:
        return
    ctx.log(f"  subdivide — levels={levels}")

    if (
        ctx.options.subdivide_free_only
        and state.tri_labels is not None
        and state.patches
    ):
        state.mesh, state.tri_labels = subdivide_mesh_free_only(
            state.mesh,
            state.tri_labels,
            levels,
        )
    else:
        state.mesh = subdivide_mesh(state.mesh, levels)
        if state.patches:
            state.tri_labels = transfer_patches(
                state.mesh,
                state.patches,
                state.origin,
                state.spacing,
                state.shape,
                band_cells=ctx.options.bc_transfer_band_cells,
                method=ctx.options.bc_transfer,
            ).astype(int)

    report = mesh_stage_report(
        state.mesh,
        ctx.domain_vol,
        "subdivide",
        vf_target=ctx.vf_voxel,
    )
    state.stage_reports.append(report)
    ctx.log(report.format_line())


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

    weights = _smooth_weights(state, ctx)
    free_only = ctx.options.smooth_free_only
    taubin_mu = (
        ctx.options.taubin_mu
        if ctx.options.taubin_mu is not None
        else taubin_mu_from_passband(ctx.options.taubin_lambda, ctx.options.taubin_k_pb)
    )

    if smoother == "laplacian_bc":
        iters = ctx.options.laplacian_iters
        if iters <= 0:
            return
        ctx.log(
            f"  smooth [laplacian_bc] — iters={iters} "
            f"relax={ctx.options.laplacian_relaxation}"
            + (" free_only" if free_only else ""),
        )
        constrained_laplacian_smooth(
            state.mesh,
            weights,
            state.patches,
            state.tri_labels,
            iterations=iters,
            relaxation=ctx.options.laplacian_relaxation,
            constrain_bc_planes=ctx.options.constrain_bc_planes,
            smooth_free_only=free_only,
        )
        stage = "laplacian"
    elif smoother == "taubin_bc":
        iters = ctx.options.taubin_iters
        if iters <= 0:
            return
        ctx.log(
            f"  smooth [taubin_bc] — iters={iters} "
            f"λ={ctx.options.taubin_lambda} μ={taubin_mu:.4f} k_PB={ctx.options.taubin_k_pb}"
            + (" free_only" if free_only else ""),
        )
        masked_taubin_smooth(
            state.mesh,
            weights,
            state.patches,
            state.tri_labels,
            iterations=iters,
            lamb=ctx.options.taubin_lambda,
            mu=taubin_mu,
            constrain_bc_planes=ctx.options.constrain_bc_planes,
            smooth_free_only=free_only,
        )
        stage = "taubin"
    elif smoother == "hc_bc":
        iters = ctx.options.hc_iters
        if iters <= 0:
            return
        ctx.log(
            f"  smooth [hc_bc] — iters={iters} "
            f"λ={ctx.options.hc_lambda} α={ctx.options.hc_alpha}"
            + (" free_only" if free_only else ""),
        )
        masked_hc_smooth(
            state.mesh,
            weights,
            state.patches,
            state.tri_labels,
            iterations=iters,
            lamb=ctx.options.hc_lambda,
            alpha=ctx.options.hc_alpha,
            constrain_bc_planes=ctx.options.constrain_bc_planes,
            smooth_free_only=free_only,
        )
        stage = "hc"
    elif smoother == "tangential_bc":
        iters = ctx.options.taubin_iters
        if iters <= 0:
            return
        ctx.log(
            f"  smooth [tangential_bc] — iters={iters} "
            f"λ={ctx.options.taubin_lambda} μ={taubin_mu:.4f} k_PB={ctx.options.taubin_k_pb}"
            + (" free_only" if free_only else ""),
        )
        tangential_taubin_smooth(
            state.mesh,
            weights,
            state.patches,
            state.tri_labels,
            iterations=iters,
            lamb=ctx.options.taubin_lambda,
            mu=taubin_mu,
            constrain_bc_planes=ctx.options.constrain_bc_planes,
            smooth_free_only=free_only,
        )
        stage = "tangential"
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


def run_snap(state: SurfaceState, ctx: PipelineContext) -> None:
    if not ctx.options.snap_to_sdf:
        return

    ctx.log("  snap [sdf_reproject]")
    sdf = signed_distance_field(state.vox, state.spacing)
    weights = None
    if state.patches and state.tri_labels is not None:
        weights, _, _ = bc_patch_freeze_weights(
            state.mesh,
            state.tri_labels,
            len(state.patches),
            transition_rings=ctx.options.laplacian_freeze_rings,
        )
    n = snap_mesh_vertices_to_sdf(
        state.mesh,
        sdf,
        state.origin,
        state.spacing,
        weights=weights,
        steps=ctx.options.snap_to_sdf_steps,
    )
    ctx.log(f"        snapped {n:,} vertices toward SDF iso-surface")
    report = mesh_stage_report(
        state.mesh,
        ctx.domain_vol,
        "snap",
        vf_target=ctx.vf_voxel,
    )
    state.stage_reports.append(report)
    ctx.log(report.format_line())


def run_bc_regrow(state: SurfaceState, ctx: PipelineContext) -> None:
    mode = ctx.options.bc_regrow
    if mode == "none" or not state.patches or state.tri_labels is None:
        return
    if ctx.bc_rows is None:
        ctx.log("  bc_regrow — skipped (no bc_rows)")
        return

    ctx.log(f"  bc_regrow [{mode}]")
    if state.tri_labels_voxel is None:
        state.tri_labels_voxel = np.asarray(state.tri_labels, dtype=np.int32).copy()

    state.tri_labels = regrow_patches_on_mesh(
        state.mesh,
        state.patches,
        ctx.bc_rows,
        state.shape,
        state.spacing,
        origin=state.origin,
        tri_labels_voxel=state.tri_labels_voxel,
        log=ctx.log,
    ).astype(np.int32)

    n_bc = int(np.count_nonzero(state.tri_labels >= 0))
    ctx.log(f"        regrow: {n_bc:,} BC triangles")
    report = mesh_stage_report(
        state.mesh,
        ctx.domain_vol,
        "bc_regrow",
        vf_target=ctx.vf_voxel,
    )
    state.stage_reports.append(report)
    ctx.log(report.format_line())
