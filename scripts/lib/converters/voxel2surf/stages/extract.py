"""Surface extraction backends (monolithic and free-boundary)."""

from __future__ import annotations

import numpy as np

from lib.converters.voxel2surf.types import PipelineContext, SurfaceState
from lib.meshing.field import (
    apply_bc_dirichlet_sdf,
    extract_surface_mesh,
    refine_sdf_field,
    signed_distance_field,
    smooth_sdf_masked,
    upsample_field,
)
from lib.meshing.pyvista_surface import extract_contour_mesh
from lib.volume_check import mesh_stage_report


def _run_sdf_extract(
    state: SurfaceState,
    ctx: PipelineContext,
    *,
    mask_bc: bool,
    pin_bc_planes: bool = False,
    default_field_smooth: bool = False,
) -> tuple[object, dict[str, float]]:
    opts = ctx.options
    vox = state.vox
    if opts.upsample_factor > 1:
        vox = upsample_field(vox, opts.upsample_factor)
    sdf = signed_distance_field(vox, state.spacing)
    iso = 0.0
    meta: dict[str, float] = {"vf_input": float((sdf < 0).mean())}
    use_mask = mask_bc or opts.field_bc_mask
    sigma = opts.field_smooth_sigma
    if default_field_smooth and sigma <= 0:
        sigma = 0.35

    if pin_bc_planes and state.patches:
        sdf = apply_bc_dirichlet_sdf(
            sdf,
            state.patches,
            state.origin,
            state.spacing,
            state.shape,
        )
        meta["field_bc_pin"] = 1.0
    elif use_mask and state.patches:
        if sigma > 0:
            sdf = smooth_sdf_masked(
                sdf,
                state.patches,
                state.origin,
                state.spacing,
                state.shape,
                sigma=sigma,
            )
            meta["smooth_sigma"] = sigma
            meta["field_bc_mask"] = 1.0
        else:
            sdf = apply_bc_dirichlet_sdf(
                sdf,
                state.patches,
                state.origin,
                state.spacing,
                state.shape,
            )
            meta["field_bc_mask"] = 1.0
    elif sigma > 0:
        from scipy import ndimage

        sdf = ndimage.gaussian_filter(sdf, sigma=sigma)
        meta["smooth_sigma"] = sigma

    calibrate = opts.calibrate_vf or default_field_smooth
    if calibrate:
        sdf, iso, cal = refine_sdf_field(
            sdf,
            ctx.vf_voxel,
            smooth_sigma=0.0,
            calibrate=True,
        )
        meta.update({k: float(v) for k, v in cal.items()})
    meta["iso_level"] = iso
    mesh, _removed = extract_surface_mesh(
        sdf,
        state.origin,
        state.spacing,
        iso_level=iso,
        repair=opts.repair,
        meshfix_verbose=opts.meshfix_verbose,
        domain_vol=ctx.domain_vol,
        vf_target=ctx.vf_voxel,
        log=ctx.log,
    )
    return mesh, meta


def _run_extractor(state: SurfaceState, ctx: PipelineContext) -> tuple[object, dict[str, float]]:
    opts = ctx.options
    name = opts.extractor

    if name == "pyvista_binary":
        mesh = extract_contour_mesh(
            state.vox,
            state.origin,
            state.spacing,
            iso_level=opts.iso_level,
            repair=opts.repair,
            meshfix_verbose=opts.meshfix_verbose,
            log=ctx.log,
        )
        return mesh, {"iso_level": opts.iso_level}

    if name in ("sdf_lewiner", "sdf_masked", "sdf_field_smooth", "sdf_bc_planes"):
        return _run_sdf_extract(
            state,
            ctx,
            mask_bc=(name == "sdf_masked"),
            pin_bc_planes=(name == "sdf_bc_planes"),
            default_field_smooth=(name == "sdf_field_smooth"),
        )

    raise ValueError(f"Unknown extractor {name!r}")


def run_extract(state: SurfaceState, ctx: PipelineContext) -> None:
    opts = ctx.options
    name = opts.extractor
    ctx.log(f"  extract [{name}]")

    mesh, meta = _run_extractor(state, ctx)
    state.mesh = mesh
    state.extract_meta = meta
    state.tri_labels = np.full(len(mesh.faces), -1, dtype=np.int32)

    report = mesh_stage_report(
        mesh,
        ctx.domain_vol,
        "extract",
        vf_target=ctx.vf_voxel,
    )
    state.stage_reports.append(report)
    ctx.log(report.format_line())

