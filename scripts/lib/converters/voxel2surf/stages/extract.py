"""Surface extraction backends."""

from __future__ import annotations

import numpy as np

from lib.converters.voxel2surf.types import PipelineContext, SurfaceState
from lib.meshing.field import (
    extract_surface_mesh,
    refine_sdf_field,
    signed_distance_field,
)
from lib.meshing.pyvista_surface import extract_contour_mesh
from lib.volume_check import mesh_stage_report


def run_extract(state: SurfaceState, ctx: PipelineContext) -> None:
    opts = ctx.options
    name = opts.extractor
    ctx.log(f"  extract [{name}]")

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
        state.extract_meta = {"iso_level": opts.iso_level}
    elif name == "sdf_lewiner":
        sdf = signed_distance_field(state.vox, state.spacing)
        iso = 0.0
        meta: dict[str, float] = {"vf_input": float((sdf < 0).mean())}
        if opts.calibrate_vf:
            sdf, iso, cal = refine_sdf_field(
                sdf,
                ctx.vf_voxel,
                smooth_sigma=opts.field_smooth_sigma,
                calibrate=True,
            )
            meta.update({k: float(v) for k, v in cal.items()})
        elif opts.field_smooth_sigma > 0:
            from scipy import ndimage

            sdf = ndimage.gaussian_filter(sdf, sigma=opts.field_smooth_sigma)
            meta["smooth_sigma"] = opts.field_smooth_sigma
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
        meta["iso_level"] = iso
        state.extract_meta = meta
    else:
        raise ValueError(f"Unknown extractor {name!r}")

    state.mesh = mesh
    if state.tri_labels is None:
        state.tri_labels = np.full(len(mesh.faces), -1, dtype=np.int32)
    else:
        state.tri_labels = np.full(len(mesh.faces), -1, dtype=np.int32)

    report = mesh_stage_report(
        mesh,
        ctx.domain_vol,
        "extract",
        vf_target=ctx.vf_voxel,
    )
    state.stage_reports.append(report)
    ctx.log(report.format_line())
