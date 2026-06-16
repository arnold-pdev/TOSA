"""Final validation: volume, BC audits, load surface checks."""

from __future__ import annotations

import warnings

import numpy as np

from lib.converters.voxel2surf.types import PipelineContext, SurfaceState
from lib.meshing.smoothness_metrics import free_boundary_smoothness
from lib.nito_physics import (
    failed_load_surface_checks,
    format_load_surface_failure,
    load_surface_checks,
    load_surface_tolerance,
)
from lib.volume_check import mesh_stage_report, volume_fraction_mesh


def _log_load_surface_checks(
    log,
    mesh,
    load_rows,
    shape,
    spacing,
    *,
    stage: str,
    max_distance_cells: float,
):
    tol = load_surface_tolerance(shape, spacing, max_distance_cells)
    checks = load_surface_checks(mesh, load_rows, shape, spacing=spacing)
    log(f"        load surface ({stage}), tol={max_distance_cells} cells")
    for check in checks:
        a = check.nito_anchor
        status = (
            "ok"
            if check.distance <= tol + 1e-12
            else f"FAIL>{max_distance_cells:g}cells"
        )
        log(
            f"        load {check.index} [{status}]: distance={check.distance:.4g} "
            f"nito=({a[0]:.3g}, {a[1]:.3g}, {a[2]:.3g})",
        )
    bad = failed_load_surface_checks(checks, tol)
    if bad and checks:
        deltas = [c.distance - tol for c in bad]
        log(
            f"        load surface ({stage}): {len(bad)}/{len(checks)} "
            f"over tolerance (max overage={max(deltas):.4g})",
        )
    return checks


def run_validate(state: SurfaceState, ctx: PipelineContext) -> None:
    ctx.log("  validate")
    final = mesh_stage_report(
        state.mesh,
        ctx.domain_vol,
        "final",
        vf_target=ctx.vf_voxel,
    )
    state.stage_reports.append(final)
    ctx.log(final.format_line())

    vf_mesh = volume_fraction_mesh(state.mesh, ctx.domain_vol)
    cell_vol = float(np.prod(state.spacing))
    vf_tol = ctx.options.vf_tol_cells * cell_vol / ctx.domain_vol
    if vf_mesh is not None:
        vf_delta = vf_mesh - ctx.vf_voxel
        state.bc_audit["vf_mesh"] = vf_mesh
        state.bc_audit["vf_delta"] = vf_delta
        state.bc_audit["vf_tol"] = vf_tol
        status = "ok" if abs(vf_delta) <= vf_tol + 1e-12 else "WARN"
        ctx.log(
            f"        volume fraction [{status}]: mesh={vf_mesh:.6f} "
            f"voxel={ctx.vf_voxel:.6f} delta={vf_delta:+.6f} "
            f"(tol={ctx.options.vf_tol_cells:g} cells)",
        )

    if state.bc_audit.get("bc_plane_max_residual", 0.0) > ctx.options.bc_plane_tol + 1e-12:
        ctx.log(
            "        BC plane residual WARN: "
            f"{state.bc_audit['bc_plane_max_residual']:.3g} "
            f"(tol={ctx.options.bc_plane_tol:.3g})",
        )

    min_patch = state.bc_audit.get("bc_min_patch_tris")
    if min_patch is not None:
        ctx.log(f"        bc_min_patch_tris={int(min_patch)}")
        if min_patch == 0 and state.patches:
            ctx.log("        BC patch WARN: at least one patch has 0 labeled triangles")

    if state.bc_audit.get("bc_oracle_max_gap", 0.0) > 0.1:
        ctx.log(
            "        BC oracle gap WARN: "
            f"{state.bc_audit['bc_oracle_max_gap']:.3g}",
        )

    if state.tri_labels is not None:
        smooth = free_boundary_smoothness(state.mesh, state.tri_labels)
        state.bc_audit.update(smooth)
        if not np.isnan(smooth["free_dihedral_mean_deg"]):
            ctx.log(
                "        free boundary smoothness: "
                f"dihedral_mean={smooth['free_dihedral_mean_deg']:.2f}° "
                f"p95={smooth['free_dihedral_p95_deg']:.2f}° "
                f"axis_aligned_edges={smooth['axis_aligned_edge_frac']:.1%}",
            )

    has_loads = (
        ctx.options.check_loads
        and ctx.load_rows is not None
        and np.atleast_2d(ctx.load_rows).size > 0
    )
    if has_loads:
        state.load_checks = _log_load_surface_checks(
            ctx.log,
            state.mesh,
            ctx.load_rows,
            state.shape,
            state.spacing,
            stage="final",
            max_distance_cells=ctx.options.load_surface_tol_cells,
        )
        tol = load_surface_tolerance(
            state.shape, state.spacing, ctx.options.load_surface_tol_cells,
        )
        bad = failed_load_surface_checks(state.load_checks, tol)
        if bad:
            message = format_load_surface_failure(
                bad,
                max_distance_cells=ctx.options.load_surface_tol_cells,
                tol=tol,
            )
            state.load_check_failed = True
            if ctx.options.on_load_fail == "raise":
                raise ValueError(message)
            warnings.warn(message, stacklevel=2)
            ctx.log(
                f"        load surface check: FAILED "
                f"(on_load_fail={ctx.options.on_load_fail}, export continues)",
            )
