"""BC assembly backends (parallel alternatives to monolithic MC BC faces)."""

from __future__ import annotations

import numpy as np

from lib.converters.voxel2surf.types import PipelineContext, SurfaceState
from lib.meshing.shared_seam import (
    SharedSeamAudit,
    assemble_shared_seam_mesh,
    strip_labeled_bc_faces,
)
from lib.volume_check import mesh_stage_report


def run_bc_assembly_monolithic(_state: SurfaceState, _ctx: PipelineContext) -> None:
    """No-op — BC faces remain on the extracted iso-surface."""


def run_bc_assembly_shared_seam(state: SurfaceState, ctx: PipelineContext) -> None:
    if not state.patches or state.tri_labels is None:
        ctx.log("  bc_assembly [shared_seam] — skipped (no patches/labels)")
        return

    opts = ctx.options
    tri_labels = np.asarray(state.tri_labels, dtype=np.int32)
    free_mesh = strip_labeled_bc_faces(state.mesh, tri_labels)
    n_stripped = len(state.mesh.faces) - len(free_mesh.faces)
    ctx.log(
        f"  bc_assembly [shared_seam] — stripped {n_stripped:,} BC tris, "
        f"free skin {len(free_mesh.faces):,} faces",
    )

    mesh, labels, audit = assemble_shared_seam_mesh(
        free_mesh,
        state.patches,
        state.origin,
        state.spacing,
        state.shape,
        plane_tol=opts.bc_plane_tol,
    )
    state.mesh = mesh
    state.tri_labels = labels.astype(np.int32)
    _record_audit(state, audit)
    ctx.log(
        f"        caps={audit.n_caps_built} seam_segments={audit.n_seam_segments} "
        f"loops={audit.n_seam_loops} fallback_caps={audit.n_fallback_caps} "
        f"merged_verts={audit.seam_verts_merged} "
        f"watertight={bool(audit.meta.get('watertight', 0))} "
        f"bodies={audit.meta.get('bodies', '?')}",
    )
    report = mesh_stage_report(
        mesh,
        ctx.domain_vol,
        "bc_assembly",
        vf_target=ctx.vf_voxel,
    )
    state.stage_reports.append(report)
    ctx.log(report.format_line())


def _record_audit(state: SurfaceState, audit: SharedSeamAudit) -> None:
    state.bc_audit["bc_assembly_method"] = 1.0  # shared_seam tag for metrics CSV
    state.bc_audit["shared_seam_n_caps"] = float(audit.n_caps_built)
    state.bc_audit["shared_seam_n_segments"] = float(audit.n_seam_segments)
    state.bc_audit["shared_seam_n_loops"] = float(audit.n_seam_loops)
    state.bc_audit["shared_seam_n_fallback"] = float(audit.n_fallback_caps)
    state.bc_audit["shared_seam_verts_merged"] = float(audit.seam_verts_merged)
    for k, v in audit.meta.items():
        state.bc_audit[f"shared_seam_{k}"] = v


BC_ASSEMBLY_RUNNERS = {
    "monolithic": run_bc_assembly_monolithic,
    "shared_seam": run_bc_assembly_shared_seam,
}


def run_bc_assembly(state: SurfaceState, ctx: PipelineContext) -> None:
    method = ctx.options.bc_assembly.strip().lower()
    runner = BC_ASSEMBLY_RUNNERS.get(method)
    if runner is None:
        choices = ", ".join(sorted(BC_ASSEMBLY_RUNNERS))
        raise ValueError(f"Unknown bc_assembly {method!r}. Choose: {choices}")
    if method == "monolithic":
        return
    runner(state, ctx)
