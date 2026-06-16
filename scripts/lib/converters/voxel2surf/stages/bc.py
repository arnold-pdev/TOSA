"""BC patch build, transfer, enforce, and optional planar cap contour audit."""

from __future__ import annotations

import warnings

import numpy as np
import trimesh

from lib.converters.bc_patch import Patch, _exposed_faces, build_patch, compare_mesh_to_bc_oracle, cuberille_bc_quads
from lib.converters.bc_planar_cap import PlanarCapParams, build_planar_caps_mesh, list_planar_cap_methods
from lib.converters.voxel2surf.bc_transfer import transfer_patches
from lib.converters.voxel2surf.types import PipelineContext, SurfaceState
from lib.meshing.surface_smooth import bc_patch_vertex_map, flatten_bc_planes
from lib.volume_check import mesh_stage_report


def claim_analytic_bc_triangles(
    mesh,
    patches,
    tri_labels: np.ndarray,
    origin,
    spacing,
    shape,
    *,
    band_cells: float = 1.0,
    cos_tol: float = 0.5,
) -> np.ndarray:
    """Label free triangles on NITO analytic BC planes inside the voxel footprint."""
    labels = np.asarray(tri_labels, dtype=np.int32).copy()
    origin = np.asarray(origin, dtype=float)
    spacing = np.asarray(spacing, dtype=float)
    shape = np.asarray(shape, dtype=int)
    ndim = int(shape.size)
    verts = mesh.vertices
    centroids = mesh.triangles_center
    fnormals = mesh.face_normals
    cells = np.clip(
        np.floor((centroids - origin) / spacing).astype(int),
        0,
        shape - 1,
    )

    for pid, patch in enumerate(patches):
        if not patch.is_bc or not patch.faces:
            continue
        axis = patch.axis
        inplane = [i for i in range(ndim) if i != axis]
        foot = {tuple(f) for f in patch.faces}
        band = float(band_cells * spacing[axis])
        n_hat = np.asarray(patch.normal[:ndim], dtype=float)

        for fidx, face in enumerate(mesh.faces):
            if labels[fidx] >= 0:
                continue
            tri_verts = verts[face]
            deltas = np.abs(tri_verts[:, axis] - patch.plane_coord)
            if float(np.max(deltas)) > band:
                continue
            if float(fnormals[fidx] @ n_hat) <= cos_tol:
                continue
            foot_cell = tuple(cells[fidx, j] for j in inplane)
            if foot_cell not in foot:
                continue
            labels[fidx] = pid
    return labels


def _triangle_plane_residuals(
    mesh,
    tri_labels: np.ndarray,
    patches,
) -> tuple[float, int]:
    """Max |coord - plane_coord| on labeled BC triangles."""
    centroids = mesh.triangles_center
    max_res = 0.0
    n_off = 0
    for pid, patch in enumerate(patches):
        if not patch.is_bc:
            continue
        mask = tri_labels == pid
        if not np.any(mask):
            continue
        axis = patch.axis
        delta = np.abs(centroids[mask, axis] - patch.plane_coord)
        max_res = max(max_res, float(np.max(delta)))
        n_off += int(np.count_nonzero(delta > 1e-6))
    return max_res, n_off


def _footprint_coverage(
    mesh,
    tri_labels: np.ndarray,
    patches,
    origin,
    spacing,
    shape,
) -> float:
    """Fraction of patch footprint cells with at least one labeled triangle centroid."""
    if not patches:
        return 1.0
    shape = np.asarray(shape, dtype=int).ravel()
    ndim = int(shape.size)
    origin = np.asarray(origin, dtype=float)
    spacing = np.asarray(spacing, dtype=float)
    centroids = mesh.triangles_center
    cells = np.clip(
        np.floor((centroids - origin) / spacing).astype(int),
        0,
        shape - 1,
    )
    covered = 0
    total = 0
    for pid, patch in enumerate(patches):
        if not patch.is_bc or not patch.faces:
            continue
        axis = patch.axis
        inplane = [i for i in range(ndim) if i != axis]
        labeled = tri_labels == pid
        if not np.any(labeled):
            total += len(patch.faces)
            continue
        hit_cells = {
            tuple(cells[i, j] for j in inplane)
            for i in np.where(labeled)[0]
        }
        for foot in patch.faces:
            total += 1
            if tuple(foot) in hit_cells:
                covered += 1
    return 1.0 if total == 0 else covered / total


def _per_patch_tri_counts(tri_labels: np.ndarray, patches) -> dict[str, float]:
    out: dict[str, float] = {}
    counts = []
    for pid, patch in enumerate(patches):
        if not patch.is_bc:
            continue
        n = int(np.count_nonzero(np.asarray(tri_labels, dtype=int) == pid))
        counts.append(n)
        out[f"bc_patch_{pid}_tris"] = float(n)
    out["bc_min_patch_tris"] = float(min(counts)) if counts else 0.0
    return out


def format_bc_coplanarity_failure(
    *,
    max_residual: float,
    tol: float,
    n_off_vertices: int,
    n_labeled: int,
) -> str:
    return (
        "NITO BC planes not coplanar on output mesh "
        f"(bc_plane_max_residual={max_residual:.3g} > tol={tol:.3g}; "
        f"{n_off_vertices} off-plane triangle placements on {n_labeled:,} BC triangles)"
    )


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


def run_bc_transfer(state: SurfaceState, ctx: PipelineContext) -> None:
    if not state.patches:
        state.tri_labels = np.full(len(state.mesh.faces), -1, dtype=np.int32)
        return

    state.tri_labels = transfer_patches(
        state.mesh,
        state.patches,
        state.origin,
        state.spacing,
        state.shape,
        band_cells=ctx.options.bc_transfer_band_cells,
        method=ctx.options.bc_transfer,
    ).astype(np.int32)
    n_bc = int(np.count_nonzero(state.tri_labels >= 0))
    ctx.log(
        f"        {n_bc:,} BC triangles, "
        f"{len(state.tri_labels) - n_bc:,} free",
    )
    report = mesh_stage_report(
        state.mesh,
        ctx.domain_vol,
        "bc_transfer",
        vf_target=ctx.vf_voxel,
    )
    state.stage_reports.append(report)
    ctx.log(report.format_line())


def run_bc_enforce(state: SurfaceState, ctx: PipelineContext) -> None:
    if not state.patches or state.tri_labels is None:
        return

    tri_labels = np.asarray(state.tri_labels, dtype=np.int32)
    n_bc_patches = len(state.patches)
    opts = ctx.options

    if opts.bc_claim_coplanar:
        before = int(np.count_nonzero(tri_labels >= 0))
        tri_labels = claim_analytic_bc_triangles(
            state.mesh,
            state.patches,
            tri_labels,
            state.origin,
            state.spacing,
            state.shape,
            band_cells=opts.bc_transfer_band_cells,
        )
        claimed = int(np.count_nonzero(tri_labels >= 0)) - before
        if claimed > 0:
            ctx.log(f"        claimed {claimed:,} coplanar BC triangles on NITO planes")

    vmap = bc_patch_vertex_map(state.mesh, tri_labels, n_bc_patches)
    flatten_bc_planes(state.mesh, vmap, state.patches)

    faces = state.mesh.faces
    verts = state.mesh.vertices
    for fidx, pid in enumerate(tri_labels):
        if pid < 0:
            continue
        patch = state.patches[pid]
        verts[faces[fidx], patch.axis] = patch.plane_coord

    if opts.bc_strict_footprint:
        centroids = state.mesh.triangles_center
        cells = np.clip(
            np.floor((centroids - state.origin) / state.spacing).astype(int),
            0,
            state.shape - 1,
        )
        keep = np.ones(len(tri_labels), dtype=bool)
        ndim = int(state.shape.size)
        for fidx, pid in enumerate(tri_labels):
            if pid < 0:
                continue
            patch = state.patches[pid]
            inplane = [i for i in range(ndim) if i != patch.axis]
            foot = tuple(cells[fidx, j] for j in inplane)
            if foot not in {tuple(f) for f in patch.faces}:
                keep[fidx] = False
        tri_labels = np.where(keep, tri_labels, -1).astype(np.int32)
        state.tri_labels = tri_labels

    state.tri_labels = tri_labels
    state.freeze = (vmap >= 0).astype(float)
    max_res, n_off_plane = _triangle_plane_residuals(
        state.mesh, tri_labels, state.patches,
    )
    coverage = _footprint_coverage(
        state.mesh,
        tri_labels,
        state.patches,
        state.origin,
        state.spacing,
        state.shape,
    )
    state.bc_audit = {
        "bc_plane_max_residual": max_res,
        "bc_off_plane_triangles": float(n_off_plane),
        "bc_footprint_coverage": coverage,
        "bc_labeled_triangles": float(np.count_nonzero(tri_labels >= 0)),
        "bc_patch_vertices": float(np.count_nonzero(vmap >= 0)),
    }
    per_patch = _per_patch_tri_counts(tri_labels, state.patches)
    state.bc_audit.update(per_patch)

    if opts.bc_oracle and state.patches:
        oracle = cuberille_bc_quads(
            state.patches,
            state.origin,
            state.spacing,
        )
        oracle_metrics = compare_mesh_to_bc_oracle(
            state.mesh,
            tri_labels,
            oracle,
            state.patches,
        )
        state.bc_audit.update(oracle_metrics)
        ctx.log(
            f"        BC oracle gap: max={oracle_metrics['bc_oracle_max_gap']:.3g} "
            f"mean={oracle_metrics['bc_oracle_mean_gap']:.3g}",
        )

    ctx.log(
        f"        BC enforce: {int(state.bc_audit['bc_labeled_triangles']):,} "
        f"labeled tris, plane residual={max_res:.3g}, "
        f"footprint coverage={coverage:.1%}",
    )

    if max_res > opts.bc_plane_tol + 1e-12:
        message = format_bc_coplanarity_failure(
            max_residual=max_res,
            tol=opts.bc_plane_tol,
            n_off_vertices=n_off_plane,
            n_labeled=int(np.count_nonzero(tri_labels >= 0)),
        )
        if opts.on_bc_fail == "raise":
            raise ValueError(message)
        warnings.warn(message, stacklevel=2)
        ctx.log(f"        BC coplanarity: FAILED (on_bc_fail={opts.on_bc_fail})")

    report = mesh_stage_report(
        state.mesh,
        ctx.domain_vol,
        "bc_enforce",
        vf_target=ctx.vf_voxel,
    )
    state.stage_reports.append(report)
    ctx.log(report.format_line())


def run_bc_planar_cap(state: SurfaceState, ctx: PipelineContext) -> None:
    """Build planar caps from voxel footprints (audit only — not assembled onto the skin)."""
    opts = ctx.options
    if not opts.bc_planar_cap:
        return
    if not state.patches:
        ctx.log("  bc_planar_cap — skipped (no patches)")
        return

    method = opts.bc_planar_cap_method
    ctx.log(f"  bc_planar_cap [{method}] audit-only")
    params = PlanarCapParams(
        upsample_factor=opts.bc_planar_cap_upsample,
        blur_sigma_cells=opts.bc_planar_cap_blur_sigma,
        outward_buffer_cells=opts.bc_planar_cap_outward_buffer,
    )

    bc_mesh, audit = build_planar_caps_mesh(
        state.patches,
        state.origin,
        state.spacing,
        state.shape,
        method=method,
        params=params,
    )
    state.bc_mesh = bc_mesh
    state.bc_audit.update({f"planar_cap_{k}": v for k, v in audit.items()})

    n_bc = len(bc_mesh.faces) if bc_mesh is not None else 0
    ok = int(audit.get("containment_ok", 0))
    ctx.log(
        f"        method={method} caps={audit.get('n_caps', 0)} "
        f"cap_tris={n_bc:,} containment_ok={bool(ok)}",
    )
    if bc_mesh is not None:
        state.bc_audit["planar_cap_watertight"] = int(bool(bc_mesh.is_watertight))
        state.bc_audit["planar_cap_bodies"] = (
            int(bc_mesh.body_count) if hasattr(bc_mesh, "body_count") else -1
        )
        ctx.log(
            f"        cap mesh: watertight={bool(bc_mesh.is_watertight)} "
            f"bodies={state.bc_audit['planar_cap_bodies']}",
        )

    if not ok and opts.on_bc_fail == "raise":
        raise ValueError(
            f"bc_planar_cap containment failed ({audit.get('n_corners_violating', '?')} corners)",
        )
    if not ok:
        ctx.log(
            "        planar cap containment WARN: "
            f"{audit.get('n_corners_violating', '?')} footprint corners outside boundary",
        )

    if bc_mesh is not None:
        report = mesh_stage_report(
            bc_mesh,
            ctx.domain_vol,
            "bc_planar_cap",
            vf_target=ctx.vf_voxel,
        )
        state.stage_reports.append(report)
        ctx.log(report.format_line())


def planar_cap_method_choices() -> str:
    return ", ".join(list_planar_cap_methods())
