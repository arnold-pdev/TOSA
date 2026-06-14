"""Hard BC plane enforcement and footprint audits."""

from __future__ import annotations

import numpy as np

from lib.converters.voxel2surf.stages.bc_transfer import transfer_patches
from lib.converters.voxel2surf.types import PipelineContext, SurfaceState
from lib.meshing.surface_smooth import bc_patch_vertex_map, flatten_bc_planes
from lib.volume_check import mesh_stage_report


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

    # Snap every vertex on a labeled BC triangle onto its patch plane.
    vmap = bc_patch_vertex_map(state.mesh, tri_labels, n_bc_patches)
    flatten_bc_planes(state.mesh, vmap, state.patches)

    # Re-assert triangle vertices on plane (handles mixed free/BC corner verts).
    faces = state.mesh.faces
    verts = state.mesh.vertices
    for fidx, pid in enumerate(tri_labels):
        if pid < 0:
            continue
        patch = state.patches[pid]
        axis = patch.axis
        verts[faces[fidx], axis] = patch.plane_coord

    if ctx.options.bc_strict_footprint:
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
        tri_labels = np.where(keep, tri_labels, -1)
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
    ctx.log(
        f"        BC enforce: {int(state.bc_audit['bc_labeled_triangles']):,} "
        f"labeled tris, plane residual={max_res:.3g}, "
        f"footprint coverage={coverage:.1%}",
    )
    report = mesh_stage_report(
        state.mesh,
        ctx.domain_vol,
        "bc_enforce",
        vf_target=ctx.vf_voxel,
    )
    state.stage_reports.append(report)
    ctx.log(report.format_line())
