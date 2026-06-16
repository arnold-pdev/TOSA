"""BC patch transfer from voxel footprints onto mesh triangles."""

from __future__ import annotations

import numpy as np
import trimesh

from lib.converters.bc_patch import Patch


def transfer_cell_native(
    mesh: trimesh.Trimesh,
    patches: list[Patch],
    origin,
    spacing,
    vox_shape,
    *,
    band_cells: float = 1.0,
    cos_tol: float = 0.5,
    min_verts_in_footprint: int = 2,
) -> np.ndarray:
    """
    Label triangles when multiple vertices fall inside a patch footprint cell.

    Stricter than centroid transfer for jagged marching-cubes meshes.
    """
    origin = np.asarray(origin, float)
    spacing = np.asarray(spacing, float)
    vox_shape = np.asarray(vox_shape, dtype=int)
    ndim = int(vox_shape.size)
    verts = mesh.vertices
    vert_cells = np.clip(
        np.floor((verts - origin) / spacing).astype(int),
        0,
        vox_shape - 1,
    )
    fnormals = mesh.face_normals
    centroids = mesh.triangles_center
    labels = np.full(len(mesh.faces), -1, dtype=int)

    for pid, patch in enumerate(patches):
        if not patch.is_bc:
            continue
        axis = patch.axis
        inplane = [i for i in range(ndim) if i != axis]
        foot = {tuple(f) for f in patch.faces}
        band = band_cells * spacing[axis]
        ref = (
            patch.plane_coord
            if patch.side == "min"
            else patch.plane_coord - spacing[axis]
        )
        near = np.abs(centroids[:, axis] - ref) < band
        aligned = fnormals @ np.asarray(patch.normal[:ndim]) > cos_tol

        for fidx, face in enumerate(mesh.faces):
            if labels[fidx] >= 0:
                continue
            if not (near[fidx] and aligned[fidx]):
                continue
            hits = 0
            for vid in face:
                inplane_cell = tuple(vert_cells[vid, j] for j in inplane)
                if inplane_cell in foot:
                    hits += 1
            if hits >= min_verts_in_footprint:
                labels[fidx] = pid
    return labels


def transfer_patches(
    mesh: trimesh.Trimesh,
    patches: list[Patch],
    origin,
    spacing,
    vox_shape,
    band_cells: float = 1.0,
    cos_tol: float = 0.5,
    *,
    method: str = "centroid",
) -> np.ndarray:
    """Per-triangle patch id (index into ``patches``), or -1 for free boundary."""
    if method == "cell_native":
        return transfer_cell_native(
            mesh,
            patches,
            origin,
            spacing,
            vox_shape,
            band_cells=band_cells,
            cos_tol=cos_tol,
        )
    origin = np.asarray(origin, float)
    spacing = np.asarray(spacing, float)
    vox_shape = np.asarray(vox_shape, dtype=int)
    ndim = int(vox_shape.size)
    centroids = mesh.triangles_center
    fnormals = mesh.face_normals
    cells = np.clip(
        np.floor((centroids - origin) / spacing).astype(int),
        0,
        vox_shape - 1,
    )

    labels = np.full(len(centroids), -1, dtype=int)
    for pid, patch in enumerate(patches):
        if not patch.is_bc:
            continue
        axis = patch.axis
        inplane = [i for i in range(ndim) if i != axis]
        foot = {tuple(f) for f in patch.faces}
        band = band_cells * spacing[axis]
        ref = (
            patch.plane_coord
            if patch.side == "min"
            else patch.plane_coord - spacing[axis]
        )
        near = np.abs(centroids[:, axis] - ref) < band
        aligned = fnormals @ np.asarray(patch.normal[:ndim]) > cos_tol
        in_foot = np.fromiter(
            (
                tuple(cells[i, j] for j in inplane) in foot
                for i in range(len(cells))
            ),
            dtype=bool,
            count=len(cells),
        )
        member = near & aligned & in_foot & (labels < 0)
        labels[member] = pid
    return labels
