"""
Re-derive BC patches on the smoothed mesh by geodesic coplanar region-grow.

Load patches are stamped separately (see bc_patch.stamp_load_patches); anchors
are projected onto the smoothed iso-surface before face stamping. Only BC face
patches use planar regrow here.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import trimesh

from lib.converters.bc_patch import (
    Patch,
    is_thin_patch,
    patch_inplane_extent,
    patch_seed_point,
)

if TYPE_CHECKING:
    LogFn = Callable[[str], None]


def _face_neighbors(mesh: trimesh.Trimesh) -> list[list[int]]:
    """Face index → edge-adjacent face indices."""
    nbr: list[list[int]] = [[] for _ in range(len(mesh.faces))]
    for a, b in mesh.face_adjacency:
        nbr[a].append(b)
        nbr[b].append(a)
    return nbr


def _face_plane_deltas(
    mesh: trimesh.Trimesh,
    axis: int,
    plane_coord: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-face planarity relative to a domain plane.

    Returns (max_vertex_delta, min_vertex_delta) along the patch axis.
    """
    tri_verts = mesh.vertices[mesh.faces][:, :, axis]
    delta = np.abs(tri_verts - plane_coord)
    return delta.max(axis=1), delta.min(axis=1)


def _face_is_planar(
    mesh: trimesh.Trimesh,
    face_id: int,
    *,
    axis: int,
    plane_coord: float,
    plane_normal,
    band: float,
    cos_tol: float,
    max_delta: np.ndarray | None = None,
    min_delta: np.ndarray | None = None,
) -> bool:
    """True when every vertex of the face lies in the plane band and normal aligns."""
    n = np.asarray(plane_normal, dtype=float)
    if max_delta is None or min_delta is None:
        max_delta, min_delta = _face_plane_deltas(mesh, axis, plane_coord)
    return (
        max_delta[face_id] < band
        and min_delta[face_id] < band
        and mesh.face_normals[face_id] @ n > cos_tol
    )


def seed_face(
    mesh: trimesh.Trimesh,
    point,
    plane_normal,
    *,
    axis: int,
    plane_coord: float,
    band: float,
    cos_tol: float = 0.92,
    cos_tol_grow: float | None = None,
    band_grow: float | None = None,
    max_delta: np.ndarray | None = None,
    min_delta: np.ndarray | None = None,
) -> int:
    """Nearest planar face on the patch plane to a BC anchor point."""
    n = np.asarray(plane_normal, dtype=float)
    point = np.asarray(point, dtype=float)
    cos_tol_grow = cos_tol if cos_tol_grow is None else cos_tol_grow
    band_grow = band if band_grow is None else band_grow

    if max_delta is None or min_delta is None:
        max_delta, min_delta = _face_plane_deltas(mesh, axis, plane_coord)
    alignment = mesh.face_normals @ n
    centroids = mesh.triangles_center

    def _pick(candidates: np.ndarray) -> int:
        d = np.linalg.norm(centroids[candidates] - point, axis=1)
        return int(candidates[np.argmin(d)])

    strict = np.where(
        (max_delta < band) & (min_delta < band) & (alignment > cos_tol)
    )[0]
    if strict.size:
        return _pick(strict)

    loose = np.where(
        (max_delta < band_grow) & (min_delta < band_grow) & (alignment > cos_tol_grow)
    )[0]
    if loose.size:
        return _pick(loose)

    raise ValueError(
        "no plane-aligned face near the BC point "
        "(contact face may have been smoothed away)"
    )


def seed_face_from_labels(
    mesh: trimesh.Trimesh,
    tri_labels: np.ndarray,
    patch_id: int,
    point,
    plane_normal,
    *,
    axis: int,
    plane_coord: float,
    band: float,
    cos_tol: float,
    cos_tol_grow: float,
    band_grow: float,
    max_delta: np.ndarray | None = None,
    min_delta: np.ndarray | None = None,
) -> int | None:
    """Fallback seed from the pre-smooth voxel footprint on this patch."""
    on_patch = np.asarray(tri_labels, dtype=int) == patch_id
    if not np.any(on_patch):
        return None
    n = np.asarray(plane_normal, dtype=float)
    point = np.asarray(point, dtype=float)
    if max_delta is None or min_delta is None:
        max_delta, min_delta = _face_plane_deltas(mesh, axis, plane_coord)
    alignment = mesh.face_normals @ n
    centroids = mesh.triangles_center

    def _nearest(candidates: np.ndarray) -> int:
        d = np.linalg.norm(centroids[candidates] - point, axis=1)
        return int(candidates[np.argmin(d)])

    strict = np.where(
        on_patch & (max_delta < band) & (min_delta < band) & (alignment > cos_tol)
    )[0]
    if strict.size:
        return _nearest(strict)

    loose = np.where(
        on_patch
        & (max_delta < band_grow)
        & (min_delta < band_grow)
        & (alignment > cos_tol_grow)
    )[0]
    if loose.size:
        return _nearest(loose)

    # Post-Taubin vertices may leave the plane band; keep the voxel footprint.
    labeled = np.where(on_patch)[0]
    if labeled.size:
        return _nearest(labeled)
    return None


def seed_face_near_domain_slab(
    mesh: trimesh.Trimesh,
    point,
    plane_normal,
    *,
    axis: int,
    plane_coord: float,
    side: str,
    spacing: float,
    band: float,
    cos_tol: float = 0.5,
) -> int | None:
    """Nearest outward face on the domain slab (used when voxel transfer missed max faces)."""
    point = np.asarray(point, dtype=float)
    n = np.asarray(plane_normal, dtype=float)
    ref = plane_coord if side == "min" else plane_coord - spacing
    centroids = mesh.triangles_center
    alignment = mesh.face_normals @ n
    candidates = np.where(
        (np.abs(centroids[:, axis] - ref) < band) & (alignment > cos_tol)
    )[0]
    if not candidates.size:
        return None
    d = np.linalg.norm(centroids[candidates] - point, axis=1)
    return int(candidates[np.argmin(d)])


def pick_patch_seed(
    mesh: trimesh.Trimesh,
    *,
    patch: Patch,
    patch_id: int,
    seed_pt: np.ndarray,
    tri_labels_voxel: np.ndarray | None,
    band: float,
    band_grow: float,
    cos_tol: float,
    cos_tol_grow: float,
    seed_cos_tol: float,
    spacing: np.ndarray,
    max_delta: np.ndarray | None = None,
    min_delta: np.ndarray | None = None,
) -> tuple[int, bool]:
    """
    Choose a regrow seed and whether it satisfies the loose plane criterion.

    Prefer the pre-smooth voxel footprint (stable after Taubin); fall back to the
    NITO anchor when no labeled triangles exist.
    """
    common = dict(
        axis=patch.axis,
        plane_coord=patch.plane_coord,
        band=band,
        cos_tol=cos_tol,
        cos_tol_grow=cos_tol_grow,
        band_grow=band_grow,
        max_delta=max_delta,
        min_delta=min_delta,
    )

    seed: int | None = None
    if tri_labels_voxel is not None:
        seed = seed_face_from_labels(
            mesh,
            tri_labels_voxel,
            patch_id,
            seed_pt,
            patch.normal,
            **common,
        )
    if seed is None:
        try:
            seed = seed_face(
                mesh,
                seed_pt,
                patch.normal,
                **{**common, "cos_tol": seed_cos_tol},
            )
        except ValueError:
            seed = None
    if seed is None:
        seed = seed_face_near_domain_slab(
            mesh,
            seed_pt,
            patch.normal,
            axis=patch.axis,
            plane_coord=patch.plane_coord,
            side=patch.side,
            spacing=float(spacing[patch.axis]),
            band=band_grow,
            cos_tol=cos_tol_grow,
        )
    if seed is None:
        raise ValueError(
            f"no regrow seed for BC patch {patch_id} "
            f"(axis={patch.axis} side={patch.side}; "
            "contact face may have been smoothed away)"
        )

    seed_loose_ok = _face_on_plane(
        mesh,
        seed,
        axis=patch.axis,
        plane_coord=patch.plane_coord,
        plane_normal=patch.normal,
        band=band_grow,
        cos_tol=cos_tol_grow,
        max_delta=max_delta,
        min_delta=min_delta,
    )
    return seed, seed_loose_ok


def _face_on_plane(
    mesh: trimesh.Trimesh,
    face_id: int,
    *,
    axis: int,
    plane_coord: float,
    plane_normal,
    band: float,
    cos_tol: float,
    max_delta: np.ndarray | None = None,
    min_delta: np.ndarray | None = None,
) -> bool:
    return _face_is_planar(
        mesh,
        face_id,
        axis=axis,
        plane_coord=plane_coord,
        plane_normal=plane_normal,
        band=band,
        cos_tol=cos_tol,
        max_delta=max_delta,
        min_delta=min_delta,
    )


def regrow_patch_on_mesh(
    mesh: trimesh.Trimesh,
    seed: int,
    axis: int,
    plane_coord: float,
    plane_normal,
    band: float,
    cos_tol: float,
    *,
    band_grow: float | None = None,
    cos_grow: float | None = None,
    close_iters: int = 2,
    max_delta: np.ndarray | None = None,
    min_delta: np.ndarray | None = None,
    allow_nonplanar_seed: bool = False,
) -> np.ndarray:
    """
    Geodesic coplanar flood-fill from `seed`. Returns a boolean face mask.

    A face is admitted only when *all* vertices lie within the plane band and
    the face normal aligns — stricter than centroid tests for jagged MC meshes.
    """
    n = np.asarray(plane_normal, dtype=float)
    band_grow = band if band_grow is None else band_grow
    cos_grow = cos_tol if cos_grow is None else cos_grow

    if max_delta is None or min_delta is None:
        max_delta, min_delta = _face_plane_deltas(mesh, axis, plane_coord)
    alignment = mesh.face_normals @ n
    nbr = _face_neighbors(mesh)

    def ok(face_id: int, strict: bool) -> bool:
        b, c = (band, cos_tol) if strict else (band_grow, cos_grow)
        return (
            max_delta[face_id] < b
            and min_delta[face_id] < b
            and alignment[face_id] > c
        )

    if not ok(seed, strict=True):
        if not ok(seed, strict=False) and not allow_nonplanar_seed:
            raise ValueError("seed face fails the patch plane criterion; check the plane")

    in_patch = np.zeros(len(mesh.faces), dtype=bool)
    in_patch[seed] = True
    q: deque[int] = deque([seed])
    while q:
        face_id = q.popleft()
        for nb in nbr[face_id]:
            if not in_patch[nb] and ok(nb, strict=False):
                in_patch[nb] = True
                q.append(nb)

    for _ in range(close_iters):
        holes = [
            g
            for g in range(len(mesh.faces))
            if not in_patch[g]
            and nbr[g]
            and all(in_patch[h] for h in nbr[g])
        ]
        if not holes:
            break
        in_patch[holes] = True

    return in_patch


def patch_boundary_length(mesh: trimesh.Trimesh, mask: np.ndarray) -> int:
    """Patch/non-patch adjacent face pairs — a jaggedness proxy."""
    adj = mesh.face_adjacency
    return int(np.count_nonzero(mask[adj[:, 0]] != mask[adj[:, 1]]))


def regrow_patches_on_mesh(
    mesh: trimesh.Trimesh,
    patches: list[Patch],
    bc_rows: np.ndarray,
    shape: np.ndarray,
    spacing: np.ndarray,
    *,
    origin=None,
    tri_labels_voxel: np.ndarray | None = None,
    band_cells: float = 0.5,
    cos_tol: float = 0.92,
    cos_tol_grow: float = 0.85,
    band_grow_factor: float = 1.25,
    seed_cos_tol: float = 0.92,
    close_iters: int = 2,
    thin_min_extent: int = 2,
    log: LogFn | None = None,
) -> np.ndarray:
    """
    Regrow BC face patches on a Taubin-smoothed mesh; return partial tri_labels.

    Only patches with ``kind == "bc"`` are regrown. Load patches are stamped
    separately via ``stamp_load_patches``.
    """
    spacing = np.asarray(spacing, dtype=float)
    labels = np.full(len(mesh.faces), -1, dtype=np.int32)
    bc_patches = [p for p in patches if p.is_bc]

    for pid, patch in enumerate(bc_patches):
        if is_thin_patch(patch, min_extent=thin_min_extent):
            e0, e1 = patch_inplane_extent(patch.faces)
            if log is not None:
                log(
                    f"        patch {pid}: thin ({e0}×{e1} cells), "
                    "regrow skipped — voxel footprint kept"
                )
            if tri_labels_voxel is not None:
                mask = np.asarray(tri_labels_voxel, dtype=int) == pid
                labels[mask & (labels < 0)] = pid
            continue

        seed_pt = patch_seed_point(
            bc_rows,
            shape,
            patch.axis,
            patch.side,
            origin=origin,
            spacing=spacing,
        )
        if seed_pt is None:
            raise ValueError(
                f"no NITO BC seed for patch axis={patch.axis} side={patch.side}"
            )
        band = float(band_cells * spacing[patch.axis])
        band_grow = band * band_grow_factor
        max_delta, min_delta = _face_plane_deltas(mesh, patch.axis, patch.plane_coord)
        seed, seed_loose_ok = pick_patch_seed(
            mesh,
            patch=patch,
            patch_id=pid,
            seed_pt=seed_pt,
            tri_labels_voxel=tri_labels_voxel,
            band=band,
            band_grow=band_grow,
            cos_tol=cos_tol,
            cos_tol_grow=cos_tol_grow,
            seed_cos_tol=seed_cos_tol,
            spacing=spacing,
            max_delta=max_delta,
            min_delta=min_delta,
        )
        mask = regrow_patch_on_mesh(
            mesh,
            seed,
            patch.axis,
            patch.plane_coord,
            patch.normal,
            band,
            cos_tol,
            band_grow=band_grow,
            cos_grow=cos_tol_grow,
            close_iters=close_iters,
            max_delta=max_delta,
            min_delta=min_delta,
            allow_nonplanar_seed=not seed_loose_ok,
        )
        labels[mask & (labels < 0)] = pid

    return labels
