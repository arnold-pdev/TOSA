"""
Boundary-condition patches on voxel grids for voxel2surf.

Maps NITO boundary_conditions rows (same index as topologies[i]) onto
axis-aligned exterior faces of the solid, then builds Patch footprints
for exact cell-membership transfer onto the extracted surface mesh.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace

import numpy as np
import trimesh
from trimesh.triangles import closest_point as tri_closest_points

from lib.nito_physics import constraint_flags, force_components, physical_points


@dataclass(frozen=True)
class BCSpec:
    """One mounting face on the design-domain boundary."""

    axis: int
    side: str  # "min" | "max"
    flags: np.ndarray

    @classmethod
    def from_face(cls, axis: int, side: str, flags: np.ndarray) -> BCSpec:
        return cls(axis=int(axis), side=side, flags=np.asarray(flags, dtype=float))


@dataclass(frozen=True)
class Patch:
    """Solid cells on an exterior domain face, or a point-load anchor on the surface."""

    axis: int
    side: str
    plane_coord: float
    faces: list[tuple[int, ...]]
    normal: np.ndarray
    flags: np.ndarray
    kind: str = "bc"  # "bc" | "load"
    anchor: np.ndarray | None = None
    force: np.ndarray | None = None

    @property
    def is_bc(self) -> bool:
        return self.kind == "bc"

    @property
    def is_load(self) -> bool:
        return self.kind == "load"


def patch_inplane_extent(faces: list[tuple[int, ...]]) -> tuple[int, int]:
    """In-plane voxel extents (cells) of a patch footprint; (0, 0) if empty."""
    if not faces:
        return 0, 0
    arr = np.asarray(faces, dtype=int)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    extents = arr.max(axis=0) - arr.min(axis=0) + 1
    if extents.size == 1:
        return int(extents[0]), 1
    return int(extents[0]), int(extents[1])


def is_thin_patch(patch: Patch, *, min_extent: int = 2) -> bool:
    """True when the footprint is narrower than min_extent along any in-plane axis."""
    e0, e1 = patch_inplane_extent(patch.faces)
    return e0 < min_extent or e1 < min_extent


def _ndim_from_shape(shape: np.ndarray) -> int:
    return int(np.asarray(shape, dtype=int).size)


def _boundary_face(norm_pos: np.ndarray, ndim: int, *, tol: float = 0.08) -> tuple[int, str] | None:
    """Pick the domain face nearest a normalized BC position."""
    pos = np.asarray(norm_pos, dtype=float).ravel()[:ndim]
    best: tuple[int, str] | None = None
    best_dist = np.inf
    for a in range(ndim):
        for side, dist in (("min", pos[a]), ("max", 1.0 - pos[a])):
            if dist < best_dist:
                best_dist = dist
                best = (a, side)
    if best is None or best_dist > tol:
        return None
    return best


def bcspecs_from_nito(bc: np.ndarray, shape: np.ndarray) -> list[BCSpec]:
    """
    Build one BCSpec per constrained domain face from NITO BC rows.

    Rows with all-zero constraint flags are skipped. Multiple rows on the
    same face are merged (flags OR-ed).
    """
    ndim = _ndim_from_shape(shape)
    rows = np.atleast_2d(np.asarray(bc, dtype=float))
    merged: dict[tuple[int, str], np.ndarray] = {}

    for row in rows:
        flags = constraint_flags(row, shape)
        if not np.any(flags > 0.5):
            continue
        face = _boundary_face(row[:ndim], ndim)
        if face is None:
            continue
        key = face
        prev = merged.get(key)
        merged[key] = flags if prev is None else np.maximum(prev, flags)

    specs = [
        BCSpec.from_face(axis, side, flags)
        for (axis, side), flags in sorted(merged.items())
    ]
    return specs


def _exposed_faces(vox: np.ndarray) -> set[tuple]:
    """
    Exterior voxel faces on the solid: (axis, side, *inplane_cell_indices).

    A face is exposed when the neighbor across it is void or outside the grid.
    """
    solid = np.asarray(vox) > 0
    shape = solid.shape
    ndim = len(shape)
    exposed: set[tuple] = set()

    for cell in zip(*np.nonzero(solid)):
        for a in range(ndim):
            for side, delta in (("min", -1), ("max", 1)):
                nb = list(cell)
                nb[a] += delta
                oob = nb[a] < 0 or nb[a] >= shape[a]
                void = (not oob) and (not solid[tuple(nb)])
                if not (oob or void):
                    continue
                inplane = tuple(cell[i] for i in range(ndim) if i != a)
                exposed.add((a, side, *inplane))
    return exposed


def _slab_index(shape: np.ndarray, axis: int, side: str) -> int:
    return 0 if side == "min" else int(shape[axis]) - 1


def _plane_coord(origin: np.ndarray, spacing: np.ndarray, shape: np.ndarray,
                 axis: int, side: str) -> float:
    if side == "min":
        return float(origin[axis])
    return float(origin[axis] + shape[axis] * spacing[axis])


def _outward_normal(ndim: int, axis: int, side: str) -> np.ndarray:
    n = np.zeros(max(ndim, 3), dtype=float)
    n[axis] = -1.0 if side == "min" else 1.0
    return n


def _connected_slab_faces(
    solid: np.ndarray,
    *,
    axis: int,
    side: str,
    exposed: set[tuple],
    seeds: list[tuple[int, ...]] | None = None,
) -> list[tuple[int, ...]]:
    """Solid cells on a domain-face slab, 4/6-connected through in-plane neighbors."""
    shape = solid.shape
    ndim = len(shape)
    slab = _slab_index(shape, axis, side)
    inplane_axes = [i for i in range(ndim) if i != axis]

    def key(cell: tuple[int, ...]) -> tuple:
        return (axis, side, *tuple(cell[i] for i in inplane_axes))

    candidates: set[tuple[int, ...]] = set()
    for idx in np.argwhere(solid):
        cell = tuple(int(x) for x in idx)
        if cell[axis] != slab:
            continue
        if key(cell) not in exposed:
            continue
        candidates.add(tuple(cell[i] for i in inplane_axes))

    if not candidates:
        return []

    if seeds:
        start = [s for s in seeds if s in candidates]
        if not start:
            start = [next(iter(candidates))]
    else:
        start = [next(iter(candidates))]

    visited: set[tuple[int, ...]] = set()
    q = deque(start)
    while q:
        cur = q.popleft()
        if cur in visited or cur not in candidates:
            continue
        visited.add(cur)
        # in-plane neighbors on the slab
        full = [0] * ndim
        full[axis] = slab
        for k, ax in enumerate(inplane_axes):
            full[ax] = cur[k]
        for off in _inplane_offsets(ndim - 1):
            nb = tuple(cur[i] + off[i] for i in range(len(cur)))
            if nb in candidates and nb not in visited:
                q.append(nb)

    return sorted(visited)


def _inplane_offsets(ndim_inplane: int) -> list[tuple[int, ...]]:
    if ndim_inplane == 1:
        return [(-1,), (1,)]
    if ndim_inplane == 2:
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    raise ValueError(f"Unsupported in-plane dimension {ndim_inplane}")


def _anchor_inplane(norm_pos: np.ndarray, shape: np.ndarray, axis: int) -> tuple[int, ...]:
    ndim = _ndim_from_shape(shape)
    pos = np.asarray(norm_pos, dtype=float).ravel()[:ndim]
    shape = np.asarray(shape, dtype=int)
    out = []
    for i in range(ndim):
        if i == axis:
            continue
        out.append(int(np.clip(np.rint(pos[i] * shape[i]), 0, shape[i] - 1)))
    return tuple(out)


def patch_seed_point(
    bc_rows: np.ndarray,
    shape: np.ndarray,
    axis: int,
    side: str,
    *,
    origin=None,
    spacing=None,
) -> np.ndarray | None:
    """Physical anchor on a BC patch plane from the first matching NITO row."""
    nd = _ndim_from_shape(shape)
    rows = np.atleast_2d(np.asarray(bc_rows, dtype=float))
    origin = np.zeros(3, dtype=float) if origin is None else np.asarray(origin, dtype=float)
    spacing = np.ones(3, dtype=float) if spacing is None else np.asarray(spacing, dtype=float)
    shape_arr = np.asarray(shape, dtype=int)

    for row in rows:
        flags = constraint_flags(row, shape)
        if not np.any(flags > 0.5):
            continue
        face = _boundary_face(row[:nd], nd)
        if face != (axis, side):
            continue
        pos = np.asarray(row[:nd], dtype=float).ravel()
        pt = np.zeros(3, dtype=float)
        for i in range(nd):
            if i == axis:
                pt[i] = _plane_coord(origin, spacing, shape_arr, axis, side)
            else:
                pt[i] = float(origin[i] + pos[i] * shape_arr[i] * spacing[i])
        return pt
    return None


def build_patch(
    vox: np.ndarray,
    spec: BCSpec,
    origin,
    spacing,
    *,
    exposed: set[tuple] | None = None,
    bc_rows: np.ndarray | None = None,
    shape: np.ndarray | None = None,
) -> Patch:
    """
    Collect exposed solid cells on the domain face described by `spec`.

    When bc_rows are supplied, only the connected component touching those
    anchor cells is kept; otherwise the full face component is used.
    """
    origin = np.asarray(origin, dtype=float)
    spacing = np.asarray(spacing, dtype=float)
    solid = np.asarray(vox) > 0
    shape_arr = np.asarray(solid.shape if shape is None else shape, dtype=int)
    ndim = len(solid.shape)
    if exposed is None:
        exposed = _exposed_faces(solid)

    seeds: list[tuple[int, ...]] | None = None
    if bc_rows is not None and shape is not None:
        nd = _ndim_from_shape(shape)
        rows = np.atleast_2d(np.asarray(bc_rows, dtype=float))
        seeds = []
        for row in rows:
            flags = constraint_flags(row, shape)
            if not np.any(flags > 0.5):
                continue
            face = _boundary_face(row[:nd], nd)
            if face != (spec.axis, spec.side):
                continue
            seeds.append(_anchor_inplane(row[:nd], shape, spec.axis))

    faces = _connected_slab_faces(
        solid,
        axis=spec.axis,
        side=spec.side,
        exposed=exposed,
        seeds=seeds,
    )

    return Patch(
        axis=spec.axis,
        side=spec.side,
        plane_coord=_plane_coord(origin, spacing, shape_arr, spec.axis, spec.side),
        faces=faces,
        normal=_outward_normal(ndim, spec.axis, spec.side),
        flags=np.asarray(spec.flags, dtype=float),
    )


def build_patches_from_nito(
    vox: np.ndarray,
    bc: np.ndarray,
    shape: np.ndarray,
    origin,
    spacing,
) -> list[Patch]:
    """Convenience: NITO BC bundle → patches for voxel_to_surface."""
    exposed = _exposed_faces(vox)
    specs = bcspecs_from_nito(bc, shape)
    return [
        build_patch(
            vox,
            spec,
            origin,
            spacing,
            exposed=exposed,
            bc_rows=bc,
            shape=shape,
        )
        for spec in specs
    ]


def build_load_patches(load_rows: np.ndarray, shape: np.ndarray) -> list[Patch]:
    """One point-load patch per NITO load row (vertex stamp, not planar regrow)."""
    rows = np.atleast_2d(np.asarray(load_rows, dtype=float))
    if rows.size == 0:
        return []
    patches: list[Patch] = []
    for row in rows:
        force = force_components(row, shape)
        if not np.any(np.abs(force) > 1e-12):
            continue
        pos = physical_points(row.reshape(1, -1), shape)[0]
        anchor = np.zeros(3, dtype=float)
        anchor[: len(pos)] = pos
        patches.append(
            Patch(
                axis=-1,
                side="load",
                plane_coord=0.0,
                faces=[],
                normal=np.zeros(3, dtype=float),
                flags=np.zeros(3, dtype=float),
                kind="load",
                anchor=anchor,
                force=np.asarray(force, dtype=float),
            )
        )
    return patches


def project_point_to_mesh(
    mesh: trimesh.Trimesh,
    point: np.ndarray,
    *,
    face_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, float, int]:
    """
    Closest point on the mesh surface to ``point``.

    NITO load anchors can lie inside the solid or on the design-domain boundary
    (outside the iso-surface). Marching-cubes padding only shifts ``origin``;
    anchors still need projection onto the extracted surface before stamping.

    Optional ``face_mask`` restricts candidates (e.g. free-boundary triangles).
    """
    point = np.asarray(point, dtype=float).ravel()[:3]
    tris = np.asarray(mesh.triangles, dtype=float)
    if tris.size == 0:
        return point.copy(), 0.0, -1

    candidates = np.arange(len(tris), dtype=int)
    if face_mask is not None:
        masked = np.flatnonzero(np.asarray(face_mask, dtype=bool))
        if masked.size:
            candidates = masked

    query = np.broadcast_to(point, (len(candidates), 3)).copy()
    closest = tri_closest_points(tris[candidates], query)
    dists = np.linalg.norm(closest - point, axis=1)
    local = int(np.argmin(dists))
    return (
        np.asarray(closest[local], dtype=float),
        float(dists[local]),
        int(candidates[local]),
    )


def stamp_load_patches(
    mesh,
    load_patches: list[Patch],
    labels: np.ndarray,
    start_pid: int,
    *,
    spacing,
    radius_cells: float = 1.0,
) -> tuple[list[Patch], list[float]]:
    """
    Label faces near each NITO load and snap anchors onto the surface mesh.

    Returns (updated load patches with projected anchors, snap distances from NITO).
    """
    if not load_patches:
        return [], []
    spacing = np.asarray(spacing, dtype=float)
    radius = float(radius_cells * float(np.max(spacing)))
    verts = np.asarray(mesh.vertices, dtype=float)
    updated: list[Patch] = []
    snap_dists: list[float] = []

    for i, patch in enumerate(load_patches):
        if patch.anchor is None:
            updated.append(patch)
            snap_dists.append(0.0)
            continue
        pid = start_pid + i
        anchor = np.asarray(patch.anchor, dtype=float)

        projected, snap_dist, _ = project_point_to_mesh(mesh, anchor)

        dist = np.linalg.norm(verts - projected, axis=1)
        vids = np.where(dist <= radius)[0]
        if vids.size == 0:
            vids = np.array([int(np.argmin(dist))])

        vset = {int(v) for v in vids}

        def _stamp_faces(*, allow_labeled: bool) -> int:
            n = 0
            for f_idx, face in enumerate(mesh.faces):
                if not allow_labeled and labels[f_idx] >= 0:
                    continue
                if any(int(v) in vset for v in face):
                    labels[f_idx] = pid
                    n += 1
            return n

        n_stamped = _stamp_faces(allow_labeled=False)
        if n_stamped == 0:
            _stamp_faces(allow_labeled=True)

        updated.append(replace(patch, anchor=projected))
        snap_dists.append(snap_dist)

    return updated, snap_dists
