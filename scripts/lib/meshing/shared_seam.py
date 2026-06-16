"""
Shared-seam BC cap assembly (critique proposal a).

Intersect the free iso-skin with each analytic BC plane, build planar caps on
that seam polyline, and merge caps + skin with coincident boundary vertices.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh

from lib.converters.bc_patch import Patch
from lib.converters.bc_planar_cap import triangulate_cap_on_plane


@dataclass
class SharedSeamAudit:
    n_patches: int = 0
    n_caps_built: int = 0
    n_seam_segments: int = 0
    n_seam_loops: int = 0
    n_fallback_caps: int = 0
    seam_verts_merged: int = 0
    meta: dict[str, float | int | str] = field(default_factory=dict)


def strip_labeled_bc_faces(
    mesh: trimesh.Trimesh,
    tri_labels: np.ndarray,
) -> trimesh.Trimesh:
    """Keep only free-skin triangles (``tri_labels < 0``)."""
    labels = np.asarray(tri_labels, dtype=np.int32)
    keep = labels < 0
    if np.all(keep):
        return mesh
    return trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices, dtype=float),
        faces=mesh.faces[keep],
        process=False,
    )


def _inplane_axes(axis: int) -> tuple[int, int]:
    return tuple(i for i in range(3) if i != axis)  # type: ignore[return-value]


def _footprint_contains(
    patch: Patch,
    origin: np.ndarray,
    spacing: np.ndarray,
    shape: np.ndarray,
    point: np.ndarray,
) -> bool:
    if not patch.faces:
        return False
    axis = patch.axis
    inplane = _inplane_axes(axis)
    cells = np.floor((np.asarray(point, dtype=float) - origin) / spacing).astype(int)
    cells = np.clip(cells, 0, np.asarray(shape, dtype=int) - 1)
    foot = tuple(int(cells[j]) for j in inplane)
    return foot in {tuple(f) for f in patch.faces}


def _quantize(point: np.ndarray, tol: float) -> tuple[int, int, int]:
    return tuple(int(round(float(c) / tol)) for c in point)


def _edge_plane_point(
    p0: np.ndarray,
    p1: np.ndarray,
    axis: int,
    plane: float,
    tol: float,
) -> np.ndarray | None:
    d0 = float(p0[axis] - plane)
    d1 = float(p1[axis] - plane)
    if abs(d0) <= tol and abs(d1) <= tol:
        return None
    if d0 * d1 > tol * tol:
        return None
    denom = d0 - d1
    t = 0.5 if abs(denom) < 1e-15 else d0 / denom
    t = float(np.clip(t, 0.0, 1.0))
    hit = p0 + t * (p1 - p0)
    hit[axis] = plane
    return hit


def extract_plane_seam_segments(
    mesh: trimesh.Trimesh,
    patch: Patch,
    origin: np.ndarray,
    spacing: np.ndarray,
    shape: np.ndarray,
    *,
    plane_tol: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Line segments where free-skin triangles cross ``patch`` plane inside footprint."""
    if not patch.is_bc or not patch.faces:
        return []

    axis = patch.axis
    plane = float(patch.plane_coord)
    verts = np.asarray(mesh.vertices, dtype=float)
    segments: list[tuple[np.ndarray, np.ndarray]] = []

    for face in mesh.faces:
        tri = verts[face]
        hits: list[np.ndarray] = []
        for i in range(3):
            p = _edge_plane_point(tri[i], tri[(i + 1) % 3], axis, plane, plane_tol)
            if p is None:
                continue
            if not _footprint_contains(patch, origin, spacing, shape, p):
                continue
            hits.append(p)
        if len(hits) < 2:
            continue
        a, b = hits[0], hits[1]
        if np.linalg.norm(a - b) <= plane_tol:
            continue
        segments.append((a.copy(), b.copy()))
    return segments


def _chain_segments_to_loop(
    segments: list[tuple[np.ndarray, np.ndarray]],
    *,
    merge_tol: float,
) -> np.ndarray | None:
    """Order segment endpoints into a single closed 3D polyline."""
    if not segments:
        return None

    nodes: dict[tuple[int, int, int], np.ndarray] = {}
    adj: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}

    def node_id(p: np.ndarray) -> tuple[int, int, int]:
        key = _quantize(p, merge_tol)
        if key not in nodes:
            nodes[key] = p.copy()
            adj[key] = []
        return key

    for a, b in segments:
        ia, ib = node_id(a), node_id(b)
        if ia == ib:
            continue
        adj[ia].append(ib)
        adj[ib].append(ia)

    if not adj:
        return None

    # Prefer a degree-2 chain (simple loop); fall back to largest connected walk.
    start = max(adj, key=lambda k: len(adj[k]))
    loop_keys: list[tuple[int, int, int]] = [start]
    prev: tuple[int, int, int] | None = None
    cur = start
    for _ in range(len(nodes) + 2):
        nbrs = [n for n in adj[cur] if n != prev]
        if not nbrs:
            break
        nxt = nbrs[0]
        if nxt == start and len(loop_keys) >= 3:
            break
        loop_keys.append(nxt)
        prev, cur = cur, nxt

    if len(loop_keys) < 3:
        return None
    return np.vstack([nodes[k] for k in loop_keys])


def _loop_to_boundary_xy(loop3d: np.ndarray, patch: Patch) -> np.ndarray:
    axis = patch.axis
    a0, a1 = _inplane_axes(axis)
    return np.column_stack([loop3d[:, a0], loop3d[:, a1]])


def _fallback_cap_boundary(
    patch: Patch,
    origin: np.ndarray,
    spacing: np.ndarray,
    shape: np.ndarray,
) -> np.ndarray | None:
    """Footprint upsample_blur contour when plane-cut seam is empty."""
    from lib.converters.bc_planar_cap import PlanarCapParams, PlanarCapRequest, build_planar_cap

    req = PlanarCapRequest(
        patch=patch,
        origin=origin,
        spacing=spacing,
        shape=shape,
        params=PlanarCapParams(),
    )
    result = build_planar_cap(req, "upsample_blur")
    if result.boundary_xy.size == 0:
        return None
    return np.asarray(result.boundary_xy, dtype=float)


def assemble_shared_seam_mesh(
    free_mesh: trimesh.Trimesh,
    patches: list[Patch],
    origin,
    spacing,
    shape,
    *,
    plane_tol: float | None = None,
) -> tuple[trimesh.Trimesh, np.ndarray, SharedSeamAudit]:
    """
    Build BC caps on plane-cut seam polylines and merge with ``free_mesh``.

    Returns ``(mesh, tri_labels, audit)``.
    """
    origin = np.asarray(origin, dtype=float)
    spacing = np.asarray(spacing, dtype=float)
    shape = np.asarray(shape, dtype=int)
    if plane_tol is None:
        plane_tol = float(0.25 * np.min(spacing))
    merge_tol = max(plane_tol * 0.5, float(np.min(spacing)) * 1e-4)

    cap_vert_parts: list[np.ndarray] = []
    cap_face_parts: list[np.ndarray] = []
    cap_label_parts: list[np.ndarray] = []
    audit = SharedSeamAudit(n_patches=len(patches))

    for pid, patch in enumerate(patches):
        if not patch.is_bc or not patch.faces:
            continue
        segments = extract_plane_seam_segments(
            free_mesh,
            patch,
            origin,
            spacing,
            shape,
            plane_tol=plane_tol,
        )
        audit.n_seam_segments += len(segments)
        loop = _chain_segments_to_loop(segments, merge_tol=merge_tol)
        boundary_xy: np.ndarray | None = None
        if loop is not None:
            audit.n_seam_loops += 1
            boundary_xy = _loop_to_boundary_xy(loop, patch)
        else:
            boundary_xy = _fallback_cap_boundary(patch, origin, spacing, shape)
            if boundary_xy is not None:
                audit.n_fallback_caps += 1
        if boundary_xy is None or len(boundary_xy) < 3:
            audit.meta[f"patch_{pid}_skipped"] = 1
            continue

        cap_verts, cap_faces = triangulate_cap_on_plane(
            boundary_xy,
            patch,
            origin,
            spacing,
        )
        if cap_faces.size == 0:
            audit.meta[f"patch_{pid}_empty_cap"] = 1
            continue
        cap_vert_parts.append(cap_verts)
        cap_face_parts.append(cap_faces)
        cap_label_parts.append(np.full(len(cap_faces), pid, dtype=np.int32))
        audit.n_caps_built += 1

    free_verts = np.asarray(free_mesh.vertices, dtype=float)
    free_faces = np.asarray(free_mesh.faces, dtype=np.int64)
    labels = np.full(len(free_faces), -1, dtype=np.int32)

    if not cap_vert_parts:
        return free_mesh.copy(), labels, audit

    parts_verts = [free_verts] + cap_vert_parts
    parts_faces: list[np.ndarray] = [free_faces]
    parts_labels: list[np.ndarray] = [labels]
    voff = len(free_verts)
    for cap_v, cap_f, cap_l in zip(cap_vert_parts, cap_face_parts, cap_label_parts):
        parts_faces.append(cap_f + voff)
        parts_labels.append(cap_l)
        voff += len(cap_v)

    verts = np.vstack(parts_verts)
    faces = np.vstack(parts_faces)
    out_labels = np.concatenate(parts_labels)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    n_verts_before = len(mesh.vertices)
    mesh.merge_vertices(merge_tex=True, digits_vertex=6)
    audit.seam_verts_merged = n_verts_before - len(mesh.vertices)

    trimesh.repair.fix_normals(mesh)
    audit.meta["watertight"] = int(bool(mesh.is_watertight))
    audit.meta["bodies"] = int(mesh.body_count) if hasattr(mesh, "body_count") else -1
    return mesh, out_labels, audit
