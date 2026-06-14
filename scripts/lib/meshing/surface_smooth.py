"""Post-extract surface refinement: subdivide and constrained Laplacian smooth."""

from __future__ import annotations

from collections import deque

import numpy as np
import scipy.sparse as sp
import trimesh

from lib.converters.bc_patch import Patch


def _face_neighbors(mesh: trimesh.Trimesh) -> list[list[int]]:
    """Face index → edge-adjacent face indices."""
    nbr: list[list[int]] = [[] for _ in range(len(mesh.faces))]
    for a, b in mesh.face_adjacency:
        nbr[a].append(b)
        nbr[b].append(a)
    return nbr


def _interior_bc_faces(
    tri_labels: np.ndarray,
    face_neighbors: list[list[int]],
) -> np.ndarray:
    """BC triangles whose edge-neighbors all share the same patch id."""
    interior = np.zeros(len(tri_labels), dtype=bool)
    for f, pid in enumerate(tri_labels):
        if pid < 0:
            continue
        nbrs = face_neighbors[f]
        if nbrs and all(tri_labels[n] == pid for n in nbrs):
            interior[f] = True
    return interior


def _interior_bc_vertex_map(
    mesh: trimesh.Trimesh,
    tri_labels: np.ndarray,
    n_bc_patches: int,
) -> np.ndarray:
    """Vertices whose incident BC triangles are all interior (deep patch core)."""
    if n_bc_patches <= 0:
        return np.full(len(mesh.vertices), -1, dtype=int)
    interior = _interior_bc_faces(tri_labels, _face_neighbors(mesh))
    vmap = np.full(len(mesh.vertices), -1, dtype=int)
    vf = mesh.vertex_faces
    for v in range(len(mesh.vertices)):
        inc = vf[v][vf[v] >= 0]
        if inc.size == 0:
            continue
        bc_inc = inc[tri_labels[inc] >= 0]
        if bc_inc.size == 0 or not np.all(interior[bc_inc]):
            continue
        pid = int(tri_labels[bc_inc[0]])
        if 0 <= pid < n_bc_patches and np.all(tri_labels[bc_inc] == pid):
            vmap[v] = pid
    return vmap


def freeze_weights(
    mesh: trimesh.Trimesh,
    vmap: np.ndarray,
    transition_rings: int = 0,
) -> np.ndarray:
    """Per-vertex weight 1 = frozen, 0 = free; optional ramp into free boundary."""
    n = len(mesh.vertices)
    freeze = (vmap >= 0).astype(float)
    patch_ids = np.where(vmap >= 0)[0]
    if patch_ids.size == 0 or transition_rings <= 0:
        return freeze

    neighbors = mesh.vertex_neighbors
    dist = np.full(n, -1)
    dist[patch_ids] = 0
    q = deque(int(i) for i in patch_ids)
    while q:
        i = q.popleft()
        if dist[i] >= transition_rings:
            continue
        for j in neighbors[i]:
            if dist[j] < 0:
                dist[j] = dist[i] + 1
                q.append(j)
    ramp = (dist > 0) & (dist <= transition_rings)
    freeze[ramp] = 1.0 - dist[ramp] / (transition_rings + 1.0)
    return freeze


def bc_patch_vertex_map(
    mesh: trimesh.Trimesh,
    tri_labels: np.ndarray,
    n_bc_patches: int,
) -> np.ndarray:
    """
    Vertex → dominant BC patch id for any vertex incident to a BC triangle.

    Used to freeze and project the full BC footprint (core + rims), not only
    triangles whose neighbors all share the same patch.
    """
    if n_bc_patches <= 0:
        return np.full(len(mesh.vertices), -1, dtype=int)
    vmap = np.full(len(mesh.vertices), -1, dtype=int)
    vf = mesh.vertex_faces
    for v in range(len(mesh.vertices)):
        inc = vf[v][vf[v] >= 0]
        if inc.size == 0:
            continue
        bc_inc = tri_labels[inc]
        bc_inc = bc_inc[bc_inc >= 0]
        if bc_inc.size == 0:
            continue
        vals, counts = np.unique(bc_inc, return_counts=True)
        pid = int(vals[int(np.argmax(counts))])
        if 0 <= pid < n_bc_patches:
            vmap[v] = pid
    return vmap


def bc_patch_freeze_weights(
    mesh: trimesh.Trimesh,
    tri_labels: np.ndarray,
    n_bc_patches: int,
    *,
    transition_rings: int = 0,
) -> tuple[np.ndarray, int, int]:
    """
    Per-vertex smooth weights (1 = frozen) for all BC patch vertices.

    Returns ``(weights, n_bc_triangles, n_frozen_vertices)``.
    """
    vmap = bc_patch_vertex_map(mesh, tri_labels, n_bc_patches)
    weights = freeze_weights(mesh, vmap, transition_rings=transition_rings)
    n_frozen = int(np.count_nonzero(weights >= 1.0 - 1e-12))
    n_bc_tris = int(np.count_nonzero(np.asarray(tri_labels) >= 0))
    return weights, n_bc_tris, n_frozen


def bc_interior_freeze_weights(
    mesh: trimesh.Trimesh,
    tri_labels: np.ndarray,
    n_bc_patches: int,
    *,
    transition_rings: int = 0,
) -> tuple[np.ndarray, int, int]:
    """
    Per-vertex smooth weights (1 = frozen) from interior BC triangles.

    Returns ``(weights, n_interior_faces, n_frozen_vertices)``.
    """
    interior = _interior_bc_faces(tri_labels, _face_neighbors(mesh))
    vmap = _interior_bc_vertex_map(mesh, tri_labels, n_bc_patches)
    weights = freeze_weights(mesh, vmap, transition_rings=transition_rings)
    n_frozen = int(np.count_nonzero(weights >= 1.0 - 1e-12))
    return weights, int(np.count_nonzero(interior)), n_frozen


def _uniform_laplacian(mesh: trimesh.Trimesh) -> sp.csr_matrix:
    """Row-normalized adjacency: (L @ v) is each vertex's neighbor mean."""
    n = len(mesh.vertices)
    e = mesh.edges_unique
    rows = np.concatenate([e[:, 0], e[:, 1]])
    cols = np.concatenate([e[:, 1], e[:, 0]])
    adj = sp.coo_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n)).tocsr()
    deg = np.maximum(np.asarray(adj.sum(1)).ravel(), 1.0)
    return sp.diags(1.0 / deg) @ adj


def _project_bc_vertices(
    mesh: trimesh.Trimesh,
    vmap: np.ndarray,
    patches: list[Patch],
) -> None:
    """Snap BC patch vertices onto their patch plane (normal coordinate only)."""
    for pid, patch in enumerate(patches):
        if not patch.is_bc:
            continue
        vids = np.where(vmap == pid)[0]
        if vids.size:
            mesh.vertices[vids, patch.axis] = patch.plane_coord


def subdivide_mesh(mesh: trimesh.Trimesh, levels: int = 1) -> trimesh.Trimesh:
    """Linear subdivide via PyVista (``levels`` passes)."""
    if levels <= 0:
        return mesh
    import pyvista as pv

    faces = np.column_stack(
        [np.full(len(mesh.faces), 3, dtype=np.int64), mesh.faces.astype(np.int64)],
    ).ravel()
    poly = pv.PolyData(mesh.vertices, faces)
    for _ in range(levels):
        poly = poly.subdivide(1, subfilter="linear")
    new_faces = poly.faces.reshape(-1, 4)[:, 1:4]
    return trimesh.Trimesh(
        vertices=np.asarray(poly.points, dtype=float),
        faces=np.asarray(new_faces, dtype=np.int64),
        process=False,
    )


def constrained_laplacian_smooth(
    mesh: trimesh.Trimesh,
    weights: np.ndarray,
    patches: list[Patch],
    tri_labels: np.ndarray,
    *,
    iterations: int = 0,
    relaxation: float = 0.3,
    constrain_bc_planes: bool = True,
) -> tuple[int, int]:
    """
    Laplacian smooth with frozen vertices; optional BC plane projection each iter.

    BC vertices (any triangle with ``tri_labels >= 0``) are frozen during the
    Laplacian step and projected onto their patch plane after every iteration
    when ``constrain_bc_planes`` is True.

    Returns ``(n_bc_triangles, n_frozen_vertices)`` when BC patches exist.
    """
    if iterations <= 0:
        return 0, 0

    n_bc = sum(1 for p in patches if p.is_bc)
    vmap = bc_patch_vertex_map(mesh, tri_labels, n_bc) if n_bc else np.full(
        len(mesh.vertices), -1, dtype=int,
    )
    n_bc_tris = int(np.count_nonzero(np.asarray(tri_labels) >= 0)) if n_bc else 0
    n_frozen = int(np.count_nonzero(vmap >= 0))

    L = _uniform_laplacian(mesh)
    w = np.asarray(weights, dtype=float)[:, None]
    free = 1.0 - w
    alpha = float(relaxation)
    v = mesh.vertices.copy()
    for _ in range(iterations):
        lap = L.dot(v) - v
        v = v + alpha * lap * free
        mesh.vertices = v
        if constrain_bc_planes and n_bc:
            _project_bc_vertices(mesh, vmap, patches)
            v = mesh.vertices.copy()

    return n_bc_tris, n_frozen


def masked_taubin_smooth(
    mesh: trimesh.Trimesh,
    weights: np.ndarray,
    patches: list[Patch],
    tri_labels: np.ndarray,
    *,
    iterations: int = 10,
    lamb: float = 0.5,
    nu: float = 0.53,
    constrain_bc_planes: bool = True,
) -> tuple[int, int]:
    """
    Taubin λ|μ smooth with frozen vertices; optional BC plane projection each iter.

    Returns ``(n_bc_triangles, n_frozen_vertices)`` when BC patches exist.
    """
    if iterations <= 0:
        return 0, 0

    n_bc = sum(1 for p in patches if p.is_bc)
    vmap = bc_patch_vertex_map(mesh, tri_labels, n_bc) if n_bc else np.full(
        len(mesh.vertices), -1, dtype=int,
    )
    n_bc_tris = int(np.count_nonzero(np.asarray(tri_labels) >= 0)) if n_bc else 0
    n_frozen = int(np.count_nonzero(vmap >= 0))

    L = _uniform_laplacian(mesh)
    w = np.asarray(weights, dtype=float)[:, None]
    free = 1.0 - w
    v = mesh.vertices.copy()
    for _ in range(iterations):
        for factor in (float(lamb), -float(nu)):
            lap = L.dot(v) - v
            v = v + factor * lap * free
        mesh.vertices = v
        if constrain_bc_planes and n_bc:
            _project_bc_vertices(mesh, vmap, patches)
            v = mesh.vertices.copy()

    return n_bc_tris, n_frozen


def vertex_patch_map(
    mesh: trimesh.Trimesh,
    tri_labels: np.ndarray,
    n_patches: int,
    min_frac: float = 0.5,
) -> np.ndarray:
    """Vertex → patch id when a majority of incident triangles share it."""
    vmap = np.full(len(mesh.vertices), -1, dtype=int)
    vf = mesh.vertex_faces
    for v in range(len(mesh.vertices)):
        inc = vf[v][vf[v] >= 0]
        if inc.size == 0:
            continue
        lab = tri_labels[inc]
        for pid in range(n_patches):
            if np.count_nonzero(lab == pid) >= min_frac * inc.size:
                vmap[v] = pid
                break
    return vmap


def flatten_bc_planes(
    mesh: trimesh.Trimesh,
    vmap: np.ndarray,
    patches: list[Patch],
) -> None:
    """Flatten all patch-assigned vertices onto their BC plane."""
    for pid, patch in enumerate(patches):
        if not patch.is_bc:
            continue
        vids = np.where(vmap == pid)[0]
        if vids.size:
            mesh.vertices[vids, patch.axis] = patch.plane_coord
