"""
voxel2surf — step 2: labelling for constrained fairing.

The *smoothing constraint* is read off a per-vertex 6-bit box-face mask
(bit ``2*ax+side``: side 0 = min plane, side 1 = max plane). Because the box
planes are axis-aligned, each face pins one *coordinate*, so the admissible
motion is fixed by the number of *distinct axes* the vertex is pinned in:
  - 0 axes → free     P = I             (3 DOF),
  - 1 axis → on_face  P = I − n nᵀ      (2 DOF, slide in the plane),
  - 2 axes → on_edge  P = d dᵀ          (1 DOF, slide along the edge line),
  - 3 axes → on_corner P = 0            (0 DOF, fixed).
The plane normal n is the pinned axis; the edge tangent d is the single *free*
axis. NITO load/support anchors are additionally fixed (P = 0).

The mask is *provenance*, not geometry: ``extract.cuberille_mesh`` establishes it
exactly from the integer grid key, and ``fair.subdivide`` carries it through
midpoint refinement via ``mask[mid] = mask[a] & mask[b]`` — a midpoint lies on
a coordinate plane iff both its parents do, so bitwise AND is exactly the
plane-intersection rule. ``classify(..., face_mask=mask)`` consumes it directly.
When no mask is passed (the post-decimation polish, where vertex parentage is
lost) ``classify`` falls back to the geometric coordinate test against the box.

BC-patch *face labels* (which faces carry loads/supports, for the .vtp cell_data)
follow the *same* provenance flow: ``bc_quad_ids`` assigns them exactly on the
cuberille quads, ``bc_vertex_mask`` lifts them to a per-vertex mask that rides
the subdivide AND alongside the box-face mask, and ``face_bc_arrays`` reads the
final face labels back off that mask (no post-mesh geometric re-derivation).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Constraints:
    """Per-vertex motion constraint for the fairing operator.

    The four classes are mutually exclusive and selected by the number of
    distinct box axes a vertex is pinned in (``free`` carries neither vector):
    ``normal`` is set only for on_face verts, ``tangent`` only for on_edge verts,
    ``fixed`` only for on_corner verts (and load anchors)."""

    fixed: np.ndarray    # (V,) bool         — on_corner / anchor → P = 0
    normal: np.ndarray   # (V, 3) pinned-axis unit normal for on_face verts, else 0 — P = I − nnᵀ
    tangent: np.ndarray  # (V, 3) free-axis unit tangent for on_edge verts, else 0 — P = ddᵀ
    on_face: np.ndarray  # (V, 6) bool       — which box faces (axis*2 + side) the vertex lies on
    face_idx: np.ndarray  # (V,) int         — the single face id for on_face verts, else -1


def _box_face_mask(verts, origin, shape, spacing, tol) -> np.ndarray:
    """Geometric fallback: (V, 6) box-face membership by coordinate distance."""
    lo, hi = origin, origin + shape * spacing
    at_lo = np.abs(verts - lo) <= tol
    at_hi = np.abs(verts - hi) <= tol
    on_face = np.zeros((len(verts), 6), dtype=bool)
    for ax in range(3):
        on_face[:, 2 * ax] = at_lo[:, ax]
        on_face[:, 2 * ax + 1] = at_hi[:, ax]
    return on_face


def classify(
    verts: np.ndarray,
    origin=(0.0, 0.0, 0.0),
    spacing=(1.0, 1.0, 1.0),
    shape=None,
    *,
    anchors: np.ndarray | None = None,
    tol: float | None = None,
    face_mask: np.ndarray | None = None,
) -> Constraints:
    """Classify each vertex as free / on_face / on_edge / on_corner.

    With ``face_mask`` (V, 6) the membership comes straight from provenance and
    is taken as ground truth. Without it, membership is re-derived geometrically
    by testing each coordinate against the box extents (``tol`` defaults tight).
    The class is then set by the number of *distinct axes* pinned (1/2/3 →
    face/edge/corner); the derivation is identical either way — only the source
    of the 6-bit mask differs.
    """
    verts = np.asarray(verts, dtype=float)
    origin = np.asarray(origin, dtype=float)
    spacing = np.asarray(spacing, dtype=float)
    if shape is None:
        raise ValueError("shape (nx, ny, nz) is required to locate the domain box")
    shape = np.asarray(shape, dtype=int)

    if face_mask is not None:
        on_face = np.asarray(face_mask, dtype=bool)
        if on_face.shape != (len(verts), 6):
            raise ValueError(
                f"face_mask must be (V, 6) = ({len(verts)}, 6), got {on_face.shape}")
    else:
        if tol is None:
            # tight: only vertices *exactly* on a box plane are constrained. A
            # loose tol catches subdivision midpoints a fraction of a cell off
            # the plane, which never get snapped on (the projection only removes
            # the normal *displacement*) → a spurious BC-plane residual.
            tol = 1e-6
        on_face = _box_face_mask(verts, origin, shape, spacing, tol)

    # class is set by the number of distinct axes pinned (each box face pins one
    # coordinate): 0 free, 1 on_face (slide in plane), 2 on_edge (slide along the
    # edge line), 3 on_corner (fixed). A single axis pinned on *both* sides (only
    # possible in a 1-voxel-thick domain) still counts as that one axis.
    V = len(verts)
    ax_hit = np.zeros((V, 3), dtype=bool)
    for a in range(3):
        ax_hit[:, a] = on_face[:, 2 * a] | on_face[:, 2 * a + 1]
    n_ax = ax_hit.sum(axis=1)

    onface = n_ax == 1
    onedge = n_ax == 2
    fixed = n_ax >= 3

    # on_face → plane normal of the single pinned axis (sign: + at max plane)
    normal = np.zeros_like(verts)
    face_idx = np.full(V, -1, dtype=int)
    iface = np.where(onface)[0]
    if len(iface):
        fa = np.argmax(ax_hit[iface], axis=1)                 # the one pinned axis
        at_max = on_face[iface, 2 * fa + 1]
        normal[iface, fa] = np.where(at_max, 1.0, -1.0)
        face_idx[iface] = 2 * fa + at_max.astype(int)         # representative face id

    # on_edge → tangent of the single *free* axis (the one with no bit)
    tangent = np.zeros_like(verts)
    iedge = np.where(onedge)[0]
    if len(iedge):
        free_ax = np.argmin(ax_hit[iedge], axis=1)            # the lone unpinned axis
        tangent[iedge, free_ax] = 1.0

    if anchors is not None and len(anchors):
        from scipy.spatial import cKDTree

        _, idx = cKDTree(verts).query(np.asarray(anchors, dtype=float))
        anchored = np.unique(np.atleast_1d(idx).astype(int))
        fixed[anchored] = True
        normal[anchored] = 0.0          # anchors override to P = 0
        tangent[anchored] = 0.0
        face_idx[anchored] = -1

    return Constraints(
        fixed=fixed, normal=normal, tangent=tangent, on_face=on_face, face_idx=face_idx)

# --- BC patch labels: same provenance flow as the box-face mask ---------------
# Established exactly on the cuberille quads (integer voxel-face identity, no
# geometry), lifted to a per-vertex multi-hot mask, then propagated through
# subdivision by the *same* bitwise AND that carries the box-face mask. Output
# face labels fall out of the mask by "all corners agree", so there is no
# post-mesh geometric re-derivation (the source of the old spill-over artifacts).


def bc_quad_ids(prov, patches, shape) -> np.ndarray:
    """(F,) cuberille-quad → BC-patch id (or -1), matched *exactly* on the
    integer voxel-face provenance ``(cell, axis, side)``.

    A quad is patch ``p`` iff it shares ``p``'s axis, lies on ``p``'s domain
    boundary slab (so an interior exposed face with a coinciding in-plane
    footprint is *not* caught), and its in-plane cell is in ``p``'s footprint.
    ``patches`` are duck-typed (``axis`` int, ``side`` "min"/"max", ``faces``
    in-plane footprint cells, ``kind`` "bc") — no import of the BC module.
    """
    cell = np.asarray(prov["cell"])
    axis = np.asarray(prov["axis"])
    side = np.asarray(prov["side"])
    shape = np.asarray(shape, dtype=int)
    ids = np.full(len(axis), -1, dtype=np.int32)
    for pid, p in enumerate(patches):
        if getattr(p, "kind", "bc") != "bc" or not getattr(p, "faces", None):
            continue
        ax = int(p.axis)
        pside = 0 if p.side == "min" else 1
        slab = 0 if pside == 0 else int(shape[ax]) - 1
        b, d = (a for a in range(3) if a != ax)
        foot = {(int(f[0]), int(f[1])) for f in p.faces}
        cand = np.where((axis == ax) & (side == pside) & (cell[:, ax] == slab) & (ids < 0))[0]
        for f in cand:
            if (int(cell[f, b]), int(cell[f, d])) in foot:
                ids[f] = pid
    return ids


def bc_vertex_mask(faces, quad_ids, n_verts, n_patches) -> np.ndarray:
    """Lift per-quad ids to a ``(V, P)`` bool mask: vertex on patch ``p`` iff it
    is a corner of a quad labelled ``p``. This is the BC analogue of the
    box-face ``vmask``; ``fair.subdivide`` propagates it by ``mask[mid] =
    mask[a] & mask[b]`` (a midpoint is in the footprint iff both parents are)."""
    faces = np.asarray(faces)
    m = np.zeros((n_verts, max(n_patches, 0)), dtype=bool)
    for pid in range(n_patches):
        q = faces[quad_ids == pid]
        if len(q):
            m[np.unique(q), pid] = True
    return m


def face_bc_arrays(faces, bc_vmask, bc_patches) -> dict:
    """Per-face cell_data for the .vtp from the propagated BC vertex mask: a face
    is patch ``p`` iff *all* its corners carry ``p``. Returns ``patch_id``,
    ``fix_x/y/z`` (constrained DOFs from the patch flags) and ``boundary_type``
    (0=free, 1=support). Works for triangle or quad faces alike."""
    faces = np.asarray(faces)
    n = len(faces)
    bc_vmask = np.asarray(bc_vmask, dtype=bool)
    labels = np.full(n, -1, dtype=np.int32)
    if bc_vmask.shape[1] and n:
        all_share = bc_vmask[faces].all(axis=1)  # (F, P): every corner carries p
        hit = all_share.any(axis=1)
        labels[hit] = np.argmax(all_share[hit], axis=1)

    fix = np.zeros((n, 3), dtype=np.float32)
    btype = np.zeros(n, dtype=np.int32)
    for pid, p in enumerate(bc_patches):
        m = labels == pid
        if not m.any():
            continue
        fl = np.asarray(p.flags, dtype=float).ravel()
        fix[m, : min(3, len(fl))] = fl[:3]
        btype[m] = 1
    return {
        "patch_id": labels,
        "fix_x": fix[:, 0],
        "fix_y": fix[:, 1],
        "fix_z": fix[:, 2],
        "boundary_type": btype,
    }
