"""
voxel2surf — step 3: cotangent (Laplace-Beltrami weighted)
Taubin smoothing with intermittent refinement.

One constrained pass is  x ← x + f · Pᵢ · (W x − x)ᵢ , where
  W   row-normalized neighbor operator — uniform or cotangent weights,
  Pᵢ  per-vertex motion subspace from label.Constraints (I / I−nnᵀ / 0),
  f   the Taubin coefficient (λ > 0 then μ < 0).
The neighborhood is optionally asymmetric and hierarchical: a box-face vertex
averages only neighbors on its face (fairs as a 2-D region), a box-edge vertex
only same-edge neighbors (fairs as a 1-D curve), corners are fixed, and interior
vertices remain coupled to the boundary. Refinement (midpoint
subdivide / quadric decimate) interleaves with smoothing — ``subdivide`` carries
the box-face provenance mask through (``mask[mid] = mask[a] & mask[b]``) so the
constraint is exact across levels; only the terminal decimation, which drops
parentage, re-derives it geometrically.

NOTE: we want to add in Delaunay flips and remove the obtuse angle gate.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from lib.converters.voxel2surf.label import Constraints, classify


def to_triangles(faces: np.ndarray, *, return_map: bool = False):
    """Split quads into two triangles (triangles pass through unchanged).

    With ``return_map=True`` also return ``parent`` (length = n_tris): the source
    face index of each triangle, so per-face provenance (BC patch id, feature
    flags, …) rides onto the triangulation. For quads the two triangles of face
    ``i`` sit at positions ``i`` and ``F + i`` to match the ``vstack`` order.
    """
    faces = np.asarray(faces)
    if faces.shape[1] == 3:
        tris = faces
        parent = np.arange(len(faces))
    else:
        tris = np.vstack([faces[:, [0, 1, 2]], faces[:, [0, 2, 3]]])
        parent = np.tile(np.arange(len(faces)), 2)
    return (tris, parent) if return_map else tris


def _uniform_edges(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Directed edges (both ways) with unit weight from the face boundary graph."""
    f = np.asarray(faces)
    k = f.shape[1]
    e = np.concatenate([np.stack([f[:, i], f[:, (i + 1) % k]], 1) for i in range(k)], 0)
    e = np.unique(np.sort(e, 1), axis=0)
    src = np.concatenate([e[:, 0], e[:, 1]])
    dst = np.concatenate([e[:, 1], e[:, 0]])
    return src, dst, np.ones(len(src))


def _cotangent_edges(verts, faces) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Directed edges with cotangent weights wᵤᵥ = Σ cot(opposite angle), clamped ≥0."""
    v = np.asarray(verts, dtype=float)
    t = to_triangles(faces)
    i, j, k = t[:, 0], t[:, 1], t[:, 2]

    def cot_at(a, b, c):  # cot of the angle at a in triangle (a, b, c)
        u, w = v[b] - v[a], v[c] - v[a]
        return (u * w).sum(1) / np.maximum(np.linalg.norm(np.cross(u, w), axis=1), 1e-12)

    ci, cj, ck = cot_at(i, j, k), cot_at(j, k, i), cot_at(k, i, j)  # opposite edges (j,k),(k,i),(i,j)
    rows = np.concatenate([j, k, k, i, i, j])
    cols = np.concatenate([k, j, i, k, j, i])
    vals = np.concatenate([ci, ci, cj, cj, ck, ck])
    n = len(v)
    cw = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    cw.data = np.maximum(cw.data, 0.0)  # clamp negative (obtuse) weights for stability
    cw.eliminate_zeros()
    co = cw.tocoo()
    return co.row, co.col, co.data


def build_W(verts, faces, c: Constraints, *, weights="cotangent", asymmetric=True) -> sp.csr_matrix:
    """Row-normalized neighbor-mean operator with optional asymmetric coupling."""
    n = len(verts)
    if weights == "cotangent":
        src, dst, w = _cotangent_edges(verts, faces)
    elif weights == "uniform":
        src, dst, w = _uniform_edges(faces)
    else:
        raise ValueError(f"weights must be 'cotangent' or 'uniform', got {weights!r}")

    if asymmetric:
        # each boundary vertex fairs only within its own (or a deeper) box region:
        # an on_face vertex keeps neighbours on that face, an on_edge vertex keeps
        # only same-edge neighbours (mask ⊇ both its bits). Interior verts keep all.
        sm = c.on_face
        src_m = sm[src]
        constrained = src_m.any(1)
        on_region = ((sm[dst] & src_m) == src_m).all(1)  # dst lies on all of src's faces
        keep = ~constrained | on_region
        src, dst, w = src[keep], dst[keep], w[keep]

    a = sp.coo_matrix((w, (src, dst)), shape=(n, n)).tocsr()
    deg = np.asarray(a.sum(1)).ravel()
    deg[deg == 0] = 1.0
    return sp.diags(1.0 / deg) @ a


def _project(disp: np.ndarray, c: Constraints) -> np.ndarray:
    """Apply the per-vertex motion projector Pᵢ:
      on_face  → strip the plane-normal component   (slide in the plane),
      on_edge  → keep only the edge-tangent component (slide along the edge),
      on_corner/anchor → zero                         (fixed).
    The three classes are disjoint, so the branches don't overlap."""
    out = np.asarray(disp, dtype=float).copy()
    on_f = (c.normal * c.normal).sum(1) > 1e-12
    if on_f.any():
        nn, d = c.normal[on_f], out[on_f]
        out[on_f] = d - (d * nn).sum(1, keepdims=True) * nn
    on_e = (c.tangent * c.tangent).sum(1) > 1e-12
    if on_e.any():
        tt, d = c.tangent[on_e], out[on_e]
        out[on_e] = (d * tt).sum(1, keepdims=True) * tt
    out[c.fixed] = 0.0
    return out


def constrained_taubin(
    verts,
    faces,
    c: Constraints,
    *,
    lamb: float = 0.5,
    mu: float = -0.53,
    iters: int = 10,
    weights: str = "cotangent",
    asymmetric: bool = True,
    schedule: str | None = None,
) -> np.ndarray:
    """Constrained λ|μ fairing. ``schedule`` is a string over {'L','M','R'} that
    overrides the default ``'LM'*iters`` (R = caller-handled refine marker)."""
    n = len(verts)
    w = build_W(verts, faces, c, weights=weights, asymmetric=asymmetric)
    lap = w - sp.identity(n, format="csr")
    v = np.asarray(verts, dtype=float).copy()
    passes = schedule if schedule is not None else "LM" * iters
    for p in passes:
        if p == "R":
            continue
        v = v + (lamb if p == "L" else mu) * _project(lap.dot(v), c)
    return v


# --- implicit variational solve (primary path) -------------------------------
def _cotangent_laplacian(verts, faces) -> sp.csr_matrix:
    """Symmetric cotangent Laplacian ``L = D − A`` (PSD; A = clamped cotangent
    weights). This is the operator whose Dirichlet energy ``xᵀLx`` the membrane
    solve minimizes — the same weights as the Taubin path, un-normalized and
    symmetrized so it forms a proper quadratic for the constrained solve."""
    src, dst, w = _cotangent_edges(verts, faces)
    n = len(verts)
    a = sp.coo_matrix((w, (src, dst)), shape=(n, n)).tocsr()  # symmetric (cotan)
    deg = np.asarray(a.sum(1)).ravel()
    return (sp.diags(deg) - a).tocsr()


def _lumped_mass(verts, faces) -> np.ndarray:
    """Barycentric (lumped Voronoi) vertex areas M_ii = ⅓ Σ incident triangle
    areas — the diagonal mass matrix for the thin-plate operator L M⁻¹ L."""
    verts = np.asarray(verts, dtype=float)
    t = to_triangles(faces)
    v0, v1, v2 = verts[t[:, 0]], verts[t[:, 1]], verts[t[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    m = np.zeros(len(verts))
    for k in range(3):
        np.add.at(m, t[:, k], area / 3.0)
    m[m <= 0] = float(m[m > 0].mean()) if (m > 0).any() else 1.0
    return m


def _solve_axis_bounded(Qr, pinned, target, lo, hi, x0, eps, iters):
    """Bound-constrained scalar QP for one coordinate channel by **primal-dual
    active set** (semismooth Newton): minimize ½ xᵀQx + ½ε‖x−x0‖² s.t.
    x[pinned]=target, lo≤x≤hi.

    On the free (un-pinned) variables ``A y = c + μ`` with bound multiplier ``μ``;
    each round predicts the active set from ``μ + c₀(y − bound)`` — which lets a
    variable *leave* a bound, not just join it — fixes active vars to their bound,
    solves the SPD system on the inactive set, and recomputes ``μ``. Converges to
    the true constrained minimizer, so the surface meets the corridor tangentially
    (no crease) instead of the kink the old clamp-and-freeze produced."""
    from scipy.sparse.linalg import spsolve

    pinned = np.asarray(pinned, dtype=bool)
    out = np.where(pinned, target, x0).astype(float)
    fidx = np.where(~pinned)[0]
    if len(fidx) == 0:
        return out
    pidx = np.where(pinned)[0]

    A = Qr[fidx][:, fidx].tocsc()
    c = eps * x0[fidx]
    if len(pidx):
        c = c - (Qr[fidx][:, pidx] @ target[pidx])   # coupling to the Dirichlet pins
    loF, hiF = lo[fidx], hi[fidx]
    c0 = float(A.diagonal().mean()) or 1.0

    def energy(z):  # ½ zᵀA z − cᵀz on the feasible (clipped) iterate
        zf = np.clip(z, loF, hiF)
        return 0.5 * float(zf @ (A @ zf)) - float(c @ zf), zf

    y = np.clip(x0[fidx], loF, hiF)
    mu = np.zeros(len(fidx))
    best_e, best_y = energy(y)
    prev = None
    for _ in range(max(1, iters)):
        al = (mu + c0 * (y - loF)) < 0          # lower-active
        au = (mu + c0 * (y - hiF)) > 0          # upper-active
        au &= ~al
        free = ~(al | au)
        if prev is not None and np.array_equal(al, prev[0]) and np.array_equal(au, prev[1]):
            break                                # active set fixed → converged
        prev = (al.copy(), au.copy())
        y[al] = loF[al]
        y[au] = hiF[au]
        if free.any():
            rhs = c[free].copy()
            if al.any():
                rhs -= A[free][:, al] @ loF[al]
            if au.any():
                rhs -= A[free][:, au] @ hiF[au]
            y[free] = spsolve(A[free][:, free].tocsc(), rhs)
        e, yf = energy(y)                         # globalization: keep best feasible
        if e < best_e:
            best_e, best_y = e, yf
        mu = np.zeros(len(fidx))
        act = al | au
        if act.any():
            mu[act] = (A[act] @ y) - c[act]
    out[fidx] = best_y
    return out


def solve_implicit(verts, faces, pinned, target, lo, hi, *,
                   order: str = "membrane", iters: int = 6, reg: float = 1e-4):
    """Minimize the fairing energy per coordinate channel subject to the box pins
    (``pinned``→``target``) and the occupancy corridor ``[lo, hi]`` — the implicit
    fair-and-constrain. ``order`` is ``"membrane"`` (Dirichlet, `Q=L`) or
    ``"thin-plate"`` (bi-Laplacian, `Q=L M⁻¹ L`, C¹ at constraints). ``reg`` weights a
    soft pull ``½·reg·mean(diagQ)·‖x−x0‖²`` toward the input positions: this is the
    **anti-shrink data term** (binary gives no better target than "stay on the
    cuberille boundary") — the thin-plate energy mostly wants *normal* (shrinking)
    motion, so this isotropic spring resists shrinkage while leaving tangential
    smoothing free, and it keeps `Q` SPD/well-conditioned (no folds)."""
    verts = np.asarray(verts, dtype=float)
    n = len(verts)
    L = _cotangent_laplacian(verts, faces)
    if order == "thin-plate":
        minv = sp.diags(1.0 / _lumped_mass(verts, faces))
        Q = (L @ minv @ L).tocsr()  # mass-weighted bi-Laplacian (geometrically correct)
    elif order == "membrane":
        Q = L
    else:
        raise ValueError(f"order must be 'membrane' or 'thin-plate', got {order!r}")
    d = Q.diagonal()
    eps = reg * (float(np.mean(d[d > 0])) if (d > 0).any() else 1.0)
    Qr = (Q + eps * sp.identity(n)).tocsr()
    pinned = np.asarray(pinned, dtype=bool)
    target = np.asarray(target, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    x = verts.copy()
    for ax in range(3):
        x[:, ax] = _solve_axis_bounded(
            Qr, pinned[:, ax], target[:, ax], lo[:, ax], hi[:, ax], verts[:, ax], eps, iters)
    return x


# --- Taubin smooth interpolation (§4.2) --------------------------------------
def free_constraints(n: int) -> Constraints:
    """All-free constraint set (Pᵢ = I everywhere) for unconstrained fairing."""
    z3 = np.zeros((n, 3))
    return Constraints(
        fixed=np.zeros(n, dtype=bool), normal=z3.copy(), tangent=z3.copy(),
        on_face=np.zeros((n, 6), dtype=bool), face_idx=np.full(n, -1, dtype=int))


def _uniform_laplacian(faces, n) -> sp.csr_matrix:
    """L = W_uniform − I — the row-normalized topological Laplacian. Purely
    combinatorial (no vertex positions), so the kernel built from it is fixed and
    precomputable, independent of the cotangent fairing of the data."""
    src, dst, w = _uniform_edges(faces)
    a = sp.coo_matrix((w, (src, dst)), shape=(n, n)).tocsr()
    deg = np.asarray(a.sum(1)).ravel()
    deg[deg == 0] = 1.0
    return (sp.diags(1.0 / deg) @ a) - sp.identity(n, format="csr")


def smooth_interpolate(
    verts_faired,
    faces,
    on_face,
    *,
    origin,
    spacing,
    shape,
    anchor_verts=None,
    anchor_targets=None,
    lamb: float = 0.5,
    mu: float = -0.53,
    kernel_iters: int = 5,
) -> np.ndarray:
    """Taubin §4.2 smooth interpolation of the box / anchor constraints.

    Adds a smooth correction to the *unconstrained* faired surface ``verts_faired``
    (= xᴺ) so every constraint is met exactly:

        x_C = xᴺ + F_nm F_mm⁻¹ (t − xᴺ_C)            (eqn 10)

    applied independently per coordinate channel. Because the box planes are
    axis-aligned, each face pins exactly one coordinate, so the constrained set of
    channel ``a`` is just the vertices carrying an ``a``-face bit; an on_edge vertex
    lands in two channels and a corner in three, reproducing the free/face/edge/
    corner hierarchy automatically. Anchors pin all three channels to their
    original position.

    The blending kernel F = [(I−μK)(I−λK)]^kernel_iters uses the FIXED uniform
    Laplacian (K = −L_uniform), so its |C| impulse responses F[:, C] are a one-time
    precompute and the correction is exact no matter how xᴺ was faired (cotangent
    or otherwise). ``kernel_iters`` sets the *locality* (bump width) of the
    correction and is decoupled from the data-fairing iteration count: it must stay
    small, because F_mm conditioning degrades fast with it (~1e6 at 5, ~1e18 at 50),
    and a wide low-pass kernel makes neighbouring constraints' bumps collinear →
    F_mm singular → the correction blows up. Returns the corrected (V, 3) positions.
    """
    x = np.array(verts_faired, dtype=float, copy=True)
    on_face = np.asarray(on_face, dtype=bool)
    origin = np.asarray(origin, dtype=float)
    spacing = np.asarray(spacing, dtype=float)
    shape = np.asarray(shape, dtype=int)
    V = len(x)
    hi = origin + shape * spacing

    pinned = np.zeros((V, 3), dtype=bool)
    target = np.zeros((V, 3), dtype=float)
    for a in range(3):
        lo_m, hi_m = on_face[:, 2 * a], on_face[:, 2 * a + 1]
        pinned[:, a] = lo_m | hi_m
        target[lo_m, a] = origin[a]
        target[hi_m, a] = hi[a]  # max overrides min in a 1-thick degenerate axis
    if anchor_verts is not None and len(anchor_verts):
        pinned[anchor_verts] = True
        target[anchor_verts] = anchor_targets

    cols = np.where(pinned.any(1))[0]  # union constrained set C (sorted ascending)
    if len(cols) == 0:
        return x

    # one-time precompute: F[:, C] = fixed uniform kernel applied to |C| impulses
    L = _uniform_laplacian(faces, V)
    F = np.zeros((V, len(cols)))
    F[cols, np.arange(len(cols))] = 1.0
    for _ in range(kernel_iters):
        F = F + lamb * (L @ F)
        F = F + mu * (L @ F)

    for a in range(3):
        ca = np.where(pinned[:, a])[0]
        if len(ca) == 0:
            continue
        loc = np.searchsorted(cols, ca)  # column positions of C_a inside C
        fmm = F[ca][:, loc]
        r = target[ca, a] - x[ca, a]
        # light diagonal (Tikhonov) insurance: harmless when well-conditioned,
        # bounds the correction if kernel_iters is pushed too high.
        eps = 1e-9 * float(np.median(np.abs(np.diag(fmm))) or 1.0)
        try:
            sol = np.linalg.solve(fmm + eps * np.eye(len(ca)), r)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(fmm, r, rcond=None)[0]
        x[:, a] += F[:, loc] @ sol
    return x


# --- refinement (decimate needs pyvista) -------------------------------------
def subdivide(verts, faces, mask=None):
    """One 1→4 midpoint subdivision; returns triangles (input quads triangulated).

    Implemented in numpy (no trimesh) so a per-vertex provenance ``mask`` (V, k)
    can ride through: each new edge-midpoint vertex inherits ``mask[a] & mask[b]``
    from its two parents. For the box-face mask this is exact — a midpoint lies
    on a coordinate plane iff *both* endpoints do, which is precisely the bitwise
    AND. Original vertices keep indices ``0..V-1`` (and their mask rows); unique
    edge ``k`` becomes vertex ``V + k``.

    Returns ``(verts, faces)`` normally, or ``(verts, faces, mask)`` when ``mask``
    is supplied.
    """
    verts = np.asarray(verts, dtype=float)
    tris = to_triangles(faces)
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]

    # unique undirected edges; inv maps each (local-edge, tri) → unique edge id.
    e = np.concatenate([
        np.sort(np.stack([a, b], 1), 1),
        np.sort(np.stack([b, c], 1), 1),
        np.sort(np.stack([c, a], 1), 1),
    ], 0)
    ue, inv = np.unique(e, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel().reshape(3, len(tris))  # rows: ab, bc, ca

    V = len(verts)
    mab, mbc, mca = V + inv[0], V + inv[1], V + inv[2]
    mid = 0.5 * (verts[ue[:, 0]] + verts[ue[:, 1]])
    new_verts = np.vstack([verts, mid])
    new_tris = np.vstack([
        np.stack([a, mab, mca], 1),
        np.stack([b, mbc, mab], 1),
        np.stack([c, mca, mbc], 1),
        np.stack([mab, mbc, mca], 1),
    ])
    if mask is None:
        return new_verts, new_tris
    mask = np.asarray(mask, dtype=bool)
    new_mask = np.vstack([mask, mask[ue[:, 0]] & mask[ue[:, 1]]])
    return new_verts, new_tris, new_mask


def decimate_to(verts, faces, target_faces: int):
    """Quadric edge-collapse to ~``target_faces`` (boundary-preserving) via pyvista."""
    import pyvista as pv

    tris = to_triangles(faces)
    pd = pv.PolyData(np.asarray(verts), np.c_[np.full(len(tris), 3), tris].ravel())
    reduction = max(0.0, 1.0 - target_faces / len(tris))
    out = pd.decimate_pro(reduction, preserve_topology=True, boundary_vertex_deletion=False)
    return np.asarray(out.points, dtype=float), out.faces.reshape(-1, 4)[:, 1:]


def run(
    verts,
    faces,
    *,
    origin,
    spacing,
    shape,
    anchors=None,
    cycles: int = 2,
    sub_levels: int = 1,
    iters: int = 10,
    target_faces: int | None = None,
    weights: str = "cotangent",
    asymmetric: bool = True,
):
    """Interleave: (subdivide → constrained smooth) × cycles, then optional
    decimate-to-target + final smooth. Constraints re-derived geometrically each round."""
    v, f = np.asarray(verts, dtype=float), np.asarray(faces)
    for _ in range(cycles):
        for _ in range(sub_levels):
            v, f = subdivide(v, f)
        c = classify(v, origin, spacing, shape, anchors=anchors)
        v = constrained_taubin(v, f, c, iters=iters, weights=weights, asymmetric=asymmetric)
    if target_faces is not None and len(to_triangles(f)) > target_faces:
        v, f = decimate_to(v, f, target_faces)
        c = classify(v, origin, spacing, shape, anchors=anchors)
        v = constrained_taubin(v, f, c, iters=iters, weights=weights, asymmetric=asymmetric)
    return v, f
