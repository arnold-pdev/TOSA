"""
voxel2surf — meshing pipeline (orchestrator + validation).

cuberille+weld (step 1) → geometric labelling (step 2) → constrained Taubin +
refine (step 3) → validate → tagged .vtp. Verbose per-stage logging and an
end-of-run validation report mirror the voxel2surf pipeline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from lib.converters.voxel2surf import corridor, fair
from lib.converters.voxel2surf.extract import cuberille_mesh
from lib.converters.voxel2surf.label import (
    bc_quad_ids,
    bc_vertex_mask,
    classify,
    face_bc_arrays,
)


@dataclass
class Options:
    density_cutoff: float = 0.5
    with_bc: bool = True
    iters: int = 50            # Taubin λ|μ iterations per fairing pass
    lamb: float = 0.5          # λ=0.5 zeroes the mesh Nyquist mode = the staircase
    mu: float = -0.53          # μ from pass-band k_PB≈0.1 (non-shrinking)
    target_faces: int | None = None  # final triangle budget; None → no decimate
    sub_levels: int = 0        # unconditional midpoint subdivisions (schedule control)
    max_sub_levels: int = 2    # extra cap on the target-driven subdivisions
    fair_mode: str = "implicit"  # "implicit" (primary: energy + corridor) | "taubin" (fast fallback)
    smooth_order: str = "thin-plate"  # implicit energy: "thin-plate" (non-shrinking) | "membrane" (shrinks)
    data_weight: float = 1e-2      # soft pull to the cuberille position — the anti-shrink/hug↔smooth knob
    corridor_voxels: float = 1.5   # SDF corridor normal half-width (voxels); loose safety + thin-feature protection
    implicit_iters: int = 12       # primal-dual active-set rounds (PDAS converges in ~10)
    # --- taubin (fast fallback) only ---
    weights: str = "cotangent"
    asymmetric: bool = True
    smooth_interp: bool = False  # Taubin §4.2: fair unconstrained, then exact smooth correction
    smooth_kernel_iters: int = 5  # locality of the §4.2 correction (keep small: F_mm conditioning)
    on_fail: str = "warn"  # "warn" | "raise" — gate on self-intersection/overlap/watertight/BC


class Logger:
    """Echo to stdout and/or append to a log file (voxel2surf-style)."""

    def __init__(self, echo: bool = True, path: str | Path | None = None):
        self.echo = echo
        self._fh = open(path, "w") if path else None

    def __call__(self, msg: str = "") -> None:
        if self.echo:
            print(msg)
        if self._fh:
            self._fh.write(msg + "\n")
            self._fh.flush()

    def close(self) -> None:
        if self._fh:
            self._fh.close()


# --- mesh statistics (numpy/scipy only — works in any env) -------------------
def _edge_incidence(faces):
    f = np.asarray(faces)
    k = f.shape[1]
    e = np.concatenate(
        [np.sort(np.stack([f[:, i], f[:, (i + 1) % k]], 1), 1) for i in range(k)], 0
    )
    return np.unique(e, axis=0, return_counts=True)


def _topo(verts, faces) -> dict:
    eu, cnt = _edge_incidence(faces)
    n = len(verts)
    a = sp.coo_matrix((np.ones(len(eu)), (eu[:, 0], eu[:, 1])), shape=(n, n))
    _, lab = connected_components(a, directed=False)
    used = np.unique(np.asarray(faces))
    return {
        "verts": len(verts),
        "faces": len(faces),
        "bodies": int(len(np.unique(lab[used]))) if len(used) else 0,
        "watertight": bool((cnt == 2).all()),
        "nonmanifold_edges": int((cnt != 2).sum()),
    }


def _signed_volume(verts, faces) -> float:
    t = fair.to_triangles(faces)
    v0, v1, v2 = verts[t[:, 0]], verts[t[:, 1]], verts[t[:, 2]]
    return float(np.einsum("ij,ij->i", np.cross(v1 - v0, v2 - v0), v0).sum() / 6.0)


def _free_dihedral_p95(verts, faces, origin, spacing, shape) -> float:
    t = fair.to_triangles(faces)
    if len(t) == 0:
        return 0.0
    v0, v1, v2 = verts[t[:, 0]], verts[t[:, 1]], verts[t[:, 2]]
    nrm = np.cross(v1 - v0, v2 - v0)
    nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    cen = verts[t].mean(1)
    tolb = 0.25 * float(np.min(spacing))
    lo = np.asarray(origin, float)
    hi = lo + np.asarray(shape, int) * np.asarray(spacing, float)
    free = ~((np.abs(cen - lo) <= tolb).any(1) | (np.abs(cen - hi) <= tolb).any(1))
    e = np.concatenate([np.sort(np.stack([t[:, i], t[:, (i + 1) % 3]], 1), 1) for i in range(3)], 0)
    tid = np.tile(np.arange(len(t)), 3)
    eu, inv = np.unique(e, axis=0, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    inv_o, tid_o = inv[order], tid[order]
    slot = np.arange(len(inv_o)) - np.searchsorted(inv_o, np.arange(len(eu)))[inv_o]
    et = np.full((len(eu), 2), -1, dtype=int)
    k2 = slot < 2
    et[inv_o[k2], slot[k2]] = tid_o[k2]
    man = (et[:, 0] >= 0) & (et[:, 1] >= 0)
    both_free = man & free[et[:, 0]] & free[et[:, 1]]
    if not both_free.any():
        return 0.0
    dot = np.abs((nrm[et[both_free, 0]] * nrm[et[both_free, 1]]).sum(1)).clip(-1, 1)
    return float(np.percentile(np.degrees(np.arccos(dot)), 95))


def _interval(d, v):
    pos, neg = d > 1e-12, d < -1e-12
    if pos.sum() == 1:
        lone, others = int(np.where(pos)[0][0]), np.where(~pos)[0]
    elif neg.sum() == 1:
        lone, others = int(np.where(neg)[0][0]), np.where(~neg)[0]
    else:
        z = np.where(np.abs(d) <= 1e-12)[0]
        if len(z) == 0:
            return None
        lone, others = int(z[0]), np.array([i for i in range(3) if i != z[0]])
    ts = []
    for o in others:
        den = d[lone] - d[o]
        ts.append(v[lone] if abs(den) < 1e-15 else v[lone] + (v[o] - v[lone]) * d[lone] / den)
    return (min(ts), max(ts))


def _tri_tri(p, q, eps=1e-9):
    """Möller (1997) triangle-triangle intersection test for 3×3 arrays p, q."""
    p0, p1, p2 = p
    q0, q1, q2 = q
    n2 = np.cross(q1 - q0, q2 - q0)
    dp = n2 @ p.T - n2.dot(q0)
    if (dp > eps).all() or (dp < -eps).all():
        return False
    n1 = np.cross(p1 - p0, p2 - p0)
    dq = n1 @ q.T - n1.dot(p0)
    if (dq > eps).all() or (dq < -eps).all():
        return False
    d = np.cross(n1, n2)
    if d @ d < eps:  # coplanar: 2D edge-cross or containment
        ax = int(np.argmax(np.abs(n1)))
        keep = [i for i in range(3) if i != ax]
        a, b = p[:, keep], q[:, keep]

        def o(u, w, r):
            return np.sign((w[0] - u[0]) * (r[1] - u[1]) - (w[1] - u[1]) * (r[0] - u[0]))

        for i in range(3):
            for j in range(3):
                if o(a[i], a[(i + 1) % 3], b[j]) != o(a[i], a[(i + 1) % 3], b[(j + 1) % 3]) and \
                   o(b[j], b[(j + 1) % 3], a[i]) != o(b[j], b[(j + 1) % 3], a[(i + 1) % 3]):
                    return True

        def inside(pt, tri):
            s = [np.sign(np.cross(tri[(k + 1) % 3] - tri[k], pt - tri[k])) for k in range(3)]
            return all(x >= 0 for x in s) or all(x <= 0 for x in s)

        return inside(a[0], b) or inside(b[0], a)
    idx = int(np.argmax(np.abs(d)))
    i1, i2 = _interval(dp, p[:, idx]), _interval(dq, q[:, idx])
    if i1 is None or i2 is None:
        return False
    return max(i1[0], i2[0]) <= min(i1[1], i2[1]) + eps


def _self_intersections(verts, faces, *, origin=None, spacing=None, shape=None) -> int:
    """Count intersecting non-adjacent triangle pairs in the *free* skin.

    Triangles lying on a domain box plane (the BC faces) are excluded: they are
    exact planar tessellations that cannot self-intersect, and — being large and
    coplanar — would otherwise flood the broadphase with pairs that all fall into
    the slow coplanar branch. The realistic failure mode (thin free members
    folding onto themselves) is between free triangles, checked here in full.

    Broadphase uses the *median* triangle diameter (robust to the few large
    decimated faces) and a vectorized AABB + Möller plane early-reject, so only
    genuine plane-crossing pairs reach the Python narrowphase. Numpy/scipy only.
    """
    from scipy.spatial import cKDTree

    t = fair.to_triangles(faces)
    if len(t) < 2:
        return 0
    V = np.asarray(verts, float)
    cen_all = V[t].mean(1)
    if origin is not None and shape is not None:
        lo = np.asarray(origin, float)
        hi = lo + np.asarray(shape, int) * np.asarray(spacing, float)
        on_box = (np.abs(cen_all - lo) <= 1e-6).any(1) | (np.abs(cen_all - hi) <= 1e-6).any(1)
        t = t[~on_box]
    if len(t) < 2:
        return 0

    p = V[t]  # (n, 3, 3)
    cen = p.mean(1)
    rad = np.linalg.norm(p - cen[:, None, :], axis=2).max(1)   # per-triangle radius
    r = 2.0 * float(np.median(rad)) + 1e-12
    pairs = cKDTree(cen).query_pairs(r, output_type="ndarray")
    if len(pairs) == 0:
        return 0
    a, b = pairs[:, 0], pairs[:, 1]
    shared = (t[a][:, :, None] == t[b][:, None, :]).any(axis=(1, 2))  # edge/vertex-adjacent
    a, b = a[~shared], b[~shared]
    if len(a) == 0:
        return 0

    PA, PB = p[a], p[b]
    ov = ((PA.min(1) <= PB.max(1)) & (PB.min(1) <= PA.max(1))).all(1)  # tight AABB overlap
    a, b, PA, PB = a[ov], b[ov], PA[ov], PB[ov]
    if len(a) == 0:
        return 0

    eps = 1e-9  # vectorized Möller plane test: all of A one side of B's plane → no hit
    nB = np.cross(PB[:, 1] - PB[:, 0], PB[:, 2] - PB[:, 0])
    dA = np.einsum("mij,mj->mi", PA - PB[:, :1], nB)
    nA = np.cross(PA[:, 1] - PA[:, 0], PA[:, 2] - PA[:, 0])
    dB = np.einsum("mij,mj->mi", PB - PA[:, :1], nA)
    sep = ((dA > eps).all(1) | (dA < -eps).all(1)) | ((dB > eps).all(1) | (dB < -eps).all(1))
    a, b = a[~sep], b[~sep]
    return int(sum(_tri_tri(p[i], p[j]) for i, j in zip(a, b)))


def _body_overlaps(verts, faces):
    """Inter-body interpenetration (None if trimesh unavailable)."""
    try:
        import trimesh
    except Exception:
        return None
    m = trimesh.Trimesh(np.asarray(verts), fair.to_triangles(faces), process=False)
    try:
        bodies = m.split(only_watertight=False)
    except Exception:
        return None
    n = 0
    for i in range(len(bodies)):
        for j in range(i + 1, len(bodies)):
            ai = bodies[j].contains(bodies[i].vertices).any() if len(bodies[i].vertices) else False
            aj = bodies[i].contains(bodies[j].vertices).any() if len(bodies[j].vertices) else False
            n += int(bool(ai or aj))
    return n


def validate(verts, faces, *, vox, origin, spacing, shape, anchors, log: Logger,
             on_fail: str = "warn", bc_tol: float = 1e-5) -> dict:
    origin = np.asarray(origin, float)
    spacing = np.asarray(spacing, float)
    shape = np.asarray(shape, int)
    rep = _topo(verts, faces)

    vol = abs(_signed_volume(verts, faces))
    domain = float(np.prod(shape * spacing))
    rep["vf_voxel"] = float(np.count_nonzero(vox)) / float(np.prod(shape))
    rep["vf_mesh"] = (vol / domain) if rep["watertight"] else float("nan")
    rep["vf_delta"] = rep["vf_mesh"] - rep["vf_voxel"]

    c = classify(verts, origin, spacing, shape, anchors=anchors)
    onp = c.face_idx >= 0
    if onp.any():
        fi = c.face_idx[onp]
        ax = fi // 2
        ext = np.where(fi % 2 == 1, (origin + shape * spacing)[ax], origin[ax])
        vax = verts[onp][np.arange(int(onp.sum())), ax]
        rep["bc_plane_max_residual"] = float(np.abs(vax - ext).max())
    else:
        rep["bc_plane_max_residual"] = 0.0

    rep["load_off_surface"], rep["load_max_dist"] = 0, 0.0
    if anchors is not None and len(anchors):
        from scipy.spatial import cKDTree

        d, _ = cKDTree(verts).query(np.asarray(anchors, float))
        rep["load_max_dist"] = float(d.max())
        rep["load_off_surface"] = int((d > 0.5 * float(spacing.min())).sum())

    rep["free_dihedral_p95_deg"] = _free_dihedral_p95(verts, faces, origin, spacing, shape)
    rep["self_intersections"] = _self_intersections(
        verts, faces, origin=origin, spacing=spacing, shape=shape)
    rep["body_overlaps"] = _body_overlaps(verts, faces)

    failed = []
    if not rep["watertight"]:
        failed.append("not-watertight")
    if rep["self_intersections"] > 0:
        failed.append(f"self_intersections={rep['self_intersections']}")
    if rep["body_overlaps"] not in (0, None):
        failed.append(f"body_overlaps={rep['body_overlaps']}")
    if rep["bc_plane_max_residual"] > bc_tol:
        failed.append(f"bc_plane_residual>{bc_tol:g}")
    if rep["load_off_surface"] > 0:
        failed.append(f"load_off_surface={rep['load_off_surface']}")
    rep["gates_failed"] = failed

    ov = "skipped" if rep["body_overlaps"] is None else rep["body_overlaps"]
    log("  validate:")
    log(f"        watertight={rep['watertight']}  bodies={rep['bodies']}  "
        f"nonmanifold_edges={rep['nonmanifold_edges']}")
    log(f"        vf_mesh={rep['vf_mesh']:.4f}  vf_voxel={rep['vf_voxel']:.4f}  "
        f"vf_delta={rep['vf_delta']:+.4f}")
    log(f"        bc_plane_max_residual={rep['bc_plane_max_residual']:.2e}  "
        f"load_off_surface={rep['load_off_surface']} (max_dist={rep['load_max_dist']:.3f})")
    log(f"        free_dihedral_p95={rep['free_dihedral_p95_deg']:.1f}deg  "
        f"self_intersections={rep['self_intersections']}  body_overlaps={ov}")
    log(f"  gates: {'PASS' if not failed else 'FAIL -> ' + ', '.join(failed)}")
    if failed and on_fail == "raise":
        raise ValueError("validation gates failed: " + ", ".join(failed))
    return rep


def mesh_surface(vox, bc_patches, load_patches, shape, origin, spacing, opts: Options, log: Logger):
    """Full pipeline with per-stage logging. Returns (verts, faces, cell_arrays, report)."""
    t0 = time.perf_counter()
    origin = np.asarray(origin, float)
    spacing = np.asarray(spacing, float)
    shape = np.asarray(shape, int)
    log(f"  shape={tuple(int(x) for x in shape)}  solid_voxels={int(np.count_nonzero(vox)):,}")
    anchors = np.asarray([p.anchor[:3] for p in load_patches], float) if load_patches else None

    ts = time.perf_counter()
    verts, faces, prov = cuberille_mesh(vox, origin, spacing, density_cutoff=0.0)
    vmask = prov["vmask"]  # (V, 6) provenance box-face mask, propagated through subdivide
    st = _topo(verts, faces)
    log(f"  extract [cuberille+weld] — verts={st['verts']:,} faces={st['faces']:,} "
        f"bodies={st['bodies']} watertight={st['watertight']}  ({time.perf_counter() - ts:.2f}s)")

    # BC patches → per-quad ids (exact integer voxel-face match), lifted to a
    # per-vertex (V, P) mask that rides the *same* subdivide AND as vmask. No
    # post-mesh geometric transfer; output face labels come from this mask.
    use_bc = opts.with_bc and bool(bc_patches)
    n_bc = len(bc_patches) if use_bc else 0
    if n_bc:
        qids = bc_quad_ids(prov, bc_patches, shape)
        bc_vmask = bc_vertex_mask(faces, qids, len(verts), n_bc)
        log(f"  bc-label [provenance] — {int((qids >= 0).sum()):,}/{len(qids):,} quads "
            f"on {n_bc} patch(es), {int(bc_vmask.any(1).sum()):,} verts tagged")
    else:
        bc_vmask = np.zeros((len(verts), 0), dtype=bool)

    def _anchor_verts(verts):
        if anchors is None or not len(anchors):
            return None
        from scipy.spatial import cKDTree

        _, ai = cKDTree(verts).query(np.asarray(anchors, float))
        return np.unique(np.atleast_1d(ai).astype(int))

    def _box_pins(c, verts, av):
        """Per-axis Dirichlet pins from the taxonomy mask + load anchors:
        ``pinned`` (V,3) bool, ``target`` (V,3). Box faces pin one coordinate to a
        plane; anchors pin all three to the current (pre-fair) position."""
        v = len(verts)
        hi_box = origin + shape * spacing
        pinned = np.zeros((v, 3), dtype=bool)
        target = np.zeros((v, 3), dtype=float)
        for a in range(3):
            lo_m, hi_m = c.on_face[:, 2 * a], c.on_face[:, 2 * a + 1]
            pinned[:, a] = lo_m | hi_m
            target[lo_m, a] = origin[a]
            target[hi_m, a] = hi_box[a]
        if av is not None and len(av):
            pinned[av] = True
            target[av] = verts[av]
        return pinned, target

    def _fair(verts, faces, n, tag, mask=None):
        ts = time.perf_counter()
        c = classify(verts, origin, spacing, shape, anchors=anchors, face_mask=mask)
        if opts.fair_mode == "implicit":
            # primary path: minimize the fairing energy subject to box pins +
            # occupancy corridor (the only honest data from binary voxels).
            pinned, target = _box_pins(c, verts, _anchor_verts(verts))
            lo, hi = corridor.derive_bounds(
                verts, vox, origin, spacing, normal_voxels=opts.corridor_voxels)
            verts = fair.solve_implicit(
                verts, faces, pinned, target, lo, hi,
                order=opts.smooth_order, iters=opts.implicit_iters, reg=opts.data_weight,
            )
            mode = f"implicit/{opts.smooth_order} corr={opts.corridor_voxels}"
        elif opts.smooth_interp:
            # Taubin §4.2 (fast path): fair UNCONSTRAINED, then add the exact
            # smooth correction; anchors pin to their pre-fair position.
            xN = fair.constrained_taubin(
                verts, faces, fair.free_constraints(len(verts)),
                lamb=opts.lamb, mu=opts.mu, iters=n, weights=opts.weights, asymmetric=False,
            )
            av = _anchor_verts(verts)
            at = verts[av] if av is not None else None
            verts = fair.smooth_interpolate(
                xN, faces, c.on_face, origin=origin, spacing=spacing, shape=shape,
                anchor_verts=av, anchor_targets=at, lamb=opts.lamb, mu=opts.mu,
                kernel_iters=opts.smooth_kernel_iters,
            )
            mode = f"{opts.weights}, smooth-interp"
        else:
            verts = fair.constrained_taubin(
                verts, faces, c, lamb=opts.lamb, mu=opts.mu, iters=n,
                weights=opts.weights, asymmetric=opts.asymmetric,
            )
            mode = f"{opts.weights}, {'asym' if opts.asymmetric else 'sym'}"
        n_edge = int((c.tangent ** 2).sum(1).astype(bool).sum())
        log(f"  fair [{tag}] — verts={len(verts):,} faces={len(faces):,} "
            f"on_face={int((c.face_idx >= 0).sum()):,} on_edge={n_edge:,} "
            f"corner={int(c.fixed.sum()):,} [{mode}]  ({time.perf_counter() - ts:.2f}s)")
        return verts

    # 1. heavy smoothing at native resolution — the staircase is the Nyquist mode
    #    here (λ=0.5 matched), so this is where it is cheapest to remove.
    verts = _fair(verts, faces, opts.iters, "native", mask=vmask)

    sub = 0

    def _refine(verts, faces, vmask, bc_vmask):
        """One midpoint subdivision + fair. The box-face and BC masks ride the one
        subdivide AND together (so constraints stay exact), then split apart."""
        nonlocal sub
        ts = time.perf_counter()
        stack = np.hstack([vmask, bc_vmask])
        verts, faces, stack = fair.subdivide(verts, faces, stack)
        vmask, bc_vmask = stack[:, :6], stack[:, 6:]
        sub += 1
        log(f"  subdivide x1 -> verts={len(verts):,} faces={len(faces):,}  "
            f"({time.perf_counter() - ts:.2f}s)")
        verts = _fair(verts, faces, opts.iters, f"sub{sub}", mask=vmask)
        return verts, faces, vmask, bc_vmask

    # 2a. explicit subdivisions (schedule control) — run regardless of any target.
    for _ in range(opts.sub_levels):
        verts, faces, vmask, bc_vmask = _refine(verts, faces, vmask, bc_vmask)

    # 2b. target-driven refinement: keep subdividing toward the budget while under
    #     it, capped by max_sub_levels. (Skipped entirely when no target is set.)
    while (opts.target_faces is not None
           and len(fair.to_triangles(faces)) < opts.target_faces
           and sub < opts.max_sub_levels):
        verts, faces, vmask, bc_vmask = _refine(verts, faces, vmask, bc_vmask)

    # 3. decimate to the target budget, then a light polish (validation runs on
    #    this small mesh, not the dense intermediate — the speed fix). Decimation
    #    drops vertex parentage, so the box-face constraint falls back to geometric
    #    and the BC mask is carried over by nearest source from the labelled mesh.
    if opts.target_faces is not None and len(fair.to_triangles(faces)) > opts.target_faces:
        ts = time.perf_counter()
        pre = verts
        verts, faces = fair.decimate_to(verts, faces, opts.target_faces)
        if bc_vmask.shape[1]:
            from scipy.spatial import cKDTree

            _, idx = cKDTree(pre).query(verts)
            bc_vmask = bc_vmask[idx]
        log(f"  decimate -> faces={len(faces):,}  ({time.perf_counter() - ts:.2f}s)")
        verts = _fair(verts, faces, max(5, opts.iters // 5), "polish")

    report = validate(verts, faces, vox=vox, origin=origin, spacing=spacing,
                      shape=shape, anchors=anchors, log=log, on_fail=opts.on_fail)
    cell_arrays = None
    if use_bc:
        cell_arrays = face_bc_arrays(faces, bc_vmask, bc_patches)
    log(f"  done — {len(verts):,} verts, {len(faces):,} faces  "
        f"({time.perf_counter() - t0:.2f}s total)")
    return verts, faces, cell_arrays, report


def write_vtp(path, verts, faces, *, cell_arrays=None, load_patches=None):
    """Write a tagged PolyData: per-face cell_data + load anchors/forces in field_data."""
    import pyvista as pv

    faces = np.asarray(faces)
    k = faces.shape[1]
    cells = np.column_stack([np.full(len(faces), k), faces]).ravel()
    pd = pv.PolyData(np.asarray(verts, float), cells)
    for name, arr in (cell_arrays or {}).items():
        pd.cell_data[name] = np.asarray(arr)
    if load_patches:
        pd.field_data["load_anchors"] = np.asarray([p.anchor[:3] for p in load_patches], float).ravel()
        pd.field_data["load_forces"] = np.asarray([p.force[:3] for p in load_patches], float).ravel()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.save(str(path))
    return pd
