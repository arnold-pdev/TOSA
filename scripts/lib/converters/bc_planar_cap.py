"""
Planar BC cap contouring: voxel footprint → smooth 2D boundary → retessellated cap.

All contour methods share ``PlanarCapRequest`` / ``PlanarCapResult`` and register
via ``PLANAR_CAP_METHODS``. The pipeline stage calls ``build_planar_caps_mesh`` for
audit metrics; caps are not assembled onto the extracted skin mesh.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import trimesh
from shapely.geometry import Point, Polygon

from lib.converters.bc_patch import Patch
PlanarCapMethod = Callable[["PlanarCapRequest"], "PlanarCapResult"]


@dataclass(frozen=True)
class PlanarCapParams:
    """Method-tunable parameters (shared container; not all fields apply to every method)."""

    upsample_factor: int = 4
    blur_sigma_cells: float = 1.0
    containment_margin_cells: float = 0.0
    outward_buffer_cells: float = 0.25
    max_buffer_attempts: int = 4


@dataclass(frozen=True)
class PlanarCapRequest:
    """Common input for every planar-cap contour method."""

    patch: Patch
    origin: np.ndarray
    spacing: np.ndarray
    shape: np.ndarray
    params: PlanarCapParams = field(default_factory=PlanarCapParams)


@dataclass
class PlanarCapResult:
    """Common output from every planar-cap contour method."""

    method: str
    boundary_xy: np.ndarray
    vertices: np.ndarray
    faces: np.ndarray
    containment_ok: bool
    n_footprint_cells: int
    n_corners_violating: int
    meta: dict[str, float | int | str] = field(default_factory=dict)


def list_planar_cap_methods() -> list[str]:
    return sorted(PLANAR_CAP_METHODS)


def build_planar_cap(request: PlanarCapRequest, method: str) -> PlanarCapResult:
    key = method.strip().lower()
    if key not in PLANAR_CAP_METHODS:
        choices = ", ".join(list_planar_cap_methods())
        raise ValueError(f"Unknown bc_planar_cap method {method!r}. Choose: {choices}")
    return PLANAR_CAP_METHODS[key](request)


def build_planar_caps_mesh(
    patches: list[Patch],
    origin,
    spacing,
    shape,
    *,
    method: str,
    params: PlanarCapParams | None = None,
) -> tuple[trimesh.Trimesh | None, dict[str, float | int | str]]:
    """
    Build analytic planar BC cap mesh from voxel footprints.

    Returns ``(bc_mesh, audit)`` for QA / future shared-seam assembly.
    """
    params = params or PlanarCapParams()
    origin = np.asarray(origin, dtype=float)
    spacing = np.asarray(spacing, dtype=float)
    shape = np.asarray(shape, dtype=int)

    cap_parts: list[tuple[np.ndarray, np.ndarray]] = []
    audit: dict[str, float | int | str] = {"method": method}

    n_violating = 0
    n_caps = 0
    for pid, patch in enumerate(patches):
        if not patch.is_bc or not patch.faces:
            continue
        req = PlanarCapRequest(
            patch=patch,
            origin=origin,
            spacing=spacing,
            shape=shape,
            params=params,
        )
        result = build_planar_cap(req, method)
        if result.vertices.size == 0 or result.faces.size == 0:
            continue
        cap_parts.append((result.vertices, result.faces))
        n_caps += 1
        n_violating += result.n_corners_violating
        audit[f"patch_{pid}_containment_ok"] = int(result.containment_ok)
        for k, v in result.meta.items():
            audit[f"patch_{pid}_{k}"] = v

    if not cap_parts:
        audit["n_caps"] = 0
        return None, audit

    cap_verts: list[np.ndarray] = []
    cap_faces: list[np.ndarray] = []
    voff = 0
    for cap_v, cap_f in cap_parts:
        cap_verts.append(cap_v)
        cap_faces.append(cap_f + voff)
        voff += len(cap_v)
    bc_mesh = trimesh.Trimesh(
        vertices=np.vstack(cap_verts),
        faces=np.vstack(cap_faces),
        process=False,
    )

    audit["n_caps"] = n_caps
    audit["n_corners_violating"] = n_violating
    audit["containment_ok"] = int(n_violating == 0)
    return bc_mesh, audit


# ---------------------------------------------------------------------------
# Shared footprint / geometry helpers
# ---------------------------------------------------------------------------


def _inplane_axes(ndim: int, axis: int) -> tuple[int, int]:
    axes = [i for i in range(ndim) if i != axis]
    if len(axes) != 2:
        raise ValueError(f"Expected 3D patch, got in-plane axes {axes}")
    return int(axes[0]), int(axes[1])


def footprint_binary_mask(
    patch: Patch,
    shape: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    """
    Boolean mask of the patch footprint on the domain-face slab.

    When ``shape`` is given the mask spans the full in-plane domain extent
    ``[0, shape[a0]) × [0, shape[a1])`` so blur/contour cannot spill outside
    the design box. Returns ``(mask, (i0, j0), inplane_axes)``.
    """
    ndim = 3
    axis = patch.axis
    inplane = _inplane_axes(ndim, axis)
    a0, a1 = inplane
    if not patch.faces:
        return np.zeros((1, 1), dtype=bool), (0, 0), inplane

    if shape is not None:
        shape = np.asarray(shape, dtype=int)
        h, w = int(shape[a0]), int(shape[a1])
        i0, j0 = 0, 0
        mask = np.zeros((h, w), dtype=bool)
        for foot in patch.faces:
            i, j = int(foot[0]), int(foot[1])
            if 0 <= i < h and 0 <= j < w:
                mask[i, j] = True
        return mask, (i0, j0), inplane

    arr = np.asarray(patch.faces, dtype=int)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    i0, j0 = int(arr[:, 0].min()), int(arr[:, 1].min())
    i1, j1 = int(arr[:, 0].max()), int(arr[:, 1].max())
    mask = np.zeros((i1 - i0 + 1, j1 - j0 + 1), dtype=bool)
    for foot in patch.faces:
        mask[int(foot[0]) - i0, int(foot[1]) - j0] = True
    return mask, (i0, j0), inplane


def domain_face_polygon_xy(
    origin: np.ndarray,
    spacing: np.ndarray,
    shape: np.ndarray,
    inplane: tuple[int, int],
) -> Polygon:
    """Axis-aligned rectangle of the design-domain face in world in-plane coords."""
    origin = np.asarray(origin, dtype=float)
    spacing = np.asarray(spacing, dtype=float)
    shape = np.asarray(shape, dtype=int)
    a0, a1 = inplane
    u0 = float(origin[a0])
    u1 = float(origin[a0] + shape[a0] * spacing[a0])
    v0 = float(origin[a1])
    v1 = float(origin[a1] + shape[a1] * spacing[a1])
    return Polygon([(u0, v0), (u1, v0), (u1, v1), (u0, v1)])


def _footprint_cells_union_polygon(
    patch: Patch,
    origin: np.ndarray,
    spacing: np.ndarray,
    inplane: tuple[int, int],
) -> Polygon:
    """Union of footprint cell squares (stairstep evidence, no blur)."""
    from shapely.geometry import box
    from shapely.ops import unary_union

    if not patch.faces:
        return Polygon()
    a0, a1 = inplane
    cells = []
    for foot in patch.faces:
        i, j = int(foot[0]), int(foot[1])
        u0 = float(origin[a0] + i * spacing[a0])
        u1 = float(origin[a0] + (i + 1) * spacing[a0])
        v0 = float(origin[a1] + j * spacing[a1])
        v1 = float(origin[a1] + (j + 1) * spacing[a1])
        cells.append(box(u0, v0, u1, v1))
    geom = unary_union(cells)
    if geom.is_empty:
        return Polygon()
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)
    return geom if geom.geom_type == "Polygon" else Polygon()


def _constrain_boundary_to_domain(
    boundary_xy: np.ndarray,
    request: PlanarCapRequest,
    inplane: tuple[int, int],
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Clip smooth contour to the analytic domain-face rectangle; prune spikes."""
    meta: dict[str, float | int] = {}
    if boundary_xy.size == 0:
        return boundary_xy, meta

    domain = domain_face_polygon_xy(
        request.origin,
        request.spacing,
        request.shape,
        inplane,
    )
    smooth = _polygon_from_boundary(boundary_xy)
    clipped = smooth.intersection(domain)
    if clipped.is_empty:
        clipped = smooth
    elif clipped.geom_type == "MultiPolygon":
        clipped = max(clipped.geoms, key=lambda g: g.area)

    if clipped.is_empty or clipped.geom_type != "Polygon":
        coords = boundary_xy
    else:
        coords = np.asarray(clipped.exterior.coords, dtype=float)
        coords = _prune_boundary_spikes(coords, request.spacing)

    meta["domain_clipped"] = 1
    return coords, meta


def _prune_boundary_spikes(
    boundary_xy: np.ndarray,
    spacing: np.ndarray,
    *,
    min_turn_deg: float = 25.0,
    max_spike_len: float | None = None,
) -> np.ndarray:
    """Drop short, acute needle vertices from a closed contour."""
    xy = np.asarray(boundary_xy, dtype=float)
    if len(xy) < 5:
        return xy
    if len(xy) > 1 and np.allclose(xy[0], xy[-1]):
        xy = xy[:-1]
    if max_spike_len is None:
        max_spike_len = float(np.min(spacing)) * 1.25
    min_turn = np.radians(min_turn_deg)
    changed = True
    pts = xy.tolist()
    while changed and len(pts) >= 4:
        changed = False
        n = len(pts)
        keep = []
        for i in range(n):
            prev = np.asarray(pts[(i - 1) % n], dtype=float)
            cur = np.asarray(pts[i], dtype=float)
            nxt = np.asarray(pts[(i + 1) % n], dtype=float)
            e0 = cur - prev
            e1 = nxt - cur
            l0 = float(np.linalg.norm(e0))
            l1 = float(np.linalg.norm(e1))
            if l0 < 1e-12 or l1 < 1e-12:
                changed = True
                continue
            cosang = float(np.dot(e0, e1) / (l0 * l1))
            cosang = np.clip(cosang, -1.0, 1.0)
            turn = np.arccos(cosang)
            if turn < min_turn and (l0 < max_spike_len or l1 < max_spike_len):
                changed = True
                continue
            keep.append(cur)
        if len(keep) < 3:
            break
        pts = keep
    out = np.asarray(pts, dtype=float)
    if out.size and not np.allclose(out[0], out[-1]):
        out = np.vstack([out, out[:1]])
    return out


def _mask_to_world_xy(
    rows: np.ndarray,
    cols: np.ndarray,
    origin: np.ndarray,
    spacing: np.ndarray,
    inplane: tuple[int, int],
    offset: tuple[int, int],
) -> np.ndarray:
    i0, j0 = offset
    a0, a1 = inplane
    u = origin[a0] + (rows + i0) * spacing[a0]
    v = origin[a1] + (cols + j0) * spacing[a1]
    return np.column_stack([u, v])


def footprint_cell_corner_xy(
    patch: Patch,
    origin: np.ndarray,
    spacing: np.ndarray,
) -> np.ndarray:
    """All footprint cell corners as (n, 2) world in-plane coordinates."""
    if not patch.faces:
        return np.zeros((0, 2), dtype=float)
    ndim = 3
    inplane = _inplane_axes(ndim, patch.axis)
    a0, a1 = inplane
    corners: list[list[float]] = []
    for foot in patch.faces:
        for di in (0, 1):
            for dj in (0, 1):
                corners.append(
                    [
                        float(origin[a0] + (int(foot[0]) + di) * spacing[a0]),
                        float(origin[a1] + (int(foot[1]) + dj) * spacing[a1]),
                    ],
                )
    return np.asarray(corners, dtype=float)


def containment_audit(
    boundary_xy: np.ndarray,
    corner_xy: np.ndarray,
    *,
    margin: float = 0.0,
) -> tuple[bool, int]:
    """True when every footprint corner lies inside the boundary polygon."""
    if boundary_xy.size == 0 or corner_xy.size == 0:
        return True, 0
    poly = _polygon_from_boundary(boundary_xy)
    if margin > 0:
        poly = poly.buffer(margin)
    violating = 0
    for pt in corner_xy:
        if not poly.covers(Point(float(pt[0]), float(pt[1]))):
            violating += 1
    return violating == 0, violating


def _polygon_from_boundary(boundary_xy: np.ndarray) -> Polygon:
    xy = np.asarray(boundary_xy, dtype=float)
    if len(xy) > 1 and np.allclose(xy[0], xy[-1]):
        xy = xy[:-1]
    poly = Polygon(xy)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def _lift_cap_to_3d(
    verts2d: np.ndarray,
    faces: np.ndarray,
    patch: Patch,
    inplane: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    verts3d = np.zeros((len(verts2d), 3), dtype=float)
    a0, a1 = inplane
    verts3d[:, a0] = verts2d[:, 0]
    verts3d[:, a1] = verts2d[:, 1]
    verts3d[:, patch.axis] = patch.plane_coord
    return verts3d, np.asarray(faces, dtype=np.int64)


def triangulate_cap_on_plane(
    boundary_xy: np.ndarray,
    patch: Patch,
    origin: np.ndarray,
    spacing: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Delaunay cap triangulation on ``patch`` plane from a closed 2D boundary."""
    inplane = _inplane_axes(3, patch.axis)
    return _triangulate_planar_region(
        boundary_xy,
        patch,
        origin,
        spacing,
        inplane,
    )


def _triangulate_planar_region(
    boundary_xy: np.ndarray,
    patch: Patch,
    origin: np.ndarray,
    spacing: np.ndarray,
    inplane: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Delaunay triangulation of boundary + interior cell centers inside the polygon."""
    from scipy.spatial import Delaunay

    poly = _polygon_from_boundary(boundary_xy)
    a0, a1 = inplane
    interior: list[list[float]] = []
    for foot in patch.faces:
        cx = float(origin[a0] + (int(foot[0]) + 0.5) * spacing[a0])
        cy = float(origin[a1] + (int(foot[1]) + 0.5) * spacing[a1])
        if poly.contains(Point(cx, cy)):
            interior.append([cx, cy])

    bnd = np.asarray(boundary_xy, dtype=float)
    if len(bnd) > 1 and np.allclose(bnd[0], bnd[-1]):
        bnd = bnd[:-1]
    if bnd.size == 0:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=np.int64)

    pts2d = np.vstack([bnd, np.asarray(interior, dtype=float)])
    tri = Delaunay(pts2d)
    faces: list[list[int]] = []
    for simplex in tri.simplices:
        cent = pts2d[simplex].mean(axis=0)
        if poly.contains(Point(float(cent[0]), float(cent[1]))):
            faces.append([int(s) for s in simplex])
    if not faces:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=np.int64)
    return _lift_cap_to_3d(pts2d, np.asarray(faces, dtype=np.int64), patch, inplane)


def _ensure_containment(
    boundary_xy: np.ndarray,
    corner_xy: np.ndarray,
    params: PlanarCapParams,
    spacing: np.ndarray,
    *,
    domain: Polygon | None = None,
) -> tuple[np.ndarray, bool, int]:
    ok, n_bad = containment_audit(boundary_xy, corner_xy)
    if ok:
        return boundary_xy, True, 0
    poly = _polygon_from_boundary(boundary_xy)
    buf = max(float(spacing.min()) * params.outward_buffer_cells, 1e-9)
    for _ in range(params.max_buffer_attempts):
        poly = poly.buffer(buf)
        if domain is not None:
            poly = poly.intersection(domain)
            if poly.is_empty:
                break
            if poly.geom_type == "MultiPolygon":
                poly = max(poly.geoms, key=lambda g: g.area)
        coords = np.asarray(poly.exterior.coords, dtype=float)
        ok, n_bad = containment_audit(coords, corner_xy)
        if ok:
            return coords, True, 0
    if domain is not None and not poly.is_empty:
        clipped = poly.intersection(domain)
        if not clipped.is_empty:
            if clipped.geom_type == "MultiPolygon":
                clipped = max(clipped.geoms, key=lambda g: g.area)
            if clipped.geom_type == "Polygon":
                poly = clipped
    coords = np.asarray(poly.exterior.coords, dtype=float)
    _, n_bad = containment_audit(coords, corner_xy)
    return coords, False, n_bad


def _result_from_boundary(
    method: str,
    boundary_xy: np.ndarray,
    request: PlanarCapRequest,
    inplane: tuple[int, int],
    *,
    extra_meta: dict | None = None,
) -> PlanarCapResult:
    patch = request.patch
    domain = domain_face_polygon_xy(
        request.origin,
        request.spacing,
        request.shape,
        inplane,
    )
    boundary_xy, dom_meta = _constrain_boundary_to_domain(boundary_xy, request, inplane)
    corner_xy = footprint_cell_corner_xy(patch, request.origin, request.spacing)
    boundary_xy, ok, n_bad = _ensure_containment(
        boundary_xy,
        corner_xy,
        request.params,
        request.spacing,
        domain=domain,
    )
    verts, faces = _triangulate_planar_region(
        boundary_xy,
        patch,
        request.origin,
        request.spacing,
        inplane,
    )
    meta = dict(extra_meta or {})
    meta.update(dom_meta)
    meta["n_boundary_verts"] = len(boundary_xy)
    meta["n_cap_tris"] = len(faces)
    return PlanarCapResult(
        method=method,
        boundary_xy=boundary_xy,
        vertices=verts,
        faces=faces,
        containment_ok=ok,
        n_footprint_cells=len(patch.faces),
        n_corners_violating=n_bad,
        meta=meta,
    )


# ---------------------------------------------------------------------------
# Contour methods
# ---------------------------------------------------------------------------


def _method_native_contour(request: PlanarCapRequest) -> PlanarCapResult:
    """Baseline: marching squares at native resolution (stairstep boundary)."""
    from skimage import measure

    mask, offset, inplane = footprint_binary_mask(request.patch, request.shape)
    contours = measure.find_contours(mask.astype(float), 0.5)
    if not contours:
        return PlanarCapResult(
            method="native_contour",
            boundary_xy=np.zeros((0, 2)),
            vertices=np.zeros((0, 3)),
            faces=np.zeros((0, 3), dtype=np.int64),
            containment_ok=True,
            n_footprint_cells=len(request.patch.faces),
            n_corners_violating=0,
            meta={"note": "empty_contour"},
        )
    contour = max(contours, key=len)
    rows = contour[:, 0]
    cols = contour[:, 1]
    boundary_xy = _mask_to_world_xy(
        rows,
        cols,
        request.origin,
        request.spacing,
        inplane,
        offset,
    )
    return _result_from_boundary(
        "native_contour",
        boundary_xy,
        request,
        inplane,
        extra_meta={"upsample_factor": 1},
    )


def _method_upsample_blur(request: PlanarCapRequest) -> PlanarCapResult:
    """Upsample footprint mask, Gaussian blur, contour at 0.5 (primary smooth cap)."""
    from scipy import ndimage
    from skimage import measure

    params = request.params
    mask, offset, inplane = footprint_binary_mask(request.patch, request.shape)
    factor = max(int(params.upsample_factor), 1)
    up = mask.astype(float)
    if factor > 1:
        up = ndimage.zoom(up, zoom=factor, order=0)
    sigma = float(params.blur_sigma_cells) * factor
    if sigma > 0:
        up = ndimage.gaussian_filter(up, sigma=sigma)
    # Keep iso-contour inside domain: work on full slab, no exterior padding.
    contours = measure.find_contours(up, 0.5)
    if not contours:
        return _method_native_contour(request)
    contour = max(contours, key=len)
    rows = contour[:, 0] / factor
    cols = contour[:, 1] / factor
    boundary_xy = _mask_to_world_xy(
        rows,
        cols,
        request.origin,
        request.spacing,
        inplane,
        offset,
    )
    return _result_from_boundary(
        "upsample_blur",
        boundary_xy,
        request,
        inplane,
        extra_meta={
            "upsample_factor": factor,
            "blur_sigma_cells": params.blur_sigma_cells,
        },
    )


PLANAR_CAP_METHODS: dict[str, PlanarCapMethod] = {
    "upsample_blur": _method_upsample_blur,
    "native_contour": _method_native_contour,
}
