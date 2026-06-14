"""
ARCHIVED v1 pipeline (Taubin + geodesic regrow). Preserved for reference and regression.

Active entry point: ``lib.converters.voxel2surf`` (field-refinement v2 pipeline).

voxel2surf v1: build the diagnostic surface directly from the voxel array.

One pipeline, voxel array as the single source of truth:

    binary voxels
      -> signed distance field            (scipy EDT)
      -> padded SDF + marching cubes       (lewiner, no degenerates)
      -> trimesh cleanup + pymeshfix       (joincomp repair)
      -> voxel BC footprint transfer       (seeds + jaggedness metric only)
      -> masked Taubin smooth              (BC patch interiors frozen; edges round)
      -> geodesic patch regrow             (smooth silhouette on smoothed mesh)
      -> flatten BC vertices to plane      (θ=0 mounting faces for export / FEA)
      -> load surface check                (NITO anchors must lie on the surface)

BC patch *interior* triangles (all neighbors same patch) are frozen before Taubin so
mounting faces stay flat; patch boundary bands and the free boundary still smooth.
Loads are not stamped as surface patches — apply them from ``loads.npy`` at FEA time.
The pipeline only checks that each load lies within tolerance of the final surface.

Export (VTK only — canonical surface for meshing + FEA):

    save_surface(result, path)              # .vtp (default) or .vtk
    cell_data: patch_id, facet_marker, fix_x/y/z
    FieldData: patch_flags, patch_axis, patch_side

CLI:

    python scripts/lib/converters/voxel2surf.py --index 119 --data-dir nito/Data/3D
    python scripts/lib/converters/voxel2surf.py --index 119 --log-dir logs -q
    # writes public/vtp/119.vtp by default; use -q/--quiet to suppress stdout
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

_SCRIPTS = Path(__file__).resolve().parents[2]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import numpy as np
import scipy.sparse as sp
from scipy import ndimage
from skimage import measure
import trimesh

from lib.volume_check import (
    domain_volume,
    format_mesh_stage_lines,
    format_volume_fraction_lines,
    volume_fraction_voxels,
)

if TYPE_CHECKING:
    from lib.converters.bc_patch import BCSpec, Patch

LogFn = Callable[[str], None]


def _noop_log(_msg: str) -> None:
    pass


class PipelineLogger:
    """Capture pipeline messages; optionally echo to stdout and write to a file."""

    def __init__(self, *, echo_stdout: bool, log_path: Path | None) -> None:
        self._echo = echo_stdout
        self._path = log_path.expanduser().resolve() if log_path is not None else None
        self._lines: list[str] = []

    def __call__(self, msg: str) -> None:
        self._lines.append(msg)
        if self._echo:
            print(msg, flush=True)

    def save(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text("\n".join(self._lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# 1. voxels -> signed distance field
# --------------------------------------------------------------------------- #
def signed_distance_field(vox: np.ndarray, spacing) -> np.ndarray:
    """Signed distance: negative inside the solid, positive in void, ~0 on dOmega."""
    spacing = np.asarray(spacing, float)
    solid = vox > 0
    d_in = ndimage.distance_transform_edt(solid, sampling=spacing)
    d_out = ndimage.distance_transform_edt(~solid, sampling=spacing)
    return d_out - d_in


# --------------------------------------------------------------------------- #
# 2. SDF -> watertight outward-oriented surface mesh
# --------------------------------------------------------------------------- #
MC_PAD_VOXELS = 1


def _pad_sdf_for_marching_cubes(
    sdf: np.ndarray,
    spacing: np.ndarray,
    *,
    pad: int = MC_PAD_VOXELS,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad SDF with exterior void so MC does not open at the grid boundary."""
    spacing = np.asarray(spacing, dtype=float)
    sdf = np.asarray(sdf, dtype=float)
    if pad <= 0:
        return sdf, np.zeros_like(spacing)
    void_val = float(np.max(sdf[sdf > 0])) if np.any(sdf > 0) else 1.0
    void_val = max(void_val, float(np.max(spacing)))
    padded = np.pad(sdf, pad, mode="constant", constant_values=void_val)
    return padded, -pad * spacing


def _pre_meshfix_cleanup(mesh: trimesh.Trimesh) -> int:
    """Remove degenerate/duplicate faces and merge vertices before MeshFix."""
    before = len(mesh.faces)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.merge_vertices()
    trimesh.repair.fix_normals(mesh)
    return before - len(mesh.faces)


def extract_surface(sdf: np.ndarray, origin, spacing,
                    presmooth_sigma: float = 0.0,
                    repair: bool = True,
                    *,
                    domain_vol: float | None = None,
                    vf_voxels: float | None = None,
                    log: LogFn | None = None) -> trimesh.Trimesh:
    origin = np.asarray(origin, float)
    spacing = np.asarray(spacing, float)
    field = np.asarray(sdf, dtype=float)
    if presmooth_sigma > 0:
        # NOTE: uniform field smoothing -- bypasses the BC mask. Use sparingly.
        field = ndimage.gaussian_filter(field, sigma=presmooth_sigma)

    field, origin_shift = _pad_sdf_for_marching_cubes(field, spacing)
    mc_origin = origin + origin_shift

    verts, faces, _normals, _vals = measure.marching_cubes(
        field,
        level=0.0,
        spacing=tuple(spacing),
        method="lewiner",
        allow_degenerate=False,
        gradient_direction="ascent",
    )
    verts = verts + mc_origin
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    removed = _pre_meshfix_cleanup(mesh)

    if log is not None and domain_vol is not None:
        for line in format_mesh_stage_lines(
            mesh,
            domain_vol,
            "marching_cubes",
            vf_reference=vf_voxels,
        ):
            log(f"        {line.strip()}")
        if removed:
            log(f"        pre_meshfix cleanup removed {removed:,} degenerate/duplicate faces")

    if repair:
        try:
            import pymeshfix
            v_in = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
            f_in = np.ascontiguousarray(mesh.faces, dtype=np.int32)
            v, f = pymeshfix.clean_from_arrays(
                v_in,
                f_in,
                joincomp=True,
                verbose=True,
            )
            mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
            trimesh.repair.fix_normals(mesh)
            if log is not None and domain_vol is not None:
                for line in format_mesh_stage_lines(
                    mesh,
                    domain_vol,
                    "pymeshfix",
                    vf_reference=vf_voxels,
                ):
                    log(f"        {line.strip()}")
        except ImportError:
            if log is not None:
                log("        pymeshfix not installed — skipping surface repair")

    return mesh


# --------------------------------------------------------------------------- #
# 3. transfer voxel BC patches onto mesh triangles (exact cell-membership)
# --------------------------------------------------------------------------- #
def transfer_patches(mesh: trimesh.Trimesh, patches: list[Patch],
                     origin, spacing, vox_shape,
                     band_cells: float = 1.0, cos_tol: float = 0.5
                     ) -> np.ndarray:
    """Per-triangle patch id (index into `patches`), or -1 for the free boundary."""
    origin = np.asarray(origin, float)
    spacing = np.asarray(spacing, float)
    vox_shape = np.asarray(vox_shape, dtype=int)
    ndim = int(vox_shape.size)
    centroids = mesh.triangles_center
    fnormals = mesh.face_normals
    cells = np.clip(np.floor((centroids - origin) / spacing).astype(int),
                    0, vox_shape - 1)

    labels = np.full(len(centroids), -1, dtype=int)
    for pid, P in enumerate(patches):
        if not P.is_bc:
            continue
        a = P.axis
        inplane = [i for i in range(ndim) if i != a]
        foot = {tuple(f) for f in P.faces}
        band = band_cells * spacing[a]
        # min faces sit just above plane_coord; max faces sit one cell below it.
        ref = (
            P.plane_coord
            if P.side == "min"
            else P.plane_coord - spacing[a]
        )
        near = np.abs(centroids[:, a] - ref) < band
        aligned = fnormals @ np.asarray(P.normal[:ndim]) > cos_tol
        in_foot = np.fromiter(
            (tuple(cells[i, j] for j in inplane) in foot
             for i in range(len(cells))), dtype=bool, count=len(cells))
        member = near & aligned & in_foot & (labels < 0)
        labels[member] = pid
    return labels


# --------------------------------------------------------------------------- #
# 4. per-vertex freeze weight (1 = frozen, 0 = free) with graded transition
# --------------------------------------------------------------------------- #
def _face_neighbors(mesh: trimesh.Trimesh) -> list[list[int]]:
    """Face index → edge-adjacent face indices."""
    nbr: list[list[int]] = [[] for _ in range(len(mesh.faces))]
    for a, b in mesh.face_adjacency:
        nbr[a].append(b)
        nbr[b].append(a)
    return nbr


def _interior_bc_faces(tri_labels: np.ndarray, face_neighbors: list[list[int]]) -> np.ndarray:
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
    """
    Vertices whose incident BC triangles are all interior (deep patch core).

    Rim vertices on the voxel stairstep are excluded so Taubin can round them.
    """
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


def taubin_freeze_weights_interior(
    mesh: trimesh.Trimesh,
    tri_labels: np.ndarray,
    n_bc_patches: int,
    *,
    transition_rings: int = 0,
) -> tuple[np.ndarray, int, int]:
    """
    Per-vertex Taubin weights (1 = frozen) from interior BC triangles only.

    Returns (weights, n_interior_faces, n_frozen_vertices).
    """
    interior = _interior_bc_faces(tri_labels, _face_neighbors(mesh))
    vmap = _interior_bc_vertex_map(mesh, tri_labels, n_bc_patches)
    weights = freeze_weights(mesh, vmap, transition_rings=transition_rings)
    n_frozen = int(np.count_nonzero(weights >= 1.0 - 1e-12))
    return weights, int(np.count_nonzero(interior)), n_frozen


def _vertex_patch_map(mesh, tri_labels, n_patches, min_frac=0.5):
    """Vertex -> patch id if a majority of its incident triangles share it."""
    vmap = np.full(len(mesh.vertices), -1, dtype=int)
    vf = mesh.vertex_faces  # padded with -1
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


def freeze_weights(mesh, vmap, transition_rings: int = 3) -> np.ndarray:
    """1 on patch vertices; ramp 1->0 over `transition_rings` into the free side."""
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


# --------------------------------------------------------------------------- #
# 5. flatten BC vertices onto their exact patch plane
# --------------------------------------------------------------------------- #
def flatten_to_planes(mesh, vmap, patches) -> None:
    for pid, P in enumerate(patches):
        if not P.is_bc:
            continue
        vids = np.where(vmap == pid)[0]
        if vids.size:
            mesh.vertices[vids, P.axis] = P.plane_coord


# --------------------------------------------------------------------------- #
# 6. masked + graded Taubin smoothing
# --------------------------------------------------------------------------- #
def _uniform_laplacian(mesh) -> sp.csr_matrix:
    """Row-normalized adjacency L: (L @ v) is the mean of each vertex's neighbors."""
    n = len(mesh.vertices)
    e = mesh.edges_unique
    rows = np.concatenate([e[:, 0], e[:, 1]])
    cols = np.concatenate([e[:, 1], e[:, 0]])
    A = sp.coo_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n)).tocsr()
    deg = np.maximum(np.asarray(A.sum(1)).ravel(), 1.0)
    return sp.diags(1.0 / deg) @ A


def masked_taubin(mesh, weights, lamb=0.5, nu=0.53, iterations=10) -> None:
    """Taubin lambda|mu flow, each vertex update scaled by `weights` (0 = frozen)."""
    L = _uniform_laplacian(mesh)
    v = mesh.vertices.copy()
    w = weights[:, None]
    free = (1.0 - w)  # frozen vertices (w=1) get 0 update; free (w=0) get full
    for _ in range(iterations):
        for factor in (lamb, -nu):
            v = v + factor * (L.dot(v) - v) * free
    mesh.vertices = v


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
@dataclass
class SurfaceResult:
    mesh: trimesh.Trimesh
    patches: list  # list[Patch] when BC patches are built
    tri_labels: np.ndarray   # per-triangle patch id (-1 = free boundary)
    freeze: np.ndarray       # per-vertex patch membership (1 on BC verts, for export)


def voxel_to_surface(vox: np.ndarray,
                     bcspecs: list | None = None,
                     *,
                     bc_rows: np.ndarray | None = None,
                     load_rows: np.ndarray | None = None,
                     grid_shape: np.ndarray | None = None,
                     origin=(0.0, 0.0, 0.0),
                     spacing=(1.0, 1.0, 1.0),
                     presmooth_sigma: float = 0.0,
                     repair: bool = True,
                     taubin_lambda: float = 0.5,
                     taubin_nu: float = 0.53,
                     taubin_iters: int = 10,
                     taubin_freeze_rings: int = 0,
                     regrow_band_cells: float = 0.5,
                     regrow_cos_tol: float = 0.92,
                     regrow_cos_tol_grow: float = 0.85,
                     regrow_band_grow_factor: float = 1.25,
                     thin_min_extent: int = 2,
                     load_surface_tol_cells: float = 1.0,
                     check_loads: bool = True,
                     log: LogFn | None = None) -> SurfaceResult:
    spacing = np.asarray(spacing, float)
    origin = np.asarray(origin, float)
    specs = list(bcspecs or [])
    log = _noop_log if log is None else log
    domain_vol = domain_volume(
        grid_shape if grid_shape is not None else vox.shape,
        spacing,
    )
    use_regrow = bool(specs and bc_rows is not None)
    if not specs:
        n_steps, taubin_step = 4, 4
    elif use_regrow:
        n_steps, taubin_step = 6, 5
    else:
        n_steps, taubin_step = 5, 5

    log(f"  [1/{n_steps}] signed distance field (scipy EDT) …")
    sdf = signed_distance_field(vox, spacing)

    log(
        f"  [2/{n_steps}] marching cubes"
        + (f" + pymeshfix repair …" if repair else " …"),
    )
    mesh = extract_surface(
        sdf,
        origin,
        spacing,
        presmooth_sigma,
        repair,
        domain_vol=domain_vol,
        vf_voxels=volume_fraction_voxels(vox) if domain_vol is not None else None,
        log=log,
    )
    log(
        f"        {len(mesh.vertices):,} vertices, {len(mesh.faces):,} triangles",
    )

    bc_patches: list = []
    if specs:
        from lib.converters.bc_patch import _exposed_faces, build_patch

        log(f"  [3/{n_steps}] BC patches from NITO ({len(specs)} face spec(s)) …")
        exposed = _exposed_faces(vox)
        for spec in specs:
            patch = build_patch(
                vox,
                spec,
                origin,
                spacing,
                exposed=exposed,
                bc_rows=bc_rows,
                shape=grid_shape,
            )
            if patch.faces:
                bc_patches.append(patch)
                log(
                    f"        BC patch {len(bc_patches) - 1}: axis={patch.axis} "
                    f"side={patch.side}, {len(patch.faces)} voxel face cells",
                )
    else:
        log(f"  [3/{n_steps}] BC patches skipped")

    patches = bc_patches

    tri_labels_voxel = np.full(len(mesh.faces), -1, dtype=np.int32)
    if bc_patches:
        log(f"  [4/{n_steps}] voxel BC transfer (seeds + jaggedness metric) …")
        tri_labels_voxel = transfer_patches(
            mesh, bc_patches, origin, spacing, vox.shape
        )
        n_labeled_voxel = int(np.count_nonzero(tri_labels_voxel >= 0))
        log(
            f"        {n_labeled_voxel:,} BC triangles (voxel footprint), "
            f"{len(tri_labels_voxel) - n_labeled_voxel:,} free-boundary triangles",
        )

    log(
        f"  [{taubin_step}/{n_steps}] masked Taubin smooth "
        f"({taubin_iters} iters, λ={taubin_lambda}, μ={taubin_nu}) …",
    )
    if bc_patches and np.any(tri_labels_voxel >= 0):
        smooth_weights, n_interior, n_frozen = taubin_freeze_weights_interior(
            mesh,
            tri_labels_voxel,
            len(bc_patches),
            transition_rings=taubin_freeze_rings,
        )
        log(
            f"        freeze BC interiors: {n_interior:,} interior triangles, "
            f"{n_frozen:,} frozen vertices"
            + (
                f", {taubin_freeze_rings} transition ring(s)"
                if taubin_freeze_rings > 0
                else ""
            ),
        )
    else:
        smooth_weights = np.zeros(len(mesh.vertices), dtype=float)
        log("        no BC interior freeze (geometry-only or unlabeled)")
    masked_taubin(mesh, smooth_weights, taubin_lambda, taubin_nu, taubin_iters)

    if use_regrow:
        from lib.converters.patch_regrow import patch_boundary_length, regrow_patches_on_mesh

        log(f"  [6/{n_steps}] geodesic BC patch regrow …")
        voxel_mask = tri_labels_voxel >= 0
        boundary_before = patch_boundary_length(mesh, voxel_mask)
        tri_labels = regrow_patches_on_mesh(
            mesh,
            bc_patches,
            bc_rows,
            grid_shape if grid_shape is not None else vox.shape,
            spacing,
            origin=origin,
            tri_labels_voxel=tri_labels_voxel,
            band_cells=regrow_band_cells,
            cos_tol=regrow_cos_tol,
            cos_tol_grow=regrow_cos_tol_grow,
            band_grow_factor=regrow_band_grow_factor,
            thin_min_extent=thin_min_extent,
            log=log,
        )
        boundary_after = patch_boundary_length(mesh, tri_labels >= 0)
        n_labeled = int(np.count_nonzero(tri_labels >= 0))
        log(
            f"        {n_labeled:,} labeled triangles (final), "
            f"BC boundary length {boundary_before} → {boundary_after}",
        )
        log("        flatten BC vertices to patch planes …")
    elif bc_patches:
        tri_labels = tri_labels_voxel.copy()
        log(f"  [{n_steps}/{n_steps}] flatten BC vertices to patch planes …")
    else:
        tri_labels = np.full(len(mesh.faces), -1, dtype=np.int32)

    if bc_patches:
        vmap = _vertex_patch_map(mesh, tri_labels, len(patches))
        flatten_to_planes(mesh, vmap, patches)
        freeze = (vmap >= 0).astype(float)
        log(f"        {int(np.count_nonzero(vmap >= 0)):,} patch vertices (BC)")
    else:
        freeze = np.zeros(len(mesh.vertices), dtype=float)

    if (
        check_loads
        and load_rows is not None
        and np.atleast_2d(load_rows).size > 0
    ):
        from lib.nito_physics import require_loads_on_surface

        log(
            f"  load surface check "
            f"(tolerance={load_surface_tol_cells} cells) …",
        )
        checks = require_loads_on_surface(
            mesh,
            load_rows,
            grid_shape if grid_shape is not None else vox.shape,
            spacing=spacing,
            max_distance_cells=load_surface_tol_cells,
        )
        for c in checks:
            a = c.nito_anchor
            log(
                f"        load {c.index}: distance={c.distance:.4g} "
                f"nito=({a[0]:.3g}, {a[1]:.3g}, {a[2]:.3g})"
            )

    return SurfaceResult(mesh=mesh, patches=patches,
                         tri_labels=tri_labels, freeze=freeze)


# --------------------------------------------------------------------------- #
# export (VTK PolyData — canonical surface representation)
# --------------------------------------------------------------------------- #
class SurfaceOutputFormat(str, Enum):
    """On-disk surface format (VTK PolyData only)."""

    VTP = "vtp"
    VTK = "vtk"


def save_surface(
    result: SurfaceResult,
    path: Path,
    *,
    fmt: SurfaceOutputFormat | str | None = None,
) -> Path:
    """Write a tagged VTK surface (.vtp default) for Gmsh, TetGen, and FEA."""
    from lib.meshing.surface_tags import write_surface_vtp

    fmt_str = None if fmt is None else str(fmt).lower().lstrip(".")
    return write_surface_vtp(result, path, fmt=fmt_str)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract a diagnostic surface from a voxel topology."
    )
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--test", action="store_true")
    p.add_argument(
        "--format",
        choices=tuple(f.value for f in SurfaceOutputFormat),
        default=None,
        help="VTK format (default: infer from --output suffix, else vtp)",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output path (default: public/vtp/<index>.vtp)",
    )
    p.add_argument(
        "--density-cutoff",
        type=float,
        default=0.5,
        help="Voxels with density >= cutoff are treated as solid",
    )
    p.add_argument(
        "--no-bc",
        action="store_true",
        help="Skip NITO BC patch transfer (geometry-only surface)",
    )
    p.add_argument("--presmooth-sigma", type=float, default=0.0)
    p.add_argument("--no-repair", action="store_true")
    p.add_argument("--taubin-iters", type=int, default=10)
    p.add_argument(
        "--taubin-freeze-rings",
        type=int,
        default=0,
        help="Graded freeze ramp (rings) outward from frozen BC interiors (default 0)",
    )
    p.add_argument("--regrow-band-cells", type=float, default=0.5)
    p.add_argument("--regrow-cos-tol", type=float, default=0.92)
    p.add_argument(
        "--load-surface-tol-cells",
        type=float,
        default=1.0,
        help="Max distance (in cells) from each NITO load to the final surface (default 1)",
    )
    p.add_argument(
        "--no-load-check",
        action="store_true",
        help="Skip NITO load-on-surface validation",
    )
    p.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Write full pipeline log to this path (verbose even with -q)",
    )
    p.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Write pipeline log to <dir>/<index>.log (verbose even with -q)",
    )
    p.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress step-by-step progress on stdout (see --log-file / --log-dir)",
    )
    return p.parse_args(argv)


def _resolve_log_path(args: argparse.Namespace) -> Path | None:
    if args.log_file is not None:
        return args.log_file.expanduser()
    if args.log_dir is not None:
        return args.log_dir.expanduser() / f"{args.index}.log"
    return None


def _vox_from_rho(rho: np.ndarray, shape: np.ndarray, cutoff: float) -> np.ndarray:
    vox = np.asarray(rho, dtype=float).reshape(tuple(int(x) for x in shape), order="C")
    return (vox >= cutoff).astype(np.uint8)


def _load_bcspecs(bc: np.ndarray, shape: np.ndarray, *, use_bc: bool) -> list:
    if not use_bc:
        return []
    from lib.converters.bc_patch import bcspecs_from_nito

    return bcspecs_from_nito(bc, shape)


def main(argv: list[str] | None = None) -> None:
    from lib.nito_io import load_nito_arrays, load_sample, resolve_data_dir
    from lib.surface_io import DEFAULT_SURFACE_DIR, vtp_path

    args = _parse_args(argv)
    verbose = not args.quiet
    log_path = _resolve_log_path(args)
    logger = PipelineLogger(echo_stdout=verbose, log_path=log_path)
    data_dir = resolve_data_dir(args.data_dir, test=args.test)
    shape, rho, bc, load, _vf = load_sample(load_nito_arrays(data_dir), args.index)

    out_path = args.output or vtp_path(DEFAULT_SURFACE_DIR, args.index)

    vox = _vox_from_rho(rho, shape, args.density_cutoff)
    n_solid = int(np.count_nonzero(vox))
    spacing = np.ones(int(shape.size), dtype=float)
    use_bc = not args.no_bc

    logger(f"voxel2surf  index={args.index}")
    logger(f"  data_dir={data_dir}")
    logger(f"  shape={tuple(int(x) for x in shape)}  solid_voxels={n_solid:,}")
    logger(f"  density_cutoff={args.density_cutoff}  bc_patches={'on' if use_bc else 'off'}")
    if not args.no_load_check:
        logger(f"  load_surface_check=tol {args.load_surface_tol_cells} cells")
    logger(f"  output={out_path.resolve()}")
    if log_path is not None:
        logger(f"  log={log_path.resolve()}")

    result = voxel_to_surface(
        vox,
        _load_bcspecs(bc, shape, use_bc=use_bc),
        bc_rows=bc if use_bc else None,
        load_rows=None if args.no_load_check else load,
        grid_shape=shape,
        spacing=spacing,
        presmooth_sigma=args.presmooth_sigma,
        repair=not args.no_repair,
        taubin_iters=args.taubin_iters,
        taubin_freeze_rings=args.taubin_freeze_rings,
        regrow_band_cells=args.regrow_band_cells,
        regrow_cos_tol=args.regrow_cos_tol,
        load_surface_tol_cells=args.load_surface_tol_cells,
        check_loads=not args.no_load_check,
        log=logger,
    )

    logger("  write tagged VTK …")
    out = save_surface(
        result,
        out_path,
        fmt=args.format,
    )

    n_patches = len(result.patches)
    n_labeled = int(np.count_nonzero(result.tri_labels >= 0))
    vf_lines = format_volume_fraction_lines(
        vox=vox,
        mesh=result.mesh,
        shape=shape,
        spacing=spacing,
        vf_nito=_vf,
    )
    summary = (
        f"done  {out}  ({result.mesh.vertices.shape[0]:,} vertices, "
        f"{n_patches} BC patches, {n_labeled:,} labeled triangles)"
    )
    logger(summary)
    for line in vf_lines:
        logger(line)
    logger.save()

    if not verbose:
        print(summary)
        for line in vf_lines:
            print(line)


if __name__ == "__main__":
    main()