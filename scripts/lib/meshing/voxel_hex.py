"""Structured hexahedral mesh from a NITO topology.

The mesher is split into an *occupancy source* (which cells are solid, at a chosen
resolution) and a backend-agnostic *hex builder* (occupancy → tagged DOLFINx hex
mesh). Two sources:

- ``occupancy_from_voxels`` — the binary voxel grid, optionally upsampled by
  ``refine`` (nearest replication). Fast; finer staircase, no new geometry.
- ``occupancy_from_surface`` — inside/outside test of fine cell centres against a
  watertight smooth surface (e.g. a voxel2surf output). Finer cells that hug the
  true boundary.

Both keep the structured Cartesian grid (so build layers / element birth for AM
are trivial) and the same physical extent regardless of ``refine``. BC support
patches are tagged on the *domain-boundary* facets only; with ``refine`` a fine
boundary facet maps back to its base footprint cell by integer division.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lib.converters.bc_patch import build_patches_from_nito
from lib.meshing.surface_tags import BC_FACET_MARKER_OFFSET, FREE_FACET_MARKER
from lib.meshing.types import MeshBackend, VolumeMesh
from lib.volume_check import binarize_topology


@dataclass
class BuiltVoxelHexMesh:
    """In-memory DOLFINx hex mesh built from a voxel/surface occupancy grid."""

    volume: VolumeMesh
    domain: object
    facet_tags: object
    cell_ijk: np.ndarray   # (n_hex, 3) cell indices at the *target* (refined) resolution
    vox: np.ndarray        # binary occupancy at *base* resolution
    refine: int = 1


# --- occupancy sources -------------------------------------------------------
def occupancy_from_voxels(vox: np.ndarray, refine: int = 1) -> np.ndarray:
    """Binary occupancy, upsampled ``refine``× per axis by nearest replication."""
    occ = np.asarray(vox) > 0
    if refine <= 1:
        return occ
    return np.repeat(np.repeat(np.repeat(occ, refine, 0), refine, 1), refine, 2)


def occupancy_from_surface(
    verts: np.ndarray,
    faces: np.ndarray,
    shape: np.ndarray,
    *,
    refine: int = 1,
    origin=(0.0, 0.0, 0.0),
    spacing=(1.0, 1.0, 1.0),
) -> np.ndarray:
    """Occupancy at ``shape·refine`` from an inside/outside test of fine cell
    centres against the (watertight) surface ``(verts, faces)``.

    The surface is assumed in the cuberille frame (``origin + grid·spacing``), so
    fine cell ``(I,J,K)`` centres sit at ``origin + (I+0.5)·spacing/refine``. Quads
    are triangulated for the ray-cast containment test (trimesh)."""
    import trimesh

    verts = np.asarray(verts, dtype=float)
    faces = np.asarray(faces)
    tris = (
        np.vstack([faces[:, [0, 1, 2]], faces[:, [0, 2, 3]]])
        if faces.shape[1] == 4
        else faces
    )
    mesh = trimesh.Trimesh(verts, tris, process=False)

    shape = np.asarray(shape, dtype=int).ravel()
    origin = np.asarray(origin, dtype=float)
    spacing = np.asarray(spacing, dtype=float)
    fine = shape * int(refine)

    gi, gj, gk = np.meshgrid(
        np.arange(fine[0]), np.arange(fine[1]), np.arange(fine[2]), indexing="ij"
    )
    idx = np.stack([gi.ravel(), gj.ravel(), gk.ravel()], axis=1).astype(float)
    centres = origin + (idx + 0.5) * spacing / float(refine)
    inside = mesh.contains(centres)
    return inside.reshape(tuple(int(x) for x in fine))


# --- topology / node coordinates ---------------------------------------------
def _grid_node_index(i: int, j: int, k: int, nx: int, ny: int) -> int:
    return i + nx * j + nx * ny * k


def _hex_vtk_corners(i: int, j: int, k: int, nx: int, ny: int) -> list[int]:
    return [
        _grid_node_index(i, j, k, nx, ny),
        _grid_node_index(i + 1, j, k, nx, ny),
        _grid_node_index(i + 1, j + 1, k, nx, ny),
        _grid_node_index(i, j + 1, k, nx, ny),
        _grid_node_index(i, j, k + 1, nx, ny),
        _grid_node_index(i + 1, j, k + 1, nx, ny),
        _grid_node_index(i + 1, j + 1, k + 1, nx, ny),
        _grid_node_index(i, j + 1, k + 1, nx, ny),
    ]


def _build_topology(occ: np.ndarray, scale: float):
    """``occ`` (bool, target res) → ``(points, cells_dolfinx, cell_ijk)``.

    Node coordinates are ``index / scale`` with ``scale = max(base_shape)·refine``,
    so the physical extent is identical for any ``refine``."""
    nelx, nely, nelz = (int(x) for x in occ.shape)
    nx, ny = nelx + 1, nely + 1

    solid_cells = [tuple(int(v) for v in idx) for idx in np.argwhere(occ)]
    if not solid_cells:
        raise ValueError("Occupancy grid has no solid cells")

    grid_to_compact: dict[int, int] = {}
    cells_vtk: list[list[int]] = []
    cell_ijk: list[tuple[int, int, int]] = []

    for cell in solid_cells:
        i, j, k = cell
        compact: list[int] = []
        for gid in _hex_vtk_corners(i, j, k, nx, ny):
            if gid not in grid_to_compact:
                grid_to_compact[gid] = len(grid_to_compact)
            compact.append(grid_to_compact[gid])
        cells_vtk.append(compact)
        cell_ijk.append(cell)

    points = np.zeros((len(grid_to_compact), 3), dtype=np.float64)
    for gid, cid in grid_to_compact.items():
        i = gid % nx
        j = (gid // nx) % ny
        k = gid // (nx * ny)
        points[cid] = (i / scale, j / scale, k / scale)

    from dolfinx.io.utils import cell_perm_vtk
    from dolfinx.mesh import CellType

    perm = cell_perm_vtk(CellType.hexahedron, 8)
    cells = np.asarray(cells_vtk, dtype=np.int64)[:, perm]
    return points, cells, np.asarray(cell_ijk, dtype=np.int32)


def _create_dolfinx_mesh(points: np.ndarray, cells: np.ndarray):
    from mpi4py import MPI

    import basix.ufl
    import ufl
    from dolfinx.mesh import create_mesh

    gdim = 3
    coord_el = ufl.Mesh(basix.ufl.element("Lagrange", "hexahedron", 1, shape=(gdim,)))
    domain = create_mesh(MPI.COMM_WORLD, cells, coord_el, points)
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim - 1)
    return domain


# --- facet (BC) tagging ------------------------------------------------------
def _patch_face_lookup(patches) -> dict[tuple[int, str, tuple[int, ...]], int]:
    """Base-resolution support-patch footprint → patch id, keyed (axis, side, *inplane)."""
    lookup: dict[tuple[int, str, tuple[int, ...]], int] = {}
    for patch_id, patch in enumerate(patches):
        if not patch.is_bc:
            continue
        for inplane in patch.faces:
            key = (patch.axis, patch.side, *tuple(int(x) for x in inplane))
            lookup[key] = patch_id
    return lookup


def _facet_tags_from_geometry(
    domain,
    cell_ijk: np.ndarray,
    grid_shape: tuple[int, int, int],
    patch_lookup: dict[tuple[int, str, tuple[int, ...]], int],
    *,
    scale: float,
    refine: int,
):
    """Tag exterior hex facets (1 = free, 2+patch = BC).

    An exterior facet adjacent to a single (solid) cell is *exposed*; a BC support
    marker is applied only when the facet lies on the actual **domain boundary**
    (the cell is on the min/max grid slab of that axis) and its **base footprint**
    cell — fine in-plane index ``// refine`` — is in a patch. Interior void-adjacent
    facets (and unmatched boundary facets) are free."""
    from dolfinx.mesh import meshtags

    n = tuple(int(s) for s in grid_shape)
    tdim = domain.topology.dim
    fdim = tdim - 1
    x = np.asarray(domain.geometry.x, dtype=float)

    domain.topology.create_connectivity(tdim, fdim)
    domain.topology.create_connectivity(fdim, tdim)
    domain.topology.create_connectivity(fdim, 0)
    c2f = domain.topology.connectivity(tdim, fdim)
    f2c = domain.topology.connectivity(fdim, tdim)
    f2v = domain.topology.connectivity(fdim, 0)

    seen: set[int] = set()
    facet_index: list[int] = []
    facet_value: list[int] = []

    n_cells = domain.topology.index_map(tdim).size_local
    for c in range(n_cells):
        cell = tuple(int(v) for v in cell_ijk[c])
        cell_center = (np.array(cell, dtype=float) + 0.5) / scale

        for lf in range(c2f.num_links(c)):
            f = int(c2f.links(c)[lf])
            if f2c.num_links(f) != 1 or f in seen:
                continue
            seen.add(f)

            facet_center = x[f2v.links(f)].mean(axis=0)
            delta = facet_center - cell_center
            axis = int(np.argmax(np.abs(delta)))
            side = "max" if delta[axis] > 0 else "min"

            on_domain_boundary = (
                (side == "min" and cell[axis] == 0)
                or (side == "max" and cell[axis] == n[axis] - 1)
            )
            marker = FREE_FACET_MARKER
            if on_domain_boundary:
                inplane = tuple(cell[a] // refine for a in range(3) if a != axis)
                patch_id = patch_lookup.get((axis, side, *inplane))
                if patch_id is not None:
                    marker = BC_FACET_MARKER_OFFSET + patch_id

            facet_index.append(f)
            facet_value.append(marker)

    if not facet_index:
        raise RuntimeError("No tagged exterior facets on hex mesh")
    return meshtags(
        domain, fdim,
        np.asarray(facet_index, dtype=np.int32),
        np.asarray(facet_value, dtype=np.int32),
    )


# --- core builder + finalize -------------------------------------------------
def build_hex_mesh(occ: np.ndarray, patches, *, base_shape: np.ndarray, refine: int = 1):
    """Occupancy grid (target res) + BC patches → ``(domain, facet_tags, cell_ijk,
    n_nodes, n_hex)``. Backend-agnostic: source decides ``occ``, this builds the mesh."""
    occ = np.asarray(occ, dtype=bool)
    scale = float(np.max(base_shape)) * int(refine)
    patch_lookup = _patch_face_lookup(patches)
    points, cells, cell_ijk = _build_topology(occ, scale)
    domain = _create_dolfinx_mesh(points, cells)
    facet_tags = _facet_tags_from_geometry(
        domain, cell_ijk, occ.shape, patch_lookup, scale=scale, refine=int(refine)
    )
    return domain, facet_tags, cell_ijk, int(points.shape[0]), int(cells.shape[0])


def _finalize(
    *, index, domain, facet_tags, cell_ijk, vox, patches, base_shape, refine,
    n_nodes, n_hex, source, output_dir,
) -> BuiltVoxelHexMesh:
    meta_path = None
    if output_dir is not None:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        meta_path = output_dir / "voxel_hex_meta.json"
        meta = {
            "index": index,
            "source": source,
            "base_shape": np.asarray(base_shape, dtype=int).tolist(),
            "refine": int(refine),
            "n_hex": n_hex,
            "n_nodes": n_nodes,
            "n_solid_voxels": int(np.count_nonzero(vox)),
            "facet_markers": {
                str(FREE_FACET_MARKER): "free_boundary",
                **{str(BC_FACET_MARKER_OFFSET + i): f"bc_patch_{i}" for i in range(len(patches))},
            },
            "patches": [
                {
                    "patch_id": i, "axis": p.axis, "side": p.side,
                    "flags": np.asarray(p.flags, dtype=float).tolist(), "n_faces": len(p.faces),
                }
                for i, p in enumerate(patches) if p.is_bc
            ],
        }
        meta_path.write_text(json.dumps(meta, indent=2))

    volume = VolumeMesh(
        index=index,
        backend=MeshBackend.VOXEL_HEX,
        path=meta_path or Path(f"voxel_hex_{index}"),
        meta_path=meta_path,
    )
    return BuiltVoxelHexMesh(
        volume=volume, domain=domain, facet_tags=facet_tags,
        cell_ijk=cell_ijk, vox=vox, refine=int(refine),
    )


# --- public builders ---------------------------------------------------------
def build_voxel_hex_from_nito(
    *,
    index: int,
    shape: np.ndarray,
    rho: np.ndarray,
    bc: np.ndarray,
    output_dir: Path | None = None,
    cutoff: float = 0.5,
    refine: int = 1,
) -> BuiltVoxelHexMesh:
    """Binary NITO topology → structured hex mesh (voxel occupancy, ``refine``×)."""
    shape = np.asarray(shape, dtype=int).ravel()
    if shape.size != 3:
        raise ValueError(f"Voxel hex meshing requires 3D NITO shape; got {shape}")
    vox = binarize_topology(rho, shape, cutoff=cutoff)
    if not np.any(vox):
        raise ValueError(f"Sample {index}: binary topology is empty after cutoff={cutoff}")

    patches = build_patches_from_nito(vox, bc, shape, np.zeros(3), np.ones(3))
    occ = occupancy_from_voxels(vox, refine)
    domain, facet_tags, cell_ijk, n_nodes, n_hex = build_hex_mesh(
        occ, patches, base_shape=shape, refine=refine
    )
    return _finalize(
        index=index, domain=domain, facet_tags=facet_tags, cell_ijk=cell_ijk,
        vox=vox, patches=patches, base_shape=shape, refine=refine,
        n_nodes=n_nodes, n_hex=n_hex, source="voxels", output_dir=output_dir,
    )


def build_hex_from_surface(
    *,
    index: int,
    verts: np.ndarray,
    faces: np.ndarray,
    shape: np.ndarray,
    rho: np.ndarray,
    bc: np.ndarray,
    output_dir: Path | None = None,
    cutoff: float = 0.5,
    refine: int = 1,
) -> BuiltVoxelHexMesh:
    """Smooth surface → structured hex mesh: occupancy by inside/outside test of
    fine cell centres against ``(verts, faces)``. BC patches still come from the
    binary (the domain box faces); ``refine`` sets the grid resolution."""
    shape = np.asarray(shape, dtype=int).ravel()
    if shape.size != 3:
        raise ValueError(f"Hex meshing requires 3D NITO shape; got {shape}")
    vox = binarize_topology(rho, shape, cutoff=cutoff)
    patches = build_patches_from_nito(vox, bc, shape, np.zeros(3), np.ones(3))

    occ = occupancy_from_surface(verts, faces, shape, refine=refine)
    if not occ.any():
        raise ValueError(f"Sample {index}: surface occupancy is empty (refine={refine})")
    domain, facet_tags, cell_ijk, n_nodes, n_hex = build_hex_mesh(
        occ, patches, base_shape=shape, refine=refine
    )
    return _finalize(
        index=index, domain=domain, facet_tags=facet_tags, cell_ijk=cell_ijk,
        vox=vox, patches=patches, base_shape=shape, refine=refine,
        n_nodes=n_nodes, n_hex=n_hex, source="surface", output_dir=output_dir,
    )
