"""Structured hexahedral mesh from NITO binary voxel topologies."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lib.converters.bc_patch import build_patches_from_nito
from lib.meshing.surface_tags import BC_FACET_MARKER_OFFSET, FREE_FACET_MARKER
from lib.meshing.types import MeshBackend, VolumeMesh
from lib.volume_check import binarize_topology

# VTK hexahedron face index → (axis, side) for cell (i, j, k).
_HEX_FACE_INFO: tuple[tuple[int, str], ...] = (
    (2, "min"),
    (2, "max"),
    (1, "min"),
    (1, "max"),
    (0, "min"),
    (0, "max"),
)


@dataclass
class BuiltVoxelHexMesh:
    """In-memory DOLFINx hex mesh built from a binary voxel grid."""

    volume: VolumeMesh
    domain: object
    facet_tags: object
    cell_ijk: np.ndarray
    vox: np.ndarray


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


def _face_exposed(
    solid: np.ndarray,
    cell: tuple[int, int, int],
    axis: int,
    side: str,
) -> bool:
    ni = list(cell)
    if side == "min":
        nb = ni[axis] - 1
        if nb < 0:
            return True
        ni[axis] = nb
    else:
        nb = ni[axis] + 1
        if nb >= solid.shape[axis]:
            return True
        ni[axis] = nb
    return not solid[tuple(ni)]


def _face_marker(
    cell: tuple[int, int, int],
    axis: int,
    side: str,
    *,
    patch_lookup: dict[tuple[int, str, tuple[int, ...]], int],
) -> int:
    inplane = tuple(cell[i] for i in range(3) if i != axis)
    key = (axis, side, *inplane)
    patch_id = patch_lookup.get(key)
    if patch_id is not None:
        return BC_FACET_MARKER_OFFSET + patch_id
    return FREE_FACET_MARKER


def _patch_face_lookup(patches) -> dict[tuple[int, str, tuple[int, ...]], int]:
    lookup: dict[tuple[int, str, tuple[int, ...]], int] = {}
    for patch_id, patch in enumerate(patches):
        if not patch.is_bc:
            continue
        for inplane in patch.faces:
            key = (patch.axis, patch.side, *tuple(int(x) for x in inplane))
            lookup[key] = patch_id
    return lookup


def _build_topology(
    solid: np.ndarray,
    shape: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (points, cells_dolfinx, cell_ijk)."""
    nelx, nely, nelz = (int(x) for x in solid.shape)
    nx, ny = nelx + 1, nely + 1

    solid_cells = [tuple(int(v) for v in idx) for idx in np.argwhere(solid)]
    if not solid_cells:
        raise ValueError("Binary voxel grid has no solid elements")

    grid_to_compact: dict[int, int] = {}
    cells_vtk: list[list[int]] = []
    cell_ijk: list[tuple[int, int, int]] = []

    for cell in solid_cells:
        i, j, k = cell
        corners = _hex_vtk_corners(i, j, k, nx, ny)
        compact: list[int] = []
        for gid in corners:
            if gid not in grid_to_compact:
                grid_to_compact[gid] = len(grid_to_compact)
            compact.append(grid_to_compact[gid])
        cells_vtk.append(compact)
        cell_ijk.append(cell)

    scale = float(np.max(shape))
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


def _facet_tags_from_geometry(
    domain,
    cell_ijk: np.ndarray,
    solid: np.ndarray,
    patch_lookup: dict[tuple[int, str, tuple[int, ...]], int],
):
    """Tag exterior hex facets using cell indices and axis-aligned normals."""
    from dolfinx.mesh import meshtags

    scale = float(np.max(solid.shape))
    tdim = domain.topology.dim
    fdim = tdim - 1
    x = np.asarray(domain.geometry.x, dtype=float)

    domain.topology.create_connectivity(tdim, fdim)
    domain.topology.create_connectivity(fdim, tdim)
    domain.topology.create_connectivity(fdim, 0)

    c2f = domain.topology.connectivity(tdim, fdim)
    f2c = domain.topology.connectivity(fdim, tdim)
    f2v = domain.topology.connectivity(fdim, 0)

    facet_index: list[int] = []
    facet_value: list[int] = []

    n_cells = domain.topology.index_map(tdim).size_local
    for c in range(n_cells):
        i, j, k = (int(v) for v in cell_ijk[c])
        cell_center = np.array([(i + 0.5), (j + 0.5), (k + 0.5)], dtype=float) / scale

        for lf in range(c2f.num_links(c)):
            f = int(c2f.links(c)[lf])
            if f2c.num_links(f) != 1:
                continue
            if f in facet_index:
                continue

            verts = f2v.links(f)
            facet_center = x[verts].mean(axis=0)
            delta = facet_center - cell_center
            axis = int(np.argmax(np.abs(delta)))
            side = "max" if delta[axis] > 0 else "min"

            if not _face_exposed(solid, (i, j, k), axis, side):
                continue

            marker = _face_marker(
                (i, j, k), axis, side, patch_lookup=patch_lookup
            )
            facet_index.append(f)
            facet_value.append(marker)

    if not facet_index:
        raise RuntimeError("No tagged exterior facets on voxel hex mesh")

    return meshtags(
        domain,
        fdim,
        np.asarray(facet_index, dtype=np.int32),
        np.asarray(facet_value, dtype=np.int32),
    )


def build_voxel_hex_from_nito(
    *,
    index: int,
    shape: np.ndarray,
    rho: np.ndarray,
    bc: np.ndarray,
    output_dir: Path | None = None,
    cutoff: float = 0.5,
) -> BuiltVoxelHexMesh:
    """
    Binary NITO topology → condensed structured hex mesh in DOLFINx.

    Solid voxels become Q1 hexahedra; exterior faces are tagged with the same
    facet-marker convention as tagged VTK surfaces (1 = free, 2+patch = BC).
    """
    shape = np.asarray(shape, dtype=int).ravel()
    if shape.size != 3:
        raise ValueError(
            f"Voxel hex meshing requires 3D NITO shape (nelx, nely, nelz); got {shape}"
        )

    vox = binarize_topology(rho, shape, cutoff=cutoff)
    if not np.any(vox):
        raise ValueError(f"Sample {index}: binary topology is empty after cutoff={cutoff}")

    origin = np.zeros(3, dtype=float)
    spacing = np.ones(3, dtype=float)
    patches = build_patches_from_nito(vox, bc, shape, origin, spacing)
    patch_lookup = _patch_face_lookup(patches)

    points, cells, cell_ijk = _build_topology(vox, shape)
    domain = _create_dolfinx_mesh(points, cells)
    facet_tags = _facet_tags_from_geometry(
        domain, cell_ijk, vox, patch_lookup
    )

    meta_path = None
    if output_dir is not None:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        meta_path = output_dir / "voxel_hex_meta.json"
        meta = {
            "index": index,
            "shape": shape.tolist(),
            "n_hex": int(cells.shape[0]),
            "n_nodes": int(points.shape[0]),
            "n_solid_voxels": int(np.count_nonzero(vox)),
            "facet_markers": {
                str(FREE_FACET_MARKER): "free_boundary",
                **{
                    str(BC_FACET_MARKER_OFFSET + i): f"bc_patch_{i}"
                    for i in range(len(patches))
                },
            },
            "patches": [
                {
                    "patch_id": i,
                    "axis": p.axis,
                    "side": p.side,
                    "flags": np.asarray(p.flags, dtype=float).tolist(),
                    "n_faces": len(p.faces),
                }
                for i, p in enumerate(patches)
                if p.is_bc
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
        volume=volume,
        domain=domain,
        facet_tags=facet_tags,
        cell_ijk=cell_ijk,
        vox=vox,
    )
