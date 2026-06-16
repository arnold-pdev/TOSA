"""Boundary shape-derivative post-processing."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lib.fea.shape_derivative import BoundaryL2Geometry
from lib.fea.types import FEASolution, ShapeDerivativeOnSurface
from lib.meshing.surface_tags import FREE_BOUNDARY_ID, PATCH_ID_KEY


def _boundary_node_coords(domain, geometry: BoundaryL2Geometry) -> np.ndarray:
    coords = np.asarray(domain.geometry.x, dtype=float)
    return coords[geometry.free_mask]


def transfer_nodal_field_to_surface(
    nodal_field: np.ndarray,
    geometry: BoundaryL2Geometry,
    domain,
    surface_path: Path,
    *,
    field_name: str = "V_C",
) -> tuple[np.ndarray, object]:
    """
    Nearest-neighbour transfer from free-boundary volume nodes to a design .vtp.

    Returns (point_values, pyvista.PolyData).
    """
    from scipy.spatial import cKDTree

    import pyvista as pv

    surface_path = Path(surface_path).resolve()
    mesh = pv.read(surface_path)
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()

    src_pts = _boundary_node_coords(domain, geometry)
    src_vals = nodal_field[geometry.free_mask]
    if src_pts.shape[0] == 0:
        raise ValueError("No free-boundary nodes found on the volume mesh")

    tree = cKDTree(src_pts)
    dist, idx = tree.query(np.asarray(mesh.points, dtype=float))
    values = src_vals[idx]

    if PATCH_ID_KEY in mesh.cell_data:
        patch_id = np.asarray(mesh.cell_data[PATCH_ID_KEY], dtype=np.int32)
        free_cells = patch_id == FREE_BOUNDARY_ID
        if np.any(free_cells):
            free_pts = mesh.extract_cells(np.where(free_cells)[0]).points
            _, free_idx = tree.query(free_pts)
            # Leave BC-patch points at nearest value; optional masking could zero them.
            _ = free_idx

    mesh[field_name] = values
    if np.any(dist > 0.0):
        mesh[f"{field_name}_transfer_dist"] = dist
    return values, mesh


def write_shape_derivative_to_vtp(
    nodal_field: np.ndarray,
    geometry: BoundaryL2Geometry,
    domain,
    *,
    surface_path: Path,
    output_path: Path,
    field_name: str = "V_C",
) -> Path:
    """Write a shape-derivative scalar field onto the design surface."""
    _values, mesh = transfer_nodal_field_to_surface(
        nodal_field,
        geometry,
        domain,
        surface_path,
        field_name=field_name,
    )
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.save(output_path)
    return output_path


def write_hex_free_boundary_vtp(
    nodal_field: np.ndarray,
    geometry: BoundaryL2Geometry,
    domain,
    facet_tags,
    *,
    output_path: Path,
    field_name: str = "V_C",
    free_marker: int = 1,
) -> Path:
    """
    Write the free-boundary staircase surface (hex exterior) with a nodal scalar.

    Quads on each exposed free facet are split into two triangles for VTK.
    """
    import pyvista as pv

    from lib.meshing.surface_tags import FREE_FACET_MARKER

    free_marker = free_marker or FREE_FACET_MARKER
    tdim = domain.topology.dim
    fdim = tdim - 1
    x = np.asarray(domain.geometry.x, dtype=float)

    domain.topology.create_connectivity(fdim, 0)
    f2v = domain.topology.connectivity(fdim, 0)

    points: list[np.ndarray] = []
    values: list[float] = []
    faces: list[int] = []
    point_map: dict[int, int] = {}

    def _get_point(vidx: int) -> int:
        if vidx not in point_map:
            point_map[vidx] = len(points)
            points.append(x[vidx])
            values.append(float(nodal_field[vidx]))
        return point_map[vidx]

    for f, tag in zip(facet_tags.indices, facet_tags.values, strict=True):
        if int(tag) != free_marker:
            continue
        verts = f2v.links(int(f))
        if len(verts) != 4:
            continue
        ids = [_get_point(int(v)) for v in verts]
        faces.extend([3, ids[0], ids[1], ids[2], 3, ids[0], ids[2], ids[3]])

    if not points:
        raise ValueError("No free-boundary facets found on hex mesh")

    mesh = pv.PolyData(
        np.asarray(points, dtype=float),
        faces=np.asarray(faces, dtype=np.int64),
    )
    mesh[field_name] = np.asarray(values, dtype=float)
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.save(output_path)
    return output_path


def boundary_eps_sigma_contraction(
    solution: FEASolution,
    surface_path: Path,
    *,
    output_path: Path,
) -> Path:
    """
    Legacy boundary-trace entry point.

    Prefer the volume-form path in ``LinearElasticityProblem.solve``, which writes
    ``V_C`` via L²(Γ) projection. If ``solution.shape_derivative_path`` exists,
    copy it to ``output_path``.
    """
    if solution.shape_derivative_path is not None and solution.shape_derivative_path.is_file():
        import shutil

        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(solution.shape_derivative_path, output_path)
        return output_path

    raise NotImplementedError(
        "Boundary ε:σ trace post-processing is superseded by the volume-form "
        "L²(Γ) projection. Run LinearElasticityProblem.solve() first."
    )


def enrich_surface_with_shape_derivative(
    result: ShapeDerivativeOnSurface,
) -> Path:
    """Persist a precomputed surface field (wrapper for batch pipelines)."""
    import pyvista as pv

    mesh = pv.read(result.surface_path)
    mesh[result.field_name] = result.values
    result.output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.save(result.output_path)
    return result.output_path
