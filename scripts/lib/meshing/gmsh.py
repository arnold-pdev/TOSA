"""Gmsh volume meshing from tagged VTK surfaces."""

from __future__ import annotations

import json
from pathlib import Path

import meshio
import numpy as np

from lib.meshing.size_fields import MeshSizing
from lib.meshing.surface_tags import (
    marker_names,
    physical_groups_by_marker,
    read_surface_tags,
)
from lib.meshing.types import MeshBackend, VolumeMesh


def _triangles_from_polydata(poly) -> np.ndarray:
    raw = np.asarray(poly.faces, dtype=np.int64)
    if raw.size == 0:
        raise ValueError("Surface PolyData has no face connectivity")
    if raw.size % 4 != 0:
        raise ValueError(f"Expected triangle vtk faces (n,i,j,k); got {raw.size} ints")
    return raw.reshape(-1, 4)[:, 1:4]


def _write_tagged_surface_msh(tags, path: Path) -> None:
    """Write a 2D msh with triangle physical groups = facet_marker."""
    triangles = _triangles_from_polydata(tags.mesh)
    markers = np.asarray(tags.facet_markers, dtype=np.int32)
    mesh = meshio.Mesh(
        points=np.asarray(tags.mesh.points, dtype=float),
        cells=[("triangle", triangles)],
        cell_data={
            "gmsh:physical": [markers],
            "gmsh:geometrical": [markers],
        },
    )
    meshio.write(path, mesh, file_format="gmsh22", binary=False)


def _apply_gmsh_sizing(sizing: MeshSizing) -> None:
    import gmsh

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", sizing.h_boundary)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", sizing.h_bulk)


def build_volume_mesh_gmsh(
    *,
    index: int,
    surface_path: Path,
    output_dir: Path,
    sizing: MeshSizing,
) -> VolumeMesh:
    """
    Tetrahedralize the solid interior from a tagged VTK surface.

    Reads facet_marker physical groups from the .vtp/.vtk, preserves them on
    the boundary, and writes a 3D .msh plus a JSON tag catalogue.
    """
    import gmsh

    tags = read_surface_tags(surface_path)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    surface_msh = output_dir / "surface_tagged.msh"
    volume_msh = output_dir / "volume.msh"
    meta_path = output_dir / "surface_tags.json"

    _write_tagged_surface_msh(tags, surface_msh)

    meta = {
        "index": index,
        "source_surface": str(surface_path.resolve()),
        "physical_groups": {
            str(marker): {
                "name": marker_names(tags).get(marker, f"marker_{marker}"),
                "n_triangles": len(tris),
            }
            for marker, tris in physical_groups_by_marker(tags).items()
        },
        "patches": [
            {
                "patch_id": i,
                "facet_marker": 2 + i,
                "flags": np.asarray(p.flags, dtype=float).tolist(),
                "axis": p.axis,
                "side": p.side,
            }
            for i, p in enumerate(tags.patches)
        ],
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    gmsh.initialize()
    try:
        gmsh.model.add(f"tosa_{index}")
        gmsh.merge(str(surface_msh))
        _apply_gmsh_sizing(sizing)

        # Promote imported 2D physical groups; mesh the interior volume.
        dim_tags = gmsh.model.getEntities(3)
        if not dim_tags:
            gmsh.model.occ.synchronize()
            surfaces = [tag for dim, tag in gmsh.model.getEntities(2)]
            if surfaces:
                loop = gmsh.model.occ.addSurfaceLoop(surfaces)
                vol = gmsh.model.occ.addVolume([loop])
                gmsh.model.occ.synchronize()
                gmsh.model.addPhysicalGroup(3, [vol], tag=1, name="domain")
        else:
            vol_tags = [tag for _dim, tag in dim_tags]
            gmsh.model.addPhysicalGroup(3, vol_tags, tag=1, name="domain")

        gmsh.model.mesh.generate(3)
        gmsh.write(str(volume_msh))
    finally:
        gmsh.finalize()

    return VolumeMesh(
        index=index,
        backend=MeshBackend.GMSH,
        path=volume_msh,
        meta_path=meta_path,
    )
