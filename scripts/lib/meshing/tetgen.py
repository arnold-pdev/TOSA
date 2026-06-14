"""TetGen volume meshing from tagged VTK surfaces."""

from __future__ import annotations

import json
from pathlib import Path

import meshio
import numpy as np

from lib.meshing.size_fields import MeshSizing
from lib.meshing.surface_tags import (
    FACET_MARKER_KEY,
    marker_names,
    physical_groups_by_marker,
    read_surface_tags,
)
from lib.meshing.types import MeshBackend, VolumeMesh


def _triangles_from_polydata(poly) -> np.ndarray:
    raw = np.asarray(poly.faces, dtype=np.int64)
    if raw.size % 4 != 0:
        raise ValueError(f"Expected triangle vtk faces (n,i,j,k); got {raw.size} ints")
    return raw.reshape(-1, 4)[:, 1:4]


def build_volume_mesh_tetgen(
    *,
    index: int,
    surface_path: Path,
    output_dir: Path,
    sizing: MeshSizing,
) -> VolumeMesh:
    """
    Tetrahedralize from a tagged VTK surface using TetGen facet markers.

    facet_marker cell_data from the .vtp/.vtk is forwarded as TetGen PLC
    boundary markers and preserved in the output .xdmf volume mesh.
    """
    import tetgen

    tags = read_surface_tags(surface_path)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_path = output_dir / "surface_tags.json"
    volume_xdmf = output_dir / "volume.xdmf"

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

    surface = tags.mesh.copy()
    surface.cell_data[FACET_MARKER_KEY] = np.asarray(tags.facet_markers, dtype=np.int32)

    tg = tetgen.TetGen(surface)
    tg.make_manifold(verbose=0)
    # PLC with quality/refinement switches; sizes are in VTK length units.
    sw = (
        f"pq{sizing.h_boundary:.4g}a{sizing.h_bulk:.4g}"
        f"Y1"
    )
    tg.tetrahedralize(switches=sw, verbose=0)

    grid = tg.grid
    points = np.asarray(grid.points, dtype=float)
    cells = []
    cell_data: dict[str, list[np.ndarray]] = {}
    if hasattr(grid, "cells_dict") and "tetra" in grid.cells_dict:
        tets = np.asarray(grid.cells_dict["tetra"], dtype=np.int64)
        cells.append(("tetra", tets))
    else:
        raise RuntimeError("TetGen did not return tetrahedral cells")

    if FACET_MARKER_KEY in surface.cell_data:
        cell_data["facet_marker_source"] = [np.asarray(tags.facet_markers, dtype=np.int32)]

    meshio.Mesh(points=points, cells=cells, cell_data=cell_data or None).write(
        volume_xdmf
    )

    return VolumeMesh(
        index=index,
        backend=MeshBackend.TETGEN,
        path=volume_xdmf,
        meta_path=meta_path,
    )
