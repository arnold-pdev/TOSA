"""Dispatch mesh generation by backend."""

from __future__ import annotations

from pathlib import Path

from lib.meshing.gmsh import build_volume_mesh_gmsh
from lib.meshing.size_fields import MeshSizing
from lib.meshing.tetgen import build_volume_mesh_tetgen
from lib.meshing.types import MeshBackend, VolumeMesh


def build_volume_mesh(
    *,
    index: int,
    surface_path: Path,
    output_dir: Path,
    sizing: MeshSizing,
    backend: MeshBackend = MeshBackend.GMSH,
) -> VolumeMesh:
    builders = {
        MeshBackend.GMSH: build_volume_mesh_gmsh,
        MeshBackend.TETGEN: build_volume_mesh_tetgen,
    }
    return builders[backend](
        index=index,
        surface_path=surface_path,
        output_dir=output_dir,
        sizing=sizing,
    )
