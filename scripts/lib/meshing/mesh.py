"""Dispatch mesh generation by backend."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lib.meshing.gmsh import build_volume_mesh_gmsh
from lib.meshing.size_fields import MeshSizing
from lib.meshing.tetgen import build_volume_mesh_tetgen
from lib.meshing.types import MeshBackend, VolumeMesh
from lib.meshing.voxel_hex import BuiltVoxelHexMesh, build_voxel_hex_from_nito


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
    if backend == MeshBackend.VOXEL_HEX:
        raise ValueError(
            "Use build_voxel_hex_from_nito() for the voxel_hex backend "
            "(requires NITO shape/rho/bc, not a surface file)."
        )
    return builders[backend](
        index=index,
        surface_path=surface_path,
        output_dir=output_dir,
        sizing=sizing,
    )


def build_nito_hex_mesh(
    *,
    index: int,
    shape: np.ndarray,
    rho: np.ndarray,
    bc: np.ndarray,
    output_dir: Path,
) -> BuiltVoxelHexMesh:
    """Binary NITO sample → DOLFINx hex mesh (primary Stage-0 path)."""
    return build_voxel_hex_from_nito(
        index=index,
        shape=shape,
        rho=rho,
        bc=bc,
        output_dir=output_dir,
    )

