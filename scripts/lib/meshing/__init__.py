"""Volume mesh generation (structured NITO voxel hex)."""

from lib.meshing.mesh import build_nito_hex_mesh
from lib.meshing.surface_tags import SurfaceTags, read_surface_tags
from lib.meshing.types import MeshBackend, VolumeMesh
from lib.meshing.voxel_hex import (
    BuiltVoxelHexMesh,
    build_hex_from_surface,
    build_hex_mesh,
    build_voxel_hex_from_nito,
    occupancy_from_surface,
    occupancy_from_voxels,
)

__all__ = [
    "BuiltVoxelHexMesh",
    "MeshBackend",
    "SurfaceTags",
    "VolumeMesh",
    "build_hex_from_surface",
    "build_hex_mesh",
    "build_nito_hex_mesh",
    "build_voxel_hex_from_nito",
    "occupancy_from_surface",
    "occupancy_from_voxels",
    "read_surface_tags",
]
