"""Volume mesh generation (Gmsh, TetGen, NITO voxel hex)."""

from lib.meshing.mesh import build_nito_hex_mesh, build_volume_mesh
from lib.meshing.size_fields import MeshSizing, sizing_from_nito_sample
from lib.meshing.surface_tags import SurfaceTags, read_surface_tags
from lib.meshing.types import MeshBackend, VolumeMesh
from lib.meshing.voxel_hex import BuiltVoxelHexMesh, build_voxel_hex_from_nito

__all__ = [
    "BuiltVoxelHexMesh",
    "MeshBackend",
    "MeshSizing",
    "SurfaceTags",
    "VolumeMesh",
    "build_nito_hex_mesh",
    "build_volume_mesh",
    "build_voxel_hex_from_nito",
    "read_surface_tags",
    "sizing_from_nito_sample",
]
