"""Load volume meshes into DOLFINx with facet markers."""

from __future__ import annotations

from lib.meshing.types import VolumeMesh
from lib.meshing.voxel_hex import BuiltVoxelHexMesh


def load_dolfinx_mesh(
    volume: VolumeMesh | BuiltVoxelHexMesh,
):
    """
    Return ``(domain, cell_tags, facet_tags)`` for a built voxel-hex mesh.

    The structured voxel-hex backend builds its DOLFINx mesh in memory, so the
    mesh is returned directly; facet markers follow ``lib.meshing.surface_tags``
    (1 = free, 2+patch = BC).
    """
    if isinstance(volume, BuiltVoxelHexMesh):
        return volume.domain, None, volume.facet_tags

    raise ValueError(
        "Only BuiltVoxelHexMesh is supported; build it with "
        "build_voxel_hex_from_nito() (no on-disk volume meshes after the "
        "tet/gmsh backends were removed)."
    )
