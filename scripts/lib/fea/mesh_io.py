"""Load volume meshes into DOLFINx with facet markers."""

from __future__ import annotations

from pathlib import Path

from lib.meshing.types import MeshBackend, VolumeMesh
from lib.meshing.voxel_hex import BuiltVoxelHexMesh


def load_dolfinx_mesh(
    volume: VolumeMesh | BuiltVoxelHexMesh,
):
    """
    Read a volume mesh into (domain, cell_tags, facet_tags).

    For ``BuiltVoxelHexMesh``, returns the in-memory mesh directly.
    Facet markers follow ``lib.meshing.surface_tags`` (1 = free, 2+patch = BC).
    """
    if isinstance(volume, BuiltVoxelHexMesh):
        return volume.domain, None, volume.facet_tags

    from mpi4py import MPI

    path = Path(volume.path).resolve()
    if volume.backend == MeshBackend.VOXEL_HEX:
        raise ValueError(
            "VOXEL_HEX VolumeMesh has no DOLFINx geometry on disk; "
            "use BuiltVoxelHexMesh from build_voxel_hex_from_nito()."
        )

    if not path.is_file():
        raise FileNotFoundError(f"Volume mesh not found: {path}")

    if volume.backend == MeshBackend.GMSH or path.suffix.lower() == ".msh":
        from dolfinx.io import gmshio

        domain, cell_tags, facet_tags = gmshio.read_from_msh(
            str(path), MPI.COMM_WORLD, gdim=3
        )
    elif volume.backend == MeshBackend.TETGEN or path.suffix.lower() == ".xdmf":
        raise NotImplementedError(
            "TetGen XDMF → DOLFINx with facet markers is not wired yet. "
            "Use --mesh-backend gmsh or voxel_hex."
        )
    else:
        raise ValueError(f"Unsupported volume mesh: {path}")

    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    return domain, cell_tags, facet_tags

