"""Volume mesh generation — structured NITO voxel-hex (DOLFINx)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lib.meshing.voxel_hex import BuiltVoxelHexMesh, build_voxel_hex_from_nito


def build_nito_hex_mesh(
    *,
    index: int,
    shape: np.ndarray,
    rho: np.ndarray,
    bc: np.ndarray,
    output_dir: Path,
    refine: int = 1,
) -> BuiltVoxelHexMesh:
    """Binary NITO sample → DOLFINx structured hex mesh (the volume-mesh path).

    ``refine`` upsamples the grid (×refine per axis) — finer staircase, same extent."""
    return build_voxel_hex_from_nito(
        index=index,
        shape=shape,
        rho=rho,
        bc=bc,
        output_dir=output_dir,
        refine=refine,
    )
