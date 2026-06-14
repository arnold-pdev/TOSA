"""Linear elasticity solve with FEniCSx."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lib.fea.bcs import apply_surface_bcs
from lib.fea.types import FEASolution, MaterialProperties
from lib.meshing.surface_tags import read_surface_tags
from lib.meshing.types import VolumeMesh


class LinearElasticityProblem:
    """Assemble and solve isotropic linear elasticity on a volume mesh."""

    def __init__(
        self,
        mesh: VolumeMesh,
        material: MaterialProperties | None = None,
        *,
        surface_path: Path | None = None,
    ) -> None:
        self.mesh = mesh
        self.material = material or MaterialProperties()
        self.surface_path = surface_path
        self.surface_bc_plan = None

    def solve(
        self,
        shape: np.ndarray,
        bc: np.ndarray,
        load: np.ndarray,
        *,
        output_dir: Path,
        surface_path: Path | None = None,
    ) -> FEASolution:
        surface = surface_path or self.surface_path
        if surface is None:
            raise ValueError(
                "Tagged VTK surface path required. "
                "Pass surface_path= to solve() or LinearElasticityProblem(...)."
            )
        tags = read_surface_tags(surface)
        apply_surface_bcs(self, tags, shape, load)
        raise NotImplementedError(
            "FEniCSx elasticity solve is not yet implemented. "
            f"Surface BC plan: {len(self.surface_bc_plan.dirichlet)} Dirichlet patches, "
            f"{len(self.surface_bc_plan.loads)} loads."
        )
