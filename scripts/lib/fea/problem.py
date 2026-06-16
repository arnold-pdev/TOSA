"""Linear elasticity solve with FEniCSx."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lib.fea.bcs import apply_surface_bcs, apply_voxel_bcs
from lib.fea.elasticity import solve_linear_elasticity
from lib.fea.mesh_io import load_dolfinx_mesh
from lib.fea.shape_derivative import (
    BoundaryMassMode,
    assemble_boundary_l2_geometry,
    compliance_shape_derivative,
)
from lib.fea.types import FEASolution, MaterialProperties
from lib.meshing.surface_tags import read_surface_tags
from lib.meshing.types import MeshBackend, VolumeMesh
from lib.meshing.voxel_hex import BuiltVoxelHexMesh


class LinearElasticityProblem:
    """Assemble and solve isotropic linear elasticity on a volume mesh."""

    def __init__(
        self,
        mesh: VolumeMesh | BuiltVoxelHexMesh,
        material: MaterialProperties | None = None,
        *,
        surface_path: Path | None = None,
        element_degree: int = 1,
        boundary_mass_mode: BoundaryMassMode = BoundaryMassMode.LUMPED,
    ) -> None:
        self.mesh = mesh.volume if isinstance(mesh, BuiltVoxelHexMesh) else mesh
        self._hex = mesh if isinstance(mesh, BuiltVoxelHexMesh) else None
        self.material = material or MaterialProperties()
        self.surface_path = surface_path
        self.element_degree = element_degree
        self.boundary_mass_mode = boundary_mass_mode
        self.surface_bc_plan = None

    @property
    def is_voxel_hex(self) -> bool:
        return self._hex is not None or self.mesh.backend == MeshBackend.VOXEL_HEX

    def solve(
        self,
        shape: np.ndarray,
        bc: np.ndarray,
        load: np.ndarray,
        *,
        output_dir: Path,
        surface_path: Path | None = None,
        smooth_normal: np.ndarray | None = None,
        vox: np.ndarray | None = None,
    ) -> FEASolution:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        mesh_input = self._hex if self._hex is not None else self.mesh
        domain, _cell_tags, facet_tags = load_dolfinx_mesh(mesh_input)

        if self.is_voxel_hex:
            grid = vox if vox is not None else (self._hex.vox if self._hex else None)
            if grid is None:
                raise ValueError("Binary voxel grid required for hex mesh solve (vox=).")
            apply_voxel_bcs(self, grid, shape, bc, load)
        else:
            surface = surface_path or self.surface_path
            if surface is None:
                raise ValueError(
                    "Tagged VTK surface path required for tet mesh solves. "
                    "Pass surface_path= or use the voxel_hex backend."
                )
            tags = read_surface_tags(surface)
            apply_surface_bcs(self, tags, shape, load)

        assert self.surface_bc_plan is not None

        uh, compliance, Vu = solve_linear_elasticity(
            domain,
            facet_tags,
            self.surface_bc_plan,
            self.material,
            element_degree=self.element_degree,
        )

        disp_path = output_dir / "displacement.bp"
        from dolfinx import io
        from mpi4py import MPI

        with io.VTXWriter(MPI.COMM_WORLD, str(disp_path), [uh], engine="BP4") as vtx:
            vtx.write(0.0)

        geometry = assemble_boundary_l2_geometry(domain, facet_tags)
        vc = compliance_shape_derivative(
            uh,
            Vu,
            self.material,
            geometry,
            domain=domain,
            facet_tags=facet_tags,
            mass_mode=self.boundary_mass_mode,
            smooth_normal=smooth_normal,
        )

        from lib.fea.postprocess import (
            write_hex_free_boundary_vtp,
            write_shape_derivative_to_vtp,
        )

        shape_deriv_path = output_dir / "V_C.vtp"
        surface = surface_path or self.surface_path
        if surface is not None and not self.is_voxel_hex:
            write_shape_derivative_to_vtp(
                vc.normalized,
                geometry,
                domain,
                surface_path=surface,
                output_path=shape_deriv_path,
                field_name="V_C",
            )
        else:
            write_hex_free_boundary_vtp(
                vc.normalized,
                geometry,
                domain,
                facet_tags,
                output_path=shape_deriv_path,
                field_name="V_C",
            )

        return FEASolution(
            index=self.mesh.index,
            displacement_path=disp_path,
            compliance=compliance,
            shape_derivative_path=shape_deriv_path,
        )
