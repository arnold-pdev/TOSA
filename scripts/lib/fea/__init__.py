"""FEniCSx linear elasticity and shape-derivative post-processing."""

from lib.fea.bcs import (
    SurfaceBCPlan,
    apply_surface_bcs,
    apply_voxel_bcs,
    plan_bcs_from_surface,
    plan_bcs_from_voxel,
)
from lib.fea.problem import LinearElasticityProblem
from lib.fea.shape_derivative import (
    BoundaryL2Geometry,
    BoundaryMassMode,
    ProjectedShapeDerivative,
    compliance_shape_derivative,
    l2_inner_product,
    project_shape_derivative,
)
from lib.fea.types import FEASolution, MaterialProperties, ShapeDerivativeOnSurface

__all__ = [
    "BoundaryL2Geometry",
    "BoundaryMassMode",
    "FEASolution",
    "LinearElasticityProblem",
    "MaterialProperties",
    "ProjectedShapeDerivative",
    "ShapeDerivativeOnSurface",
    "SurfaceBCPlan",
    "apply_surface_bcs",
    "apply_voxel_bcs",
    "compliance_shape_derivative",
    "l2_inner_product",
    "plan_bcs_from_surface",
    "plan_bcs_from_voxel",
    "project_shape_derivative",
]
