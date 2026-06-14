"""FEniCSx linear elasticity and shape-derivative post-processing."""

from lib.fea.bcs import SurfaceBCPlan, apply_surface_bcs, plan_bcs_from_surface
from lib.fea.problem import LinearElasticityProblem
from lib.fea.types import FEASolution, MaterialProperties

__all__ = [
    "FEASolution",
    "LinearElasticityProblem",
    "MaterialProperties",
    "SurfaceBCPlan",
    "apply_surface_bcs",
    "plan_bcs_from_surface",
]
