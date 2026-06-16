"""Linear elasticity solve with DOLFINx."""

from __future__ import annotations

import ufl
from dolfinx import fem
from dolfinx.fem import petsc as fem_petsc
from petsc4py import PETSc

from lib.fea.bcs import SurfaceBCPlan
from lib.fea.dolfinx_bcs import (
    add_point_forces_to_vector,
    build_dirichlet_bcs,
    build_elasticity_forms,
)
from lib.fea.forms import lame_parameters, strain, stress
from lib.fea.types import MaterialProperties


def solve_linear_elasticity(
    domain,
    facet_tags,
    bc_plan: SurfaceBCPlan,
    material: MaterialProperties,
    *,
    element_degree: int = 1,
    petsc_options: dict | None = None,
):
    """
    Solve isotropic linear elasticity and return (u_h, compliance).

    Loads are applied as nearest-node point forces (matching ATOMS).
    """
    gdim = domain.geometry.dim
    Vu = fem.functionspace(
        domain, ("Lagrange", element_degree, (gdim,))
    )
    bcs = build_dirichlet_bcs(Vu, facet_tags, bc_plan)
    a_form, L_form, loads = build_elasticity_forms(
        Vu, facet_tags, bc_plan, material
    )

    opts = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    if petsc_options:
        opts.update(petsc_options)

    A = fem_petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    b = fem_petsc.assemble_vector(L_form)
    add_point_forces_to_vector(b, Vu, loads)
    fem_petsc.apply_lifting(b, [a_form], bcs=[bcs])
    b.ghostUpdate(
        addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE
    )
    fem_petsc.set_bc(b, bcs)

    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)
    ksp.setFromOptions()
    if not petsc_options:
        ksp.setType(PETSc.KSP.Type.PREONLY)
        ksp.getPC().setType(PETSc.PC.Type.LU)
        ksp.getPC().setFactorSolverType("mumps")

    uh = fem.Function(Vu)
    ksp.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    mu, lmbda = lame_parameters(material, domain)
    dx = ufl.Measure("dx", domain=domain)
    from mpi4py import MPI

    compliance = fem.assemble_scalar(
        fem.form(ufl.inner(stress(uh, mu, lmbda, gdim), strain(uh)) * dx)
    )
    compliance = domain.comm.allreduce(compliance, op=MPI.SUM)

    return uh, float(compliance), Vu
