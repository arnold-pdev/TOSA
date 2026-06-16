"""Apply NITO / VTK surface BC plans in DOLFINx."""

from __future__ import annotations

import numpy as np
from dolfinx import fem
from petsc4py import PETSc

from lib.fea.bcs import NeumannLoad, SurfaceBCPlan


def build_dirichlet_bcs(Vu, facet_tags, bc_plan: SurfaceBCPlan) -> list:
    """
    Component-wise Dirichlet BCs from tagged BC patches.

    Each patch carries NITO ``flags`` (1 = fixed, 0 = free) on x/y/z.
    """
    mesh = Vu.mesh
    gdim = mesh.geometry.dim
    fdim = gdim - 1
    bcs: list = []

    for patch in bc_plan.dirichlet:
        facets = facet_tags.find(patch.facet_marker)
        if facets.size == 0:
            continue
        flags = np.asarray(patch.flags, dtype=float).ravel()[:gdim]
        for comp in range(gdim):
            if flags[comp] <= 0.5:
                continue
            dofs = fem.locate_dofs_topological(Vu.sub(comp), fdim, facets)
            bcs.append(
                fem.dirichletbc(PETSc.ScalarType(0.0), dofs, Vu.sub(comp))
            )
    return bcs


def nearest_node_indices(mesh, points: np.ndarray) -> np.ndarray:
    """Map physical points to nearest mesh vertex indices."""
    coords = np.asarray(mesh.geometry.x, dtype=float)
    points = np.atleast_2d(np.asarray(points, dtype=float))
    out = np.empty(len(points), dtype=np.int64)
    for i, pt in enumerate(points):
        out[i] = int(np.argmin(np.linalg.norm(coords - pt, axis=1)))
    return out


def add_point_forces_to_vector(
    b: PETSc.Vec,
    Vu,
    loads: tuple[NeumannLoad, ...],
) -> None:
    """
    Inject concentrated nodal forces (ATOMS-style nearest-node loads).

    Forces are added directly to the assembled load vector ``b``.
    """
    if not loads:
        return

    mesh = Vu.mesh
    gdim = mesh.geometry.dim
    coords = np.asarray(mesh.geometry.x, dtype=float)
    arr = b.getArray()

    for load in loads:
        node = int(nearest_node_indices(mesh, load.position.reshape(1, -1))[0])
        pt = coords[node]
        force = np.asarray(load.force, dtype=float).ravel()[:gdim]

        def _near(x, p=pt):
            return np.linalg.norm(x - p, axis=1) < 1e-12

        for comp in range(gdim):
            dofs = fem.locate_dofs_geometrical(Vu.sub(comp), _near)
            if dofs.size == 0:
                continue
            arr[dofs[0]] += force[comp]
    b.setArray(arr)


def build_elasticity_forms(
    Vu,
    facet_tags,
    bc_plan: SurfaceBCPlan,
    material,
    *,
    loads: tuple[NeumannLoad, ...] | None = None,
):
    """Weak form (a, L) for isotropic linear elasticity with point loads."""
    import ufl

    from lib.fea.forms import lame_parameters, strain, stress

    domain = Vu.mesh
    gdim = domain.geometry.dim
    mu, lmbda = lame_parameters(material, domain)
    u_tr = ufl.TrialFunction(Vu)
    v_te = ufl.TestFunction(Vu)
    dx = ufl.Measure("dx", domain=domain)

    a_form = ufl.inner(stress(u_tr, mu, lmbda, gdim), strain(v_te)) * dx
    zero = ufl.as_vector([0.0] * gdim)
    L_form = ufl.inner(zero, v_te) * dx

    return fem.form(a_form), fem.form(L_form), loads or bc_plan.loads
