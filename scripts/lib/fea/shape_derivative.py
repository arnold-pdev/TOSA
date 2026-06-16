"""
Volume-form shape derivative and L²(Γ) boundary projection.

Pipeline (per objective k):
  1. assemble b_i = ∫_Ω S(u, p) : ∇θ_i dx   (distributed / Eshelby form)
  2. project onto the free boundary in L²(Γ):  M_Γ g = b_Γ
  3. mean-correct and L²(Γ)-normalize

For compliance, p = u (self-adjoint). J1/J2/J3 plug in their own adjoints later.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem import petsc as fem_petsc
from petsc4py import PETSc

from lib.fea.forms import lame_parameters, strain, stress
from lib.fea.types import MaterialProperties
from lib.meshing.surface_tags import FREE_FACET_MARKER


class BoundaryMassMode(str, Enum):
    LUMPED = "lumped"
    CONSISTENT = "consistent"


@dataclass(frozen=True)
class BoundaryL2Geometry:
    """Nodal L²(Γ) weights and normals on the free boundary."""

    nodal_area: np.ndarray
    nodal_normal: np.ndarray
    free_mask: np.ndarray
    mass_matrix: object | None = None


@dataclass(frozen=True)
class ProjectedShapeDerivative:
    """Scalar shape-derivative density on volume mesh nodes (0 off free boundary)."""

    values: np.ndarray
    normalized: np.ndarray
    geometry: BoundaryL2Geometry
    sign_applied: float = -1.0


def eshelby_tensor(u, p, material: MaterialProperties, gdim: int):
    """Energy-momentum tensor S = w I - (∇u)ᵀ σ with w = ½ σ:ε."""
    mu, lmbda = lame_parameters(material, u.function_space.mesh)
    w = 0.5 * ufl.inner(stress(u, mu, lmbda, gdim), strain(u))
    sig_p = stress(p, mu, lmbda, gdim)
    return w * ufl.Identity(gdim) - ufl.dot(ufl.grad(u).T, sig_p)


def assemble_volume_shape_rhs(
    u,
    p,
    Vu,
    material: MaterialProperties,
) -> np.ndarray:
    """
    Assemble b_i = ∫_Ω S : ∇θ_i dx against the vector test space.

    Returns (n_nodes, gdim) array of nodal values (blocked layout).
    """
    domain = u.function_space.mesh
    gdim = domain.geometry.dim
    Sigma = eshelby_tensor(u, p, material, gdim)
    theta = ufl.TestFunction(Vu)
    dx = ufl.Measure("dx", domain=domain)
    form = fem.form(ufl.inner(Sigma, ufl.grad(theta)) * dx)
    bvec = fem_petsc.assemble_vector(form)
    bvec.ghostUpdate(
        addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE
    )
    return bvec.getArray().real.reshape(-1, gdim)


def assemble_boundary_l2_geometry(
    domain,
    facet_tags,
    *,
    free_marker: int = FREE_FACET_MARKER,
) -> BoundaryL2Geometry:
    """
    Nodal area a_i = ∫_{Γ_free} φ_i ds and area-weighted normals on the free boundary.

    Nodes with a_i > 0 are exactly the free-boundary support of P1 scalars.
    """
    gdim = domain.geometry.dim
    Vs = fem.functionspace(domain, ("Lagrange", 1))
    Vu = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))
    ds_f = ufl.Measure(
        "ds", domain=domain, subdomain_data=facet_tags, subdomain_id=free_marker
    )

    av = fem_petsc.assemble_vector(fem.form(ufl.TestFunction(Vs) * ds_f))
    av.ghostUpdate(
        addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE
    )
    nodal_area = av.getArray().real.copy()

    nrm = ufl.FacetNormal(domain)
    Nv = fem_petsc.assemble_vector(
        fem.form(ufl.inner(nrm, ufl.TestFunction(Vu)) * ds_f)
    )
    Nv.ghostUpdate(
        addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE
    )
    N = Nv.getArray().real.reshape(-1, gdim)

    free_mask = nodal_area > 1e-13
    nodal_normal = np.zeros_like(N)
    norms = np.linalg.norm(N[free_mask], axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    nodal_normal[free_mask] = N[free_mask] / norms

    return BoundaryL2Geometry(
        nodal_area=nodal_area,
        nodal_normal=nodal_normal,
        free_mask=free_mask,
        mass_matrix=None,
    )


def assemble_consistent_boundary_mass(
    domain,
    facet_tags,
    *,
    free_marker: int = FREE_FACET_MARKER,
):
    """Sparse surface mass matrix M_Γ on P1 nodes (includes zeros off Γ_free)."""
    import scipy.sparse as sp

    Vs = fem.functionspace(domain, ("Lagrange", 1))
    ds_f = ufl.Measure(
        "ds", domain=domain, subdomain_data=facet_tags, subdomain_id=free_marker
    )
    Mform = fem.form(ufl.TrialFunction(Vs) * ufl.TestFunction(Vs) * ds_f)
    M = fem_petsc.assemble_matrix(Mform)
    M.assemble()
    ai, aj, av = M.getValuesCSR()
    return sp.csr_matrix((av, aj, ai))


def project_volume_to_boundary_l2(
    b: np.ndarray,
    geometry: BoundaryL2Geometry,
    *,
    mode: BoundaryMassMode = BoundaryMassMode.LUMPED,
    domain=None,
    facet_tags=None,
    free_marker: int = FREE_FACET_MARKER,
    smooth_normal: np.ndarray | None = None,
) -> np.ndarray:
    """
    L²(Γ) Riesz projection: find g such that ∫_Γ g (V·n) ds = ∫_Ω S:∇V dx.

    ``b`` is the volume-form RHS (n_nodes, gdim). Returns scalar nodal density g.
    """
    free = geometry.free_mask
    nhat = geometry.nodal_normal.copy()
    if smooth_normal is not None:
        nhat[free] = smooth_normal[free]

    flux = np.einsum("ij,ij->i", b, nhat)
    g = np.zeros(geometry.nodal_area.shape[0], dtype=float)

    if mode == BoundaryMassMode.LUMPED:
        g[free] = flux[free] / geometry.nodal_area[free]
        return g

    if domain is None or facet_tags is None:
        raise ValueError("domain and facet_tags required for consistent mass projection")

    import scipy.sparse.linalg as spla

    Msp = assemble_consistent_boundary_mass(
        domain, facet_tags, free_marker=free_marker
    )
    idx = np.where(free)[0]
    rhs = flux[idx]
    g[idx] = spla.spsolve(Msp[np.ix_(idx, idx)].tocsc(), rhs)
    return g


def mean_correct_and_normalize_l2(
    g: np.ndarray,
    geometry: BoundaryL2Geometry,
) -> np.ndarray:
    """Volume-preserving mean correction and L²(Γ) normalization (lumped weights)."""
    free = geometry.free_mask
    A = geometry.nodal_area[free]
    gm = g[free].copy()
    gm -= np.sum(gm * A) / np.sum(A)
    norm = np.sqrt(np.sum(gm**2 * A))
    if norm > 0.0:
        gm /= norm
    out = g.copy()
    out[free] = gm
    return out


def l2_inner_product(
    f: np.ndarray,
    g: np.ndarray,
    geometry: BoundaryL2Geometry,
) -> float:
    """⟨f, g⟩_{L²(Γ)} using lumped nodal area weights."""
    free = geometry.free_mask
    A = geometry.nodal_area[free]
    return float(np.sum(f[free] * g[free] * A))


def project_shape_derivative(
    u,
    p,
    Vu,
    material: MaterialProperties,
    geometry: BoundaryL2Geometry,
    *,
    domain,
    facet_tags,
    free_marker: int = FREE_FACET_MARKER,
    mass_mode: BoundaryMassMode = BoundaryMassMode.LUMPED,
    smooth_normal: np.ndarray | None = None,
    apply_compliance_sign: bool = True,
) -> ProjectedShapeDerivative:
    """
    Full volume-form → L²(Γ) pipeline for one objective.

    ``apply_compliance_sign``: flip once so g_C = -σ:ε (not +w from Eshelby flux).
    """
    b = assemble_volume_shape_rhs(u, p, Vu, material)
    g = project_volume_to_boundary_l2(
        b,
        geometry,
        mode=mass_mode,
        domain=domain,
        facet_tags=facet_tags,
        free_marker=free_marker,
        smooth_normal=smooth_normal,
    )
    sign = -1.0 if apply_compliance_sign else 1.0
    g = sign * g
    g_norm = mean_correct_and_normalize_l2(g, geometry)
    return ProjectedShapeDerivative(
        values=g,
        normalized=g_norm,
        geometry=geometry,
        sign_applied=sign,
    )


def compliance_shape_derivative(
    u,
    Vu,
    material: MaterialProperties,
    geometry: BoundaryL2Geometry,
    *,
    domain,
    facet_tags,
    **kwargs,
) -> ProjectedShapeDerivative:
    """Compliance V_C: self-adjoint, so p = u."""
    return project_shape_derivative(
        u,
        u,
        Vu,
        material,
        geometry,
        domain=domain,
        facet_tags=facet_tags,
        **kwargs,
    )
