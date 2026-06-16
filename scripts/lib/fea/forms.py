"""UFL helpers for isotropic linear elasticity."""

from __future__ import annotations

import ufl
from dolfinx.fem import Constant
from petsc4py import PETSc


def lame_parameters(material, domain) -> tuple[Constant, Constant]:
    """Return (mu, lambda) Lamé constants from Young's modulus and Poisson ratio."""
    E = float(material.youngs_modulus)
    nu = float(material.poisson_ratio)
    mu = Constant(domain, PETSc.ScalarType(E / (2.0 * (1.0 + nu))))
    lmbda = Constant(domain, PETSc.ScalarType(E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))))
    return mu, lmbda


def strain(w) -> ufl.core.expr.Expr:
    return ufl.sym(ufl.grad(w))


def stress(w, mu, lmbda, gdim: int):
    return lmbda * ufl.tr(strain(w)) * ufl.Identity(gdim) + 2.0 * mu * strain(w)


def strain_energy_density(u, mu, lmbda, gdim: int):
    eps_u = strain(u)
    sig_u = stress(u, mu, lmbda, gdim)
    return 0.5 * ufl.inner(sig_u, eps_u)
