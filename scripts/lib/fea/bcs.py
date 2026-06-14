"""Map surface facet markers / NITO flags onto a DOLFINx problem."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lib.meshing.surface_tags import (
    BC_FACET_MARKER_OFFSET,
    FREE_FACET_MARKER,
    SurfaceTags,
    physical_groups_by_marker,
)
from lib.nito_physics import domain_scale, force_components, physical_points


@dataclass(frozen=True)
class DirichletPatch:
    """Dirichlet data on one BC patch (facet_marker = 2 + patch_id)."""

    patch_id: int
    facet_marker: int
    flags: np.ndarray
    triangle_indices: tuple[int, ...]


@dataclass(frozen=True)
class NeumannLoad:
    """Point/traction load specification (still applied via NITO load rows)."""

    position: np.ndarray
    force: np.ndarray


@dataclass(frozen=True)
class SurfaceBCPlan:
    """BC plan derived from a tagged VTK surface + NITO loads."""

    dirichlet: tuple[DirichletPatch, ...]
    loads: tuple[NeumannLoad, ...]
    free_facet_marker: int = FREE_FACET_MARKER


def plan_bcs_from_surface(
    tags: SurfaceTags,
    shape: np.ndarray,
    load: np.ndarray,
) -> SurfaceBCPlan:
    """
    Build a solver BC plan from VTK facet markers and NITO load rows.

    Dirichlet patches come from tagged BC facets; Neumann loads still use
    normalized NITO load rows scaled to physical coordinates.
    """
    groups = physical_groups_by_marker(tags)
    dirichlet: list[DirichletPatch] = []
    for marker, tris in sorted(groups.items()):
        if marker == FREE_FACET_MARKER:
            continue
        patch_id = int(marker) - BC_FACET_MARKER_OFFSET
        if patch_id < 0 or patch_id >= len(tags.patches):
            continue
        dirichlet.append(
            DirichletPatch(
                patch_id=patch_id,
                facet_marker=int(marker),
                flags=np.asarray(tags.patches[patch_id].flags, dtype=float),
                triangle_indices=tuple(tris),
            )
        )

    load_rows = np.atleast_2d(np.asarray(load, dtype=float))
    loads = tuple(
        NeumannLoad(
            position=physical_points(row, shape)[0],
            force=force_components(row, shape) * domain_scale(shape),
        )
        for row in load_rows
    )
    return SurfaceBCPlan(dirichlet=tuple(dirichlet), loads=loads)


def apply_surface_bcs(
    problem: object,
    tags: SurfaceTags,
    shape: np.ndarray,
    load: np.ndarray,
) -> SurfaceBCPlan:
    """
    Attach Dirichlet/Neumann data from a tagged VTK surface to a FEA problem.

    Returns the BC plan for inspection; DOLFINx assembly is completed when
    LinearElasticityProblem.solve is implemented.
    """
    plan = plan_bcs_from_surface(tags, shape, load)
    setattr(problem, "surface_bc_plan", plan)
    return plan


def apply_nito_bcs(
    problem: object,
    shape: np.ndarray,
    bc: np.ndarray,
    load: np.ndarray,
) -> None:
    """
    Legacy entry: apply BCs from raw NITO rows (no VTK surface tags).

    Prefer apply_surface_bcs when the mesh was built from a voxel2surf .vtp.
    """
    raise NotImplementedError(
        "Raw NITO row BC mapping is not implemented. "
        "Use apply_surface_bcs with a tagged VTK surface from voxel2surf."
    )
