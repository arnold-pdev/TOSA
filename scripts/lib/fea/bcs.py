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


def plan_bcs_from_voxel(
    bc: np.ndarray,
    shape: np.ndarray,
    load: np.ndarray,
    vox: np.ndarray,
) -> SurfaceBCPlan:
    """
    Build a solver BC plan from NITO rows and a binary voxel grid.

    Dirichlet patches come from ``build_patches_from_nito``; loads use the same
    nearest-node scaling as the surface path.
    """
    from lib.converters.bc_patch import build_patches_from_nito

    origin = np.zeros(3, dtype=float)
    spacing = np.ones(3, dtype=float)
    patches = build_patches_from_nito(vox, bc, shape, origin, spacing)

    dirichlet: list[DirichletPatch] = []
    for patch_id, patch in enumerate(patches):
        if not patch.is_bc:
            continue
        dirichlet.append(
            DirichletPatch(
                patch_id=patch_id,
                facet_marker=BC_FACET_MARKER_OFFSET + patch_id,
                flags=np.asarray(patch.flags, dtype=float),
                triangle_indices=(),
            )
        )

    load_rows = np.atleast_2d(np.asarray(load, dtype=float))
    loads = tuple(
        NeumannLoad(
            position=physical_points(row, shape)[0],
            force=force_components(row, shape) * domain_scale(shape),
        )
        for row in load_rows
        if np.any(np.abs(force_components(row, shape)) > 1e-12)
    )
    return SurfaceBCPlan(dirichlet=tuple(dirichlet), loads=loads)


def apply_voxel_bcs(
    problem: object,
    vox: np.ndarray,
    shape: np.ndarray,
    bc: np.ndarray,
    load: np.ndarray,
) -> SurfaceBCPlan:
    """Attach NITO BC/load data from a binary voxel grid (hex mesh path)."""
    plan = plan_bcs_from_voxel(bc, shape, load, vox)
    setattr(problem, "surface_bc_plan", plan)
    return plan


def apply_nito_bcs(
    problem: object,
    shape: np.ndarray,
    bc: np.ndarray,
    load: np.ndarray,
    *,
    vox: np.ndarray | None = None,
) -> SurfaceBCPlan:
    """
    Apply BCs from raw NITO rows.

    Pass ``vox`` (binary grid) for the hex-mesh path; omit for VTK-tagged surfaces.
    """
    if vox is not None:
        return apply_voxel_bcs(problem, vox, shape, bc, load)
    raise ValueError(
        "Hex mesh path requires binary vox. "
        "Use apply_surface_bcs with a tagged VTK surface, or pass vox= to apply_nito_bcs."
    )
