"""NITO boundary-condition and load conventions (shared by voxel and surface FEA)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import trimesh

LoadCheckPolicy = Literal["raise", "warn"]


def domain_scale(shape: np.ndarray) -> float:
    """Physical scale factor: max voxel count along any axis."""
    return float(np.max(np.asarray(shape, dtype=int)))


def physical_points(rows: np.ndarray, shape: np.ndarray) -> np.ndarray:
    """Map normalized NITO rows to physical coordinates."""
    dim = int(np.asarray(shape, dtype=int).size)
    rows = np.atleast_2d(np.asarray(rows, dtype=float))
    return rows[:, :dim] * domain_scale(shape)


def constraint_flags(row: np.ndarray, shape: np.ndarray) -> np.ndarray:
    """Per-axis BC flags (1 = fixed, 0 = free) from a NITO BC row."""
    dim = int(np.asarray(shape, dtype=int).size)
    row = np.asarray(row, dtype=float).ravel()
    return row[dim : 2 * dim]


def force_components(row: np.ndarray, shape: np.ndarray) -> np.ndarray:
    """Force components from a NITO load row."""
    dim = int(np.asarray(shape, dtype=int).size)
    row = np.asarray(row, dtype=float).ravel()
    return row[dim : 2 * dim]


def physical_points_3d(rows: np.ndarray, shape: np.ndarray) -> np.ndarray:
    """NITO normalized rows → 3D physical coordinates (2D rows get z=0)."""
    dim = int(np.asarray(shape, dtype=int).size)
    pts = physical_points(rows, shape)
    if dim == 2:
        return np.column_stack([pts, np.zeros(len(pts), dtype=float)])
    return np.asarray(pts, dtype=float)


def bc_anchor_positions(bc: np.ndarray, shape: np.ndarray) -> np.ndarray:
    """Physical positions of NITO BC rows with at least one constrained axis."""
    rows = np.atleast_2d(np.asarray(bc, dtype=float))
    anchors = [
        physical_points_3d(row.reshape(1, -1), shape)[0]
        for row in rows
        if np.any(constraint_flags(row, shape) > 0.5)
    ]
    if not anchors:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(anchors, dtype=float)


def load_anchor_positions(load: np.ndarray, shape: np.ndarray) -> np.ndarray:
    """Physical positions of NITO load rows with nonzero force."""
    rows = np.atleast_2d(np.asarray(load, dtype=float))
    anchors = [
        physical_points_3d(row.reshape(1, -1), shape)[0]
        for row in rows
        if np.any(np.abs(force_components(row, shape)) > 1e-12)
    ]
    if not anchors:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(anchors, dtype=float)


def marker_sphere_radius(shape: np.ndarray, *, fraction: float = 0.04) -> float:
    """Default glyph radius for BC/load markers in the design frame."""
    return fraction * float(np.max(np.asarray(shape, dtype=int)))


@dataclass(frozen=True)
class LoadSurfaceCheck:
    """Distance from one NITO load anchor to the extracted surface."""

    index: int
    nito_anchor: np.ndarray
    surface_point: np.ndarray
    distance: float


def load_surface_checks(
    mesh: trimesh.Trimesh,
    load_rows: np.ndarray,
    shape: np.ndarray,
    *,
    spacing=(1.0, 1.0, 1.0),
) -> list[LoadSurfaceCheck]:
    """
    Measure how far each NITO load lies from the closest point on ``mesh``.

    Loads are not stamped onto the surface; FEA applies them from ``loads.npy``.
    """
    from lib.converters.bc_patch import project_point_to_mesh

    rows = np.atleast_2d(np.asarray(load_rows, dtype=float))
    shape = np.asarray(shape, dtype=int)
    spacing = np.asarray(spacing, dtype=float)
    checks: list[LoadSurfaceCheck] = []
    for i, row in enumerate(rows):
        if not np.any(np.abs(force_components(row, shape)) > 1e-12):
            continue
        anchor = physical_points_3d(row.reshape(1, -1), shape)[0]
        surface_pt, dist, _ = project_point_to_mesh(mesh, anchor)
        checks.append(
            LoadSurfaceCheck(
                index=i,
                nito_anchor=np.asarray(anchor, dtype=float),
                surface_point=np.asarray(surface_pt, dtype=float),
                distance=float(dist),
            )
        )
    return checks


def load_surface_tolerance(
    shape: np.ndarray,
    spacing,
    max_distance_cells: float,
) -> float:
    """Physical distance tolerance for load-on-surface checks."""
    spacing = np.asarray(spacing, dtype=float)
    return float(max_distance_cells * float(np.max(spacing)))


def failed_load_surface_checks(
    checks: list[LoadSurfaceCheck],
    tol: float,
) -> list[LoadSurfaceCheck]:
    """Subset of checks farther than ``tol`` from the surface."""
    return [c for c in checks if c.distance > tol + 1e-12]


def format_load_surface_failure(
    bad: list[LoadSurfaceCheck],
    *,
    max_distance_cells: float,
    tol: float,
) -> str:
    """Multi-line message for out-of-tolerance NITO loads."""
    lines = [
        "NITO load(s) not on extracted surface "
        f"(tolerance={max_distance_cells} cells, {tol:.4g} physical units):",
    ]
    for c in bad:
        a = c.nito_anchor
        s = c.surface_point
        lines.append(
            f"  load {c.index}: nito=({a[0]:.4g}, {a[1]:.4g}, {a[2]:.4g}) "
            f"surface=({s[0]:.4g}, {s[1]:.4g}, {s[2]:.4g}) "
            f"distance={c.distance:.4g}",
        )
    return "\n".join(lines)


def enforce_loads_on_surface(
    mesh: trimesh.Trimesh,
    load_rows: np.ndarray,
    shape: np.ndarray,
    *,
    spacing=(1.0, 1.0, 1.0),
    max_distance_cells: float = 1.0,
    on_fail: LoadCheckPolicy = "raise",
) -> tuple[list[LoadSurfaceCheck], bool]:
    """
    Check NITO loads against the surface; raise or warn on failure.

    Returns ``(checks, failed)`` where ``failed`` is True when any load
    exceeds the tolerance.
    """
    tol = load_surface_tolerance(shape, spacing, max_distance_cells)
    checks = load_surface_checks(mesh, load_rows, shape, spacing=spacing)
    bad = failed_load_surface_checks(checks, tol)
    if bad:
        message = format_load_surface_failure(
            bad,
            max_distance_cells=max_distance_cells,
            tol=tol,
        )
        if on_fail == "raise":
            raise ValueError(message)
        warnings.warn(message, stacklevel=2)
        return checks, True
    return checks, False


def require_loads_on_surface(
    mesh: trimesh.Trimesh,
    load_rows: np.ndarray,
    shape: np.ndarray,
    *,
    spacing=(1.0, 1.0, 1.0),
    max_distance_cells: float = 1.0,
) -> list[LoadSurfaceCheck]:
    """
    Require every NITO load to lie within ``max_distance_cells`` of the surface.

    Raises ValueError when any load is farther than the tolerance. Returns all
    checks when the constraint is satisfied.
    """
    checks, _ = enforce_loads_on_surface(
        mesh,
        load_rows,
        shape,
        spacing=spacing,
        max_distance_cells=max_distance_cells,
        on_fail="raise",
    )
    return checks
