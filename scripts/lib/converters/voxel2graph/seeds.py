"""BC seed tagging on voxel grids (prescribed domain-face planes)."""

from __future__ import annotations

import numpy as np

from lib.converters.bc_patch import BCSpec, bcspecs_from_nito


def seed_mask_from_bcspecs(
    vox: np.ndarray,
    specs: list[BCSpec],
) -> np.ndarray:
    """
    Tag solid voxels on prescribed axis-aligned domain-face BC slabs.

    The contact plane is data, not inferred: each ``BCSpec`` names a domain
    boundary slab (``axis`` + ``min``/``max`` side).
    """
    solid = np.asarray(vox) > 0
    if solid.ndim != 3:
        raise ValueError(f"Expected a 3D voxel grid, got shape {solid.shape}")
    mask = np.zeros(solid.shape, dtype=bool)
    shape = solid.shape
    for spec in specs:
        slab = 0 if spec.side == "min" else shape[spec.axis] - 1
        idx: list[slice | int] = [slice(None)] * 3
        idx[spec.axis] = slab
        mask[tuple(idx)] |= solid[tuple(idx)]
    return mask


def seed_mask_from_nito(
    vox: np.ndarray,
    bc: np.ndarray,
    shape: np.ndarray,
) -> np.ndarray:
    """Build a seed mask from NITO ``boundary_conditions`` rows."""
    specs = bcspecs_from_nito(bc, shape)
    return seed_mask_from_bcspecs(vox, specs)


def snap_plane_to_voxel_face(
    plane_coord: float,
    *,
    origin: float,
    spacing: float,
    shape_axis: int,
    side: str,
) -> float:
    """
    Snap a physical plane coordinate to the nearest voxel face on a domain boundary.

    For faces already on the design-domain boundary (``side`` is ``min`` or
    ``max``), this returns the exact face coordinate; otherwise the nearest
    interior voxel interface is used.
    """
    if side == "min":
        return float(origin)
    if side == "max":
        return float(origin + shape_axis * spacing)
    face_coords = origin + np.arange(shape_axis + 1, dtype=float) * spacing
    idx = int(np.argmin(np.abs(face_coords - plane_coord)))
    return float(face_coords[idx])
