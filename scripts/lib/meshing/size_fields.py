"""Non-uniform mesh sizing from NITO sample data and surface geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lib.nito_physics import physical_points


@dataclass(frozen=True)
class MeshSizing:
    """Element-size controls for automatic meshing."""

    h_bulk: float
    h_boundary: float
    h_load: float
    h_feature: float
    load_patch_radius: float


def sizing_from_nito_sample(
    shape: np.ndarray,
    bc: np.ndarray,
    load: np.ndarray,
    *,
    h_bulk: float | None = None,
    boundary_factor: float = 0.25,
    load_factor: float = 0.2,
    feature_factor: float = 0.35,
    load_patch_fraction: float = 0.05,
) -> tuple[MeshSizing, np.ndarray, np.ndarray]:
    """
    Build sizing parameters and physical BC/load points for mesh size fields.

    Fine zones follow README automatic-meshing rules: boundary (shape-derivative
    trace), load patches (Saint-Venant near field), and thin features (geometry).
    """
    shape = np.asarray(shape, dtype=int)
    scale = float(np.max(shape))
    h0 = h_bulk if h_bulk is not None else scale / max(int(np.min(shape)), 1)

    bc_pts = physical_points(bc, shape)
    load_pts = physical_points(load, shape)

    return (
        MeshSizing(
            h_bulk=h0,
            h_boundary=h0 * boundary_factor,
            h_load=h0 * load_factor,
            h_feature=h0 * feature_factor,
            load_patch_radius=scale * load_patch_fraction,
        ),
        bc_pts,
        load_pts,
    )
