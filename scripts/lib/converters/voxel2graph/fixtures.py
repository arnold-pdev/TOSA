"""Known-topology voxel fixtures for homology and round-trip tests."""

from __future__ import annotations

import numpy as np


def solid_block(
    shape: tuple[int, int, int] = (16, 16, 16),
    *,
    margin: int = 2,
) -> np.ndarray:
    """Simply connected solid block → cubical Betti [1, 0, 0]."""
    vox = np.zeros(shape, dtype=bool)
    vox[margin:-margin, margin:-margin, margin:-margin] = True
    return vox


def two_components(shape: tuple[int, int, int] = (16, 16, 16)) -> np.ndarray:
    """Two separated solids → b₀ = 2."""
    vox = np.zeros(shape, dtype=bool)
    vox[2:6, 2:14, 2:14] = True
    vox[10:14, 2:14, 2:14] = True
    return vox


def solid_torus(shape: tuple[int, int, int] = (24, 24, 24)) -> np.ndarray:
    """
    Axis-aligned ring (solid torus) in the xy mid-plane → cubical Betti [1, 1, 0].
    """
    vox = np.zeros(shape, dtype=bool)
    cx, cy = (shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0
    cz = shape[2] // 2
    major_r = min(shape[0], shape[1]) * 0.28
    tube_r = min(shape[0], shape[1]) * 0.10
    for i in range(shape[0]):
        for j in range(shape[1]):
            dist = np.hypot(i - cx, j - cy)
            if abs(dist - major_r) <= tube_r:
                vox[i, j, max(cz - 1, 0) : min(cz + 2, shape[2])] = True
    return vox


def hollow_sphere_shell(shape: tuple[int, int, int] = (24, 24, 24)) -> np.ndarray:
    """Spherical shell with enclosed void → cubical Betti [1, 0, 1]."""
    vox = np.zeros(shape, dtype=bool)
    cx, cy, cz = [(n - 1) / 2.0 for n in shape]
    outer_r = min(shape) * 0.42
    inner_r = min(shape) * 0.28
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                r = np.sqrt((i - cx) ** 2 + (j - cy) ** 2 + (k - cz) ** 2)
                if inner_r <= r <= outer_r:
                    vox[i, j, k] = True
    return vox


KNOWN_BETTI: dict[str, tuple[int, int, int]] = {
    "solid_block": (1, 0, 0),
    "two_components": (2, 0, 0),
    "solid_torus": (1, 1, 0),
    "hollow_sphere_shell": (1, 0, 1),
}


def all_fixtures() -> dict[str, np.ndarray]:
    return {
        "solid_block": solid_block(),
        "two_components": two_components(),
        "solid_torus": solid_torus(),
        "hollow_sphere_shell": hollow_sphere_shell(),
    }
