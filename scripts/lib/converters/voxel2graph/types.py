"""Shared types for voxel ↔ graph conversion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def as_bool_tensor(vox: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Coerce a voxel grid to a contiguous 3D bool tensor."""
    if isinstance(vox, torch.Tensor):
        t = vox.detach()
        if t.dtype != torch.bool:
            t = t > 0
        return t.reshape(-1).reshape(tuple(int(x) for x in t.shape)).contiguous()
    arr = np.asarray(vox)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D voxel grid, got shape {arr.shape}")
    return torch.from_numpy((arr > 0).astype(np.bool_)).contiguous()


def grid_shape(vox: np.ndarray | torch.Tensor) -> tuple[int, int, int]:
    shape = tuple(int(x) for x in np.asarray(vox).shape)
    if len(shape) != 3:
        raise ValueError(f"Expected a 3D voxel grid, got shape {shape}")
    return shape


def linear_index(coords: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    """Flatten (i, j, k) cell indices for a C-order ``shape`` grid."""
    nx, ny, nz = shape
    return (coords[:, 0] * ny + coords[:, 1]) * nz + coords[:, 2]


def normalize_coords(coords: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    """Map integer cell indices to [0, 1] using ``coord / (size - 1)`` per axis."""
    denom = torch.tensor(
        [max(n - 1, 1) for n in shape],
        dtype=coords.dtype,
        device=coords.device,
    )
    return coords.float() / denom


@dataclass(frozen=True)
class BettiNumbers:
    """Homology counts for a 3D voxel solid."""

    b0_graph: int
    b0_cubical: int
    b1: int
    b2: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.b0_graph, self.b0_cubical, self.b1, self.b2

    def as_cubical_tuple(self) -> tuple[int, int, int]:
        return self.b0_cubical, self.b1, self.b2
