"""Vectorized voxel grid ↔ PyG graph conversion."""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

from lib.converters.voxel2graph.types import (
    as_bool_tensor,
    grid_shape,
    linear_index,
    normalize_coords,
)

_POSITIVE_OFFSETS = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.long)


def voxel2graph(
    vox: np.ndarray | torch.Tensor,
    *,
    seed_mask: np.ndarray | torch.Tensor | None = None,
) -> Data:
    """
    Build a full-volume 6-connectivity graph from a binary voxel grid.

    Nodes are filled voxels; edges are face-adjacent solid neighbors. The returned
    ``Data`` stores ``coord_index`` (dense grid → node id) for ``graph2voxel``.
    """
    solid = as_bool_tensor(vox)
    shape = tuple(int(x) for x in solid.shape)
    coords = torch.nonzero(solid, as_tuple=False)
    n_nodes = int(coords.shape[0])
    grid_size = int(np.prod(shape))

    idx_grid = torch.full((grid_size,), -1, dtype=torch.long)
    if n_nodes:
        node_ids = torch.arange(n_nodes, dtype=torch.long)
        idx_grid[linear_index(coords, shape)] = node_ids

    if n_nodes == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        x = torch.empty((0, 3), dtype=torch.float32)
        pos = torch.empty((0, 3), dtype=torch.float32)
    else:
        edges: list[torch.Tensor] = []
        shape_t = torch.tensor(shape, dtype=torch.long)
        for off in _POSITIVE_OFFSETS:
            nbr = coords + off
            in_bounds = (nbr < shape_t).all(dim=1)
            src = torch.arange(n_nodes, dtype=torch.long)[in_bounds]
            dst = idx_grid[linear_index(nbr[in_bounds], shape)]
            valid = dst >= 0
            edges.append(torch.stack([src[valid], dst[valid]], dim=0))

        edge_index = torch.cat(edges, dim=1)
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        x = normalize_coords(coords, shape)
        pos = coords.float()

    data = Data(x=x, edge_index=edge_index.contiguous(), pos=pos)
    data.grid_shape = shape
    data.coord_index = idx_grid
    if seed_mask is not None:
        seed = as_bool_tensor(seed_mask)
        if tuple(int(x) for x in seed.shape) != shape:
            raise ValueError(
                f"seed_mask shape {tuple(seed.shape)} does not match vox shape {shape}"
            )
        if n_nodes:
            data.is_seed = seed[coords[:, 0], coords[:, 1], coords[:, 2]]
        else:
            data.is_seed = torch.empty(0, dtype=torch.bool)
    return data


def graph2voxel(data: Data) -> torch.Tensor:
    """Recover the solid voxel grid encoded by a ``voxel2graph`` ``Data`` object."""
    if not hasattr(data, "coord_index"):
        raise ValueError("Data object is missing coord_index; was it built by voxel2graph?")
    shape = tuple(int(x) for x in data.grid_shape)
    grid_size = int(np.prod(shape))
    idx_grid = data.coord_index
    if idx_grid.numel() != grid_size:
        raise ValueError(
            f"coord_index length {idx_grid.numel()} does not match grid_shape {shape}"
        )
    return (idx_grid.reshape(shape) >= 0).contiguous()


def surface_voxel_mask(vox: np.ndarray | torch.Tensor) -> torch.Tensor:
    """
    Return a bool mask over the grid for frontier voxels.

    A filled voxel is on the surface when it has at least one empty face neighbor
    or lies on the domain boundary.
    """
    solid = as_bool_tensor(vox)
    interior = solid.clone()
    for axis in range(3):
        neg = torch.zeros_like(solid)
        pos = torch.zeros_like(solid)
        neg_sl = [slice(None)] * 3
        pos_sl = [slice(None)] * 3
        neg_sl[axis] = slice(1, None)
        pos_sl[axis] = slice(None, -1)
        neg[tuple(neg_sl)] = solid[tuple(pos_sl)]
        pos[tuple(pos_sl)] = solid[tuple(neg_sl)]
        interior &= neg & pos
    return solid & ~interior


def surface_node_mask(data: Data) -> torch.Tensor:
    """Node mask for the surface subgraph induced by ``surface_voxel_mask``."""
    vox = graph2voxel(data)
    surface = surface_voxel_mask(vox)
    coords = data.pos.long()
    if coords.numel() == 0:
        return torch.empty(0, dtype=torch.bool)
    return surface[coords[:, 0], coords[:, 1], coords[:, 2]]


def surface_subgraph(data: Data) -> Data:
    """Induced subgraph on frontier (surface) nodes; relabeled node indices."""
    mask = surface_node_mask(data)
    edge_index, _ = subgraph(
        mask,
        data.edge_index,
        relabel_nodes=True,
        num_nodes=data.num_nodes,
    )
    parent_idx = mask.nonzero(as_tuple=False).view(-1)
    out = Data(
        x=data.x[mask],
        edge_index=edge_index,
        pos=data.pos[mask],
    )
    out.grid_shape = data.grid_shape
    out.parent_node_index = parent_idx
    if hasattr(data, "is_seed"):
        out.is_seed = data.is_seed[mask]
    return out
