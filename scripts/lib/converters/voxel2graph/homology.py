"""Connectivity and cubical homology for voxel solids."""

from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from lib.converters.voxel2graph.convert import voxel2graph
from lib.converters.voxel2graph.types import BettiNumbers, as_bool_tensor


def b0_from_graph(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> int:
    """Connected components of the 6-connectivity graph (mechanical b₀)."""
    if num_nodes == 0:
        return 0
    row = edge_index[0].detach().cpu().numpy()
    col = edge_index[1].detach().cpu().numpy()
    adj = csr_matrix(
        (np.ones(len(row), dtype=np.float64), (row, col)),
        shape=(num_nodes, num_nodes),
    )
    n_comp = connected_components(adj, directed=False, return_labels=False)
    return int(n_comp)


def b0_from_voxel(vox: np.ndarray | torch.Tensor) -> int:
    """b₀ from the 6-connectivity graph built from ``vox``."""
    data = voxel2graph(vox)
    return b0_from_graph(data.edge_index, data.num_nodes)


def betti_numbers_cubical(
    vox: np.ndarray | torch.Tensor,
    *,
    threshold: float = 0.5,
) -> tuple[int, int, int]:
    """
    Cubical homology via GUDHI on the binary voxel field.

    Filled voxels are low filtration value (0); empty voxels are high (1). Betti
    numbers are evaluated on the sublevel set at ``threshold`` (default 0.5), so
    only filled cells are included.
    """
    import gudhi

    solid = as_bool_tensor(vox).cpu().numpy()
    cc = gudhi.CubicalComplex(
        dimensions=list(int(x) for x in solid.shape),
        top_dimensional_cells=(~solid).astype(np.float64).ravel(order="C"),
    )
    cc.compute_persistence()
    betti: list[int] = []
    for dim in range(3):
        intervals = cc.persistence_intervals_in_dimension(dim)
        betti.append(
            sum(
                1
                for birth, death in intervals
                if birth <= threshold < death
            )
        )
    return betti[0], betti[1], betti[2]


def betti_numbers(vox: np.ndarray | torch.Tensor) -> BettiNumbers:
    """
    Full Betti summary: graph b₀ plus cubical (b₀, b₁, b₂).

    Graph b₀ is the connectivity check used for mechanical guarantees; cubical
    b₁/b₂ count topological loops and enclosed voids.
    """
    data = voxel2graph(vox)
    b0_graph = b0_from_graph(data.edge_index, data.num_nodes)
    b0_cubical, b1, b2 = betti_numbers_cubical(vox)
    return BettiNumbers(
        b0_graph=b0_graph,
        b0_cubical=b0_cubical,
        b1=b1,
        b2=b2,
    )
