"""Map cubical persistence features to voxel highlights for visualization."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from lib.converters.voxel2graph.homology import betti_numbers
from lib.converters.voxel2graph.types import BettiNumbers


def flat_to_ijk(index: int, shape: tuple[int, int, int]) -> tuple[int, int, int]:
    """C-order flat index → ``(i, j, k)`` voxel coordinates."""
    nx, ny, nz = shape
    idx = int(index)
    k = idx % nz
    j = (idx // nz) % ny
    i = idx // (ny * nz)
    return i, j, k


def ijk_to_flat(ijk: tuple[int, int, int], shape: tuple[int, int, int]) -> int:
    i, j, k = ijk
    _, ny, nz = shape
    return (int(i) * ny + int(j)) * nz + int(k)


def _as_solid(vox: np.ndarray) -> np.ndarray:
    solid = np.asarray(vox) > 0
    if solid.ndim != 3:
        raise ValueError(f"Expected a 3D voxel grid, got shape {solid.shape}")
    return solid


def _flood_void_component(
    void: np.ndarray,
    seed: tuple[int, int, int],
) -> tuple[np.ndarray, bool]:
    """6-connected void component from ``seed``; ``True`` if it touches the domain box."""
    shape = void.shape
    if not void[seed]:
        return np.zeros(shape, dtype=bool), False
    visited = np.zeros(shape, dtype=bool)
    touches_boundary = False
    q: deque[tuple[int, int, int]] = deque([seed])
    while q:
        cell = q.popleft()
        if visited[cell]:
            continue
        visited[cell] = True
        for axis in range(3):
            for delta in (-1, 1):
                nb = list(cell)
                nb[axis] += delta
                if nb[axis] < 0 or nb[axis] >= shape[axis]:
                    touches_boundary = True
                    continue
                nbt = tuple(nb)
                if void[nbt] and not visited[nbt]:
                    q.append(nbt)
    return visited, touches_boundary


def _dilate_solid_seeds(
    solid: np.ndarray,
    seeds: list[tuple[int, int, int]],
    *,
    radius: int,
) -> np.ndarray:
    """Grow solid seed voxels by ``radius`` face steps within the solid."""
    if radius <= 0 or not seeds:
        mask = np.zeros(solid.shape, dtype=bool)
        for seed in seeds:
            if solid[seed]:
                mask[seed] = True
        return mask

    active = np.zeros(solid.shape, dtype=bool)
    for seed in seeds:
        if solid[seed]:
            active[seed] = True

    for _ in range(radius):
        grown = active.copy()
        for idx in zip(*np.nonzero(active)):
            cell = tuple(int(x) for x in idx)
            for axis in range(3):
                for delta in (-1, 1):
                    nb = list(cell)
                    nb[axis] += delta
                    if (
                        0 <= nb[axis] < solid.shape[axis]
                        and solid[tuple(nb)]
                    ):
                        grown[tuple(nb)] = True
        active = grown
    return active


@dataclass
class HomologyHighlight:
    """Per-voxel labels for tunnel and cavity persistence features."""

    betti: BettiNumbers
    tunnel_solid: np.ndarray = field(repr=False)  # int, 0 = none, 1..b1
    cavity_void: np.ndarray = field(repr=False)  # int, 0 = none, 1..b2
    tunnel_seeds: list[tuple[int, int, int]] = field(default_factory=list)
    cavity_seeds: list[tuple[int, int, int]] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(x) for x in self.tunnel_solid.shape)

    def summary(self) -> str:
        b = self.betti
        return (
            f"b0={b.b0_graph}  cubical=({b.b0_cubical}, {b.b1}, {b.b2})  "
            f"tunnels={max(int(self.tunnel_solid.max()), 0)}  "
            f"cavities={max(int(self.cavity_void.max()), 0)}"
        )


def homology_highlight(
    vox: np.ndarray,
    *,
    threshold: float = 0.5,
    tunnel_dilation: int = 2,
) -> HomologyHighlight:
    """
    Label voxels that carry active cubical homology features at ``threshold``.

    - **Tunnels (b₁):** dilated solid neighborhood around each dim-1 birth cell
      from GUDHI ``cofaces_of_persistence_pairs``.
    - **Cavities (b₂):** enclosed void components seeded from dim-2 death cells.
    """
    import gudhi

    solid = _as_solid(vox)
    shape = tuple(int(x) for x in solid.shape)
    void = ~solid
    tunnel_solid = np.zeros(shape, dtype=np.int32)
    cavity_void = np.zeros(shape, dtype=np.int32)
    tunnel_seeds: list[tuple[int, int, int]] = []
    cavity_seeds: list[tuple[int, int, int]] = []

    cc = gudhi.CubicalComplex(
        dimensions=list(shape),
        top_dimensional_cells=(~solid).astype(np.float64).ravel(order="C"),
    )
    cc.compute_persistence()
    regular, _essential = cc.cofaces_of_persistence_pairs()

    tunnel_id = 0
    intervals1 = cc.persistence_intervals_in_dimension(1)
    pairs1 = regular[1] if len(regular) > 1 and regular[1] is not None else np.empty((0, 2))
    for row, (birth, death) in enumerate(intervals1):
        if not (birth <= threshold < death):
            continue
        if row >= len(pairs1):
            continue
        pos_flat = int(pairs1[row, 0])
        seed = flat_to_ijk(pos_flat, shape)
        tunnel_seeds.append(seed)
        tunnel_id += 1
        region = _dilate_solid_seeds(solid, [seed], radius=tunnel_dilation)
        tunnel_solid[region] = tunnel_id

    cavity_id = 0
    intervals2 = cc.persistence_intervals_in_dimension(2)
    pairs2 = regular[2] if len(regular) > 2 and regular[2] is not None else np.empty((0, 2))
    for row, (birth, death) in enumerate(intervals2):
        if not (birth <= threshold < death):
            continue
        if row >= len(pairs2):
            continue
        neg_flat = int(pairs2[row, 1])
        seed = flat_to_ijk(neg_flat, shape)
        cavity_seeds.append(seed)
        component, touches_boundary = _flood_void_component(void, seed)
        if touches_boundary:
            continue
        cavity_id += 1
        cavity_void[component] = cavity_id

    summary = betti_numbers(solid)
    return HomologyHighlight(
        betti=summary,
        tunnel_solid=tunnel_solid,
        cavity_void=cavity_void,
        tunnel_seeds=tunnel_seeds,
        cavity_seeds=cavity_seeds,
    )
