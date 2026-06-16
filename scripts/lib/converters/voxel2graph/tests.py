"""Self-tests for voxel2graph (round-trip and known-topology Betti numbers)."""

from __future__ import annotations

import sys
import unittest

import numpy as np

from lib.converters.voxel2graph.convert import (
    graph2voxel,
    surface_node_mask,
    surface_subgraph,
    surface_voxel_mask,
    voxel2graph,
)
from lib.converters.voxel2graph.fixtures import KNOWN_BETTI, all_fixtures
from lib.converters.voxel2graph.highlight import homology_highlight
from lib.converters.voxel2graph.homology import betti_numbers, betti_numbers_cubical
from lib.converters.voxel2graph.seeds import seed_mask_from_bcspecs
from lib.converters.bc_patch import BCSpec


def _random_volume(shape: tuple[int, int, int], fill: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(shape) < fill


class VoxelGraphTests(unittest.TestCase):
    def test_round_trip_random_32(self) -> None:
        for seed in range(8):
            vox = _random_volume((32, 32, 32), fill=0.3, seed=seed)
            data = voxel2graph(vox)
            back = graph2voxel(data).cpu().numpy()
            self.assertTrue(np.array_equal(back, vox > 0))

    def test_round_trip_empty(self) -> None:
        vox = np.zeros((8, 8, 8), dtype=bool)
        data = voxel2graph(vox)
        self.assertEqual(data.num_nodes, 0)
        self.assertTrue(graph2voxel(data).sum() == 0)

    def test_seed_mask_attached(self) -> None:
        vox = np.zeros((8, 8, 8), dtype=bool)
        vox[2:6, 2:6, 2:6] = True
        seed = np.zeros_like(vox, dtype=bool)
        seed[:, :, 2] = vox[:, :, 2]
        data = voxel2graph(vox, seed_mask=seed)
        self.assertTrue(hasattr(data, "is_seed"))
        self.assertEqual(int(data.is_seed.sum()), int(seed.sum()))

    def test_surface_is_induced_subgraph(self) -> None:
        vox = all_fixtures()["solid_block"]
        data = voxel2graph(vox)
        surface = surface_subgraph(data)
        node_mask = surface_node_mask(data)
        self.assertEqual(surface.num_nodes, int(node_mask.sum()))
        self.assertGreater(surface.num_nodes, 0)
        self.assertLess(surface.num_nodes, data.num_nodes)

    def test_surface_voxel_mask_matches_manual(self) -> None:
        vox = all_fixtures()["hollow_sphere_shell"]
        surface = surface_voxel_mask(vox).cpu().numpy()
        solid = vox > 0
        manual = np.zeros_like(solid)
        shape = solid.shape
        for idx in zip(*np.nonzero(solid)):
            for axis in range(3):
                for delta in (-1, 1):
                    nb = list(idx)
                    nb[axis] += delta
                    oob = nb[axis] < 0 or nb[axis] >= shape[axis]
                    empty = (not oob) and (not solid[tuple(nb)])
                    if oob or empty:
                        manual[idx] = True
                        break
                if manual[idx]:
                    break
        self.assertTrue(np.array_equal(surface, manual))

    def test_known_betti_numbers(self) -> None:
        for name, expected in KNOWN_BETTI.items():
            vox = all_fixtures()[name]
            b0_cubical, b1, b2 = betti_numbers_cubical(vox)
            self.assertEqual(
                (b0_cubical, b1, b2),
                expected,
                msg=f"{name} cubical Betti mismatch",
            )
            summary = betti_numbers(vox)
            self.assertEqual(summary.b0_graph, b0_cubical, msg=f"{name} graph/cubical b0")
            if name == "two_components":
                self.assertEqual(summary.b0_graph, 2)

    def test_highlight_fixtures(self) -> None:
        torus = homology_highlight(all_fixtures()["solid_torus"])
        self.assertEqual(torus.betti.b1, 1)
        self.assertGreater(int(torus.tunnel_solid.max()), 0)

        shell = homology_highlight(all_fixtures()["hollow_sphere_shell"])
        self.assertEqual(shell.betti.b2, 1)
        self.assertGreater(int(shell.cavity_void.max()), 0)
        self.assertGreater(int(shell.cavity_void.sum()), 0)

    def test_seed_mask_from_bcspec(self) -> None:
        vox = np.zeros((8, 8, 8), dtype=bool)
        vox[0:6, 1:7, 0:6] = True
        specs = [BCSpec.from_face(2, "min", np.array([1.0, 1.0, 0.0]))]
        mask = seed_mask_from_bcspecs(vox, specs)
        self.assertTrue(mask[:, :, 0].any())
        self.assertFalse(mask[:, :, 1:].any())


def run_self_test() -> None:
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    unittest.main()
