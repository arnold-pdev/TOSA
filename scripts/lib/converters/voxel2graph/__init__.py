"""Voxel grid ↔ PyG graph conversion and homology utilities."""

from lib.converters.voxel2graph.convert import (
    graph2voxel,
    surface_node_mask,
    surface_subgraph,
    surface_voxel_mask,
    voxel2graph,
)
from lib.converters.voxel2graph.fixtures import (
    KNOWN_BETTI,
    all_fixtures,
    hollow_sphere_shell,
    solid_block,
    solid_torus,
    two_components,
)
from lib.converters.voxel2graph.homology import (
    b0_from_graph,
    b0_from_voxel,
    betti_numbers,
    betti_numbers_cubical,
)
from lib.converters.voxel2graph.seeds import (
    seed_mask_from_bcspecs,
    seed_mask_from_nito,
    snap_plane_to_voxel_face,
)
from lib.converters.voxel2graph.highlight import HomologyHighlight, homology_highlight
from lib.converters.voxel2graph.types import BettiNumbers

__all__ = [
    "BettiNumbers",
    "KNOWN_BETTI",
    "all_fixtures",
    "b0_from_graph",
    "b0_from_voxel",
    "betti_numbers",
    "betti_numbers_cubical",
    "graph2voxel",
    "homology_highlight",
    "HomologyHighlight",
    "hollow_sphere_shell",
    "seed_mask_from_bcspecs",
    "seed_mask_from_nito",
    "snap_plane_to_voxel_face",
    "solid_block",
    "solid_torus",
    "surface_node_mask",
    "surface_subgraph",
    "surface_voxel_mask",
    "two_components",
    "voxel2graph",
]
