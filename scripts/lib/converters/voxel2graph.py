#!/usr/bin/env python3
"""
CLI entry for voxel2graph conversion and sanity checks.

    python scripts/lib/converters/voxel2graph.py --self-test
    python scripts/lib/converters/voxel2graph.py --index 119 --data-dir nito/Data/3D
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from lib.converters.voxel2graph import (  # noqa: E402
    betti_numbers,
    graph2voxel,
    seed_mask_from_nito,
    voxel2graph,
)
from lib.converters.voxel2graph.tests import run_self_test  # noqa: E402
from lib.nito_io import load_nito_arrays, load_sample, resolve_data_dir  # noqa: E402
from lib.volume_check import binarize_topology  # noqa: E402


def _cmd_self_test(_: argparse.Namespace) -> int:
    run_self_test()
    print("voxel2graph self-test passed")
    return 0


def _cmd_index(args: argparse.Namespace) -> int:
    data_dir = resolve_data_dir(args.data_dir, test=args.test)
    data = load_nito_arrays(data_dir)
    shape, rho, bc, _load, vf = load_sample(data, args.index)
    vox = binarize_topology(rho, shape)
    seed_mask = seed_mask_from_nito(vox, bc, shape)
    graph = voxel2graph(vox, seed_mask=seed_mask)
    recovered = graph2voxel(graph)
    betti = betti_numbers(vox)
    n_seed = int(graph.is_seed.sum()) if hasattr(graph, "is_seed") else 0
    print(f"index={args.index} shape={tuple(int(x) for x in shape)} vf={vf:.4f}")
    print(
        f"  nodes={graph.num_nodes} edges={graph.edge_index.shape[1]} "
        f"seeds={n_seed} round_trip={bool((recovered.numpy() == (vox > 0)).all())}"
    )
    print(
        f"  betti graph_b0={betti.b0_graph} cubical=({betti.b0_cubical}, "
        f"{betti.b1}, {betti.b2})"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Voxel grid to PyG graph conversion")
    parser.add_argument("--self-test", action="store_true", help="Run built-in tests")
    parser.add_argument("--index", type=int, help="NITO sample index to convert")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--test", action="store_true", help="Use NITO Data/Test split")
    args = parser.parse_args(argv)

    if args.self_test:
        return _cmd_self_test(args)
    if args.index is not None:
        return _cmd_index(args)
    parser.error("Specify --self-test or --index")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
