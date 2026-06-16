#!/usr/bin/env python3
"""
CLI: compliance shape derivative V_C on NITO binary → voxel hex mesh.

Volume-form assembly + L²(Γ) projection on the staircased hex boundary.

Example (Docker):

    python scripts/lib/fea/tosa_compliance_shape_derivative.py --index 0 --test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from lib.fea.problem import LinearElasticityProblem
from lib.fea.shape_derivative import BoundaryMassMode
from lib.fea.types import MaterialProperties
from lib.meshing.mesh import build_nito_hex_mesh
from lib.nito_io import load_nito_arrays, load_sample, resolve_data_dir
from lib.paths import REPO_ROOT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compliance V_C on NITO binary → voxel hex (volume form → L²(Γ))."
    )
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--test", action="store_true")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--youngs-modulus", type=float, default=1.0)
    p.add_argument("--poisson-ratio", type=float, default=0.33)
    p.add_argument("--element-degree", type=int, default=1)
    p.add_argument(
        "--boundary-mass",
        choices=("lumped", "consistent"),
        default="lumped",
    )
    p.add_argument(
        "--cutoff",
        type=float,
        default=0.5,
        help="Binarization threshold for NITO topology density.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = resolve_data_dir(args.data_dir, test=args.test)
    shape, rho, bc, load, vf = load_sample(load_nito_arrays(data_dir), args.index)

    out_dir = args.output_dir or (
        REPO_ROOT / "output" / "hex_sensitivity" / str(args.index)
    )
    mesh_dir = out_dir / "mesh"

    hex_mesh = build_nito_hex_mesh(
        index=args.index,
        shape=shape,
        rho=rho,
        bc=bc,
        output_dir=mesh_dir,
    )

    problem = LinearElasticityProblem(
        hex_mesh,
        MaterialProperties(
            youngs_modulus=args.youngs_modulus,
            poisson_ratio=args.poisson_ratio,
        ),
        element_degree=args.element_degree,
        boundary_mass_mode=BoundaryMassMode(args.boundary_mass),
    )
    solution = problem.solve(
        shape,
        bc,
        load,
        output_dir=out_dir,
        vox=hex_mesh.vox,
    )

    print(f"shape={tuple(int(x) for x in shape)} vf={vf:.4f}")
    print(f"n_hex={hex_mesh.cell_ijk.shape[0]} n_nodes={hex_mesh.domain.geometry.x.shape[0]}")
    print(f"Compliance C = {solution.compliance:.6g}")
    print(f"Displacement: {solution.displacement_path}")
    print(f"V_C surface:  {solution.shape_derivative_path}")


if __name__ == "__main__":
    main()
