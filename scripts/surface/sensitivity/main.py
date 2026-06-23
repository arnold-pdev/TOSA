#!/usr/bin/env python3
"""
Surface / hex shape-derivative sensitivity analysis.

NITO binary topology → structured voxel hex → FEniCSx → volume-form shape derivative.

Voxel density-based compliance (∂C/∂ρ on grids):

    ./scripts/voxel/sensitivity/compliance.sh --index 0 --test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from lib.fea.postprocess import boundary_eps_sigma_contraction
from lib.fea.problem import LinearElasticityProblem
from lib.fea.shape_derivative import BoundaryMassMode
from lib.fea.types import MaterialProperties
from lib.meshing.mesh import build_nito_hex_mesh
from lib.nito_io import load_nito_arrays, load_sample, resolve_data_dir
from lib.paths import REPO_ROOT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Shape-derivative sensitivity (NITO binary → voxel hex → FEA → .vtp)."
    )
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--test", action="store_true")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: output/hex_sensitivity/<index>/",
    )
    p.add_argument(
        "--refine", type=int, default=1,
        help="upsample the hex grid ×refine per axis (finer staircase, same extent)",
    )
    p.add_argument("--youngs-modulus", type=float, default=1.0)
    p.add_argument("--poisson-ratio", type=float, default=0.33)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = resolve_data_dir(args.data_dir, test=args.test)
    shape, rho, bc, load, _vf = load_sample(load_nito_arrays(data_dir), args.index)

    out_dir = args.output_dir or (
        REPO_ROOT / "output" / "hex_sensitivity" / str(args.index)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    hex_mesh = build_nito_hex_mesh(
        index=args.index,
        shape=shape,
        rho=rho,
        bc=bc,
        output_dir=out_dir / "mesh",
        refine=args.refine,
    )
    problem = LinearElasticityProblem(
        hex_mesh,
        MaterialProperties(
            youngs_modulus=args.youngs_modulus,
            poisson_ratio=args.poisson_ratio,
        ),
        boundary_mass_mode=BoundaryMassMode.LUMPED,
    )
    solution = problem.solve(
        shape, bc, load, output_dir=out_dir, vox=hex_mesh.vox
    )

    surface_out = out_dir / "surface.vtp"
    boundary_eps_sigma_contraction(
        solution,
        surface_path=solution.shape_derivative_path or surface_out,
        output_path=surface_out,
    )

    print(f"Wrote {surface_out}")
    print(f"Visualize: python scripts/surface/visualize.py --index {args.index}")


if __name__ == "__main__":
    main()
