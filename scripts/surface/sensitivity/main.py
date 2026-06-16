#!/usr/bin/env python3
"""
Surface / hex shape-derivative sensitivity analysis.

Primary path (Stage 0): NITO binary topology → voxel hex → FEniCSx V_C.

Optional tet path (later): tagged .vtp → Gmsh/TetGen → same volume-form pipeline.

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
from lib.meshing.mesh import build_nito_hex_mesh, build_volume_mesh
from lib.meshing.size_fields import sizing_from_nito_sample
from lib.meshing.types import MeshBackend
from lib.nito_io import load_nito_arrays, load_sample, resolve_data_dir
from lib.paths import REPO_ROOT
from lib.surface_io import resolve_surface_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Shape-derivative sensitivity (hex or tet mesh → FEA → .vtp)."
    )
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--test", action="store_true")
    p.add_argument(
        "--mesh-backend",
        choices=tuple(b.value for b in MeshBackend),
        default=MeshBackend.VOXEL_HEX.value,
        help="Default voxel_hex: NITO binary → structured hex. Use gmsh/tetgen for body-fitted tets.",
    )
    p.add_argument(
        "--surface-file",
        type=Path,
        default=None,
        help="Required for gmsh/tetgen backends (default: public/vtp/<index>.vtp)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: output/surface_sensitivity/<index>/ (hex: output/hex_sensitivity/)",
    )
    p.add_argument("--youngs-modulus", type=float, default=1.0)
    p.add_argument("--poisson-ratio", type=float, default=0.33)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = resolve_data_dir(args.data_dir, test=args.test)
    shape, rho, bc, load, _vf = load_sample(load_nito_arrays(data_dir), args.index)

    backend = MeshBackend(args.mesh_backend)
    if backend == MeshBackend.VOXEL_HEX:
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
    else:
        out_dir = args.output_dir or (
            REPO_ROOT / "output" / "surface_sensitivity" / str(args.index)
        )
        surface_path = resolve_surface_path(
            index=args.index,
            surface_file=args.surface_file,
            surface_dir=REPO_ROOT / "public" / "vtp",
        )
        sizing, _bc_pts, _load_pts = sizing_from_nito_sample(shape, bc, load)
        mesh = build_volume_mesh(
            index=args.index,
            surface_path=surface_path,
            output_dir=out_dir / "mesh",
            sizing=sizing,
            backend=backend,
        )
        problem = LinearElasticityProblem(
            mesh,
            MaterialProperties(
                youngs_modulus=args.youngs_modulus,
                poisson_ratio=args.poisson_ratio,
            ),
            surface_path=surface_path,
        )
        solution = problem.solve(shape, bc, load, output_dir=out_dir)

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
