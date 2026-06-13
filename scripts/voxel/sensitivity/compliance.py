#!/usr/bin/env python3
"""
Voxel density-based compliance sensitivity (dC/drho via ATOMS).

Operates on voxel/grid designs from NITO .npy bundles (or --rho-file). Gradient
w.r.t. element densities rho — not shape derivatives on STL surfaces.

Usage:

    ./scripts/voxel/sensitivity/compliance.sh --index 0 --test
    python scripts/voxel/sensitivity/compliance.py --index 3 --test \\
        --rho-file output/nito_predictions/3/rho_pred.npy \\
        --output-dir output/atoms_results/nito_pred/3
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[2]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from lib.nito_io import load_nito_arrays, load_sample
from lib.paths import REPO_ROOT, ensure_nito_on_path

ensure_nito_on_path()

NITO_ROOT = REPO_ROOT / "nito"

from ATOMS.MaterialModels import SingleMaterial
from ATOMS.geometry import generate_structured_mesh
from ATOMS.solver import Solver
from ATOMS.utils import filter_2D_structured, filter_3D_structured

logger = logging.getLogger(__name__)


def build_solver(
    shape: np.ndarray,
    *,
    penalty: float,
    volume_fraction: float,
    filter_radius: float | None = None,
    solver: str = "cholesky",
) -> Solver:
    """Construct an ATOMS Solver for a structured NITO grid."""
    shape = np.asarray(shape, dtype=int)
    dim = shape.size
    if dim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D shape, got {shape}")

    scale = float(shape.max())
    elements, nodes = generate_structured_mesh(dim=shape / scale, nel=shape)

    if filter_radius is None:
        filter_radius = 1.5 / scale

    if dim == 2:
        nelx, nely = int(shape[0]), int(shape[1])
        filter_kernel = filter_2D_structured(
            elements=elements,
            nodes=nodes,
            nelx=nelx,
            nely=nely,
            r_min=filter_radius,
        )
    else:
        nelx, nely, nelz = int(shape[0]), int(shape[1]), int(shape[2])
        filter_kernel = filter_3D_structured(
            elements=elements,
            nodes=nodes,
            nelx=nelx,
            nely=nely,
            nelz=nelz,
            r_min=filter_radius,
        )

    material = SingleMaterial(
        E=1.0,
        nu=0.33,
        penalty=penalty,
        volume_fraction=volume_fraction,
        void=1e-9,
    )

    return Solver(
        mesh=(nodes, elements),
        material_model=material,
        filter_kernel=filter_kernel,
        structured=True,
        solver=solver,
    )


def apply_nito_bcs(
    solver: Solver, shape: np.ndarray, bc: np.ndarray, load: np.ndarray
) -> None:
    """Map NITO BC/load rows onto nearest mesh nodes (unit-domain convention)."""
    dim = shape.size

    solver.reset_BC()
    solver.reset_F()

    bc = np.atleast_2d(bc)
    load = np.atleast_2d(load)

    solver.add_BCs(bc[:, :dim], bc[:, dim : 2 * dim])
    solver.add_Forces(load[:, :dim], load[:, dim : 2 * dim])


def _solver_fn(solver: Solver):
    if solver.solver == "cholesky":
        return solver.solve_cholesky
    if solver.solver == "splu":
        return solver.solve_splu
    if solver.solver == "cg":
        return solver.solve_cg
    if solver.solver == "bicgstab":
        return solver.solve_bicgstab
    raise ValueError(f"Unsupported solver: {solver.solver}")


def compliance_and_gradient(
    solver: Solver,
    rho: np.ndarray,
    *,
    iteration: int = 0,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Compute compliance C and dC/drho at fixed design rho.

    Gradient includes chain rule through density filter and SIMP penalty.
    """
    rho = np.asarray(rho, dtype=float).reshape(-1, 1)
    solver_fn = _solver_fn(solver)

    if solver.reordering:
        order, inv_order = solver.compute_ordering(
            rho,
            solver.K_kernel,
            solver.k_map,
            solver.non_constrained_map,
        )
    else:
        order, inv_order = None, None

    compliance, dC_drho, U_free, _ = Solver.system_solve(
        rho,
        solver.material_model,
        solver.K_kernel,
        solver.k_map,
        solver.F,
        solver.dK,
        solver.grad_map,
        solver.non_constrained_map,
        solver.filter_kernel,
        solver_fn,
        iteration,
        mb_order=order,
        inv_order=inv_order,
        max_iter=solver.solver_max_iter,
        tol=solver.solve_tol,
    )

    U = np.zeros(solver.node_positions.shape[0] * solver.dim, dtype=float)
    free = np.where(solver.c.reshape(-1) == 0)[0]
    U[free] = U_free
    U = U.reshape(-1, solver.dim)

    return float(compliance), np.asarray(dC_drho).reshape(-1), U


def finite_difference_check(
    solver: Solver,
    rho: np.ndarray,
    analytic: np.ndarray,
    *,
    eps: float = 1e-6,
    n_probe: int = 5,
    seed: int = 0,
) -> dict[str, float]:
    """Spot-check a few elements with central differences on C."""
    rng = np.random.default_rng(seed)
    n_elem = rho.size
    probes = rng.choice(n_elem, size=min(n_probe, n_elem), replace=False)

    C0, _, _ = compliance_and_gradient(solver, rho)
    rel_errors = []

    for e in probes:
        rho_p = rho.copy().reshape(-1)
        rho_m = rho.copy().reshape(-1)
        rho_p[e] = np.clip(rho_p[e] + eps, 0.0, 1.0)
        rho_m[e] = np.clip(rho_m[e] - eps, 0.0, 1.0)
        Cp, _, _ = compliance_and_gradient(solver, rho_p.reshape(-1, 1))
        Cm, _, _ = compliance_and_gradient(solver, rho_m.reshape(-1, 1))
        fd = (Cp - Cm) / (rho_p[e] - rho_m[e])
        rel_errors.append(abs(fd - analytic[e]) / (abs(fd) + 1e-12))

    return {
        "compliance": C0,
        "n_probe": len(probes),
        "max_rel_error": float(np.max(rel_errors)),
        "mean_rel_error": float(np.mean(rel_errors)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "[Archived] Density-based compliance sensitivity (dC/drho) on "
            "voxel/grid designs via ATOMS."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "nito" / "Data",
        help="NITO data directory (shapes/topologies/BC/load/vfs .npy)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Use <data-dir>/Test (test split)",
    )
    parser.add_argument("--index", type=int, default=0, help="Sample index in dataset")
    parser.add_argument(
        "--rho-file",
        type=Path,
        default=None,
        help="Density field instead of topologies.npy (e.g. NITO rho_pred.npy)",
    )
    parser.add_argument("--penalty", type=float, default=3.0, help="SIMP penalty p")
    parser.add_argument(
        "--solver",
        type=str,
        default="cholesky",
        choices=("cholesky", "splu", "cg", "bicgstab"),
    )
    parser.add_argument(
        "--filter-radius",
        type=float,
        default=None,
        help="Density filter radius (default 1.5/shape.max())",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for result .npy files (default: output/atoms_results/<index>)",
    )
    parser.add_argument(
        "--verify-fd",
        action="store_true",
        help="Run a small finite-difference spot check on dC/drho",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    data_dir = args.data_dir / "Test" if args.test else args.data_dir
    data = load_nito_arrays(data_dir)
    n_samples = len(data["shapes"])
    if not (0 <= args.index < n_samples):
        raise IndexError(f"index {args.index} out of range [0, {n_samples})")

    shape, rho, bc, load, vf = load_sample(data, args.index)
    if args.rho_file is not None:
        rho_file = Path(args.rho_file)
        if not rho_file.is_file():
            raise FileNotFoundError(f"--rho-file not found: {rho_file}")
        rho = np.load(rho_file, allow_pickle=True).astype(float).reshape(-1, 1)
        if rho.size != int(np.prod(shape)):
            raise ValueError(
                f"--rho-file length {rho.size} != shape.prod() {int(np.prod(shape))}"
            )
        logger.info("Using design from %s", rho_file)
    logger.info(
        "Sample %d: shape=%s, n_elements=%d, vf=%.4f",
        args.index,
        shape.tolist(),
        rho.size,
        vf,
    )

    solver = build_solver(
        shape,
        penalty=args.penalty,
        volume_fraction=vf,
        filter_radius=args.filter_radius,
        solver=args.solver,
    )
    apply_nito_bcs(solver, shape, bc, load)

    compliance, dC_drho, displacement = compliance_and_gradient(solver, rho)

    print(f"compliance C = {compliance:.6e}")
    print(f"dC/drho: shape {dC_drho.shape}, min {dC_drho.min():.4e}, max {dC_drho.max():.4e}")

    if args.verify_fd:
        stats = finite_difference_check(solver, rho, dC_drho)
        print(
            f"FD check ({stats['n_probe']} elements): "
            f"max rel err {stats['max_rel_error']:.3e}, "
            f"mean rel err {stats['mean_rel_error']:.3e}"
        )

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = REPO_ROOT / "output" / "atoms_results" / str(args.index)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "compliance.npy", compliance)
    np.save(out_dir / "dC_drho.npy", dC_drho)
    np.save(out_dir / "displacement.npy", displacement)
    np.save(out_dir / "shape.npy", shape)
    np.save(out_dir / "rho.npy", rho.reshape(-1))
    logger.info("Wrote results to %s", out_dir)


if __name__ == "__main__":
    main()
