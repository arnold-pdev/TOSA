"""
TOSA sensitivity analysis — compliance gradient w.r.t. element densities rho.

Loads a NITO-3D design from the .npy bundles produced by nito/download.py, runs
ATOMS forward FEA, and returns dC/drho using the same adjoint chain rule as the
SIMP optimizer (density filter + SIMP penalty).

Run from anywhere; ATOMS is imported from the nito/ submodule:

    conda activate tosa
    python scripts/sensitivity_analysis/main.py --index 0 --data-dir nito/Data --test

Prerequisite data (from repo root):

    cd nito && python download.py --data
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# ATOMS lives under nito/; evaluate.py uses the same import style.
REPO_ROOT = Path(__file__).resolve().parents[2]
NITO_ROOT = REPO_ROOT / "nito"
if str(NITO_ROOT) not in sys.path:
    sys.path.insert(0, str(NITO_ROOT))

from ATOMS.MaterialModels import SingleMaterial
from ATOMS.geometry import generate_structured_mesh
from ATOMS.solver import Solver
from ATOMS.utils import filter_2D_structured, filter_3D_structured

logger = logging.getLogger(__name__)


def load_nito_arrays(data_dir: Path) -> dict[str, np.ndarray]:
    """Load aligned NITO arrays from a download.py output directory."""
    required = (
        "shapes.npy",
        "topologies.npy",
        "boundary_conditions.npy",
        "loads.npy",
        "vfs.npy",
    )
    missing = [name for name in required if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing in {data_dir}: {', '.join(missing)}. "
            "Run: ./scripts/fetch_data.sh --test-only"
        )
    return {
        "shapes": np.load(data_dir / "shapes.npy", allow_pickle=True),
        "topologies": np.load(data_dir / "topologies.npy", allow_pickle=True),
        "boundary_conditions": np.load(
            data_dir / "boundary_conditions.npy", allow_pickle=True
        ),
        "loads": np.load(data_dir / "loads.npy", allow_pickle=True),
        "vfs": np.load(data_dir / "vfs.npy", allow_pickle=True),
    }


def load_sample(
    data: dict[str, np.ndarray], index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (shape, rho, BC, load, vf) for one dataset index."""
    shape = np.asarray(data["shapes"][index], dtype=int)
    rho = np.asarray(data["topologies"][index], dtype=float).reshape(-1, 1)
    bc = np.asarray(data["boundary_conditions"][index], dtype=float)
    load = np.asarray(data["loads"][index], dtype=float)
    vf = float(data["vfs"][index])
    return shape, rho, bc, load, vf


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
    """
    Map NITO BC/load rows onto nearest mesh nodes.

    Positions are already normalized to the unit domain used by
    generate_structured_mesh(dim=shape / shape.max(), ...) — same convention as
    nito/evaluate.py (no extra scaling by shape.max()).
    """
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

    rho is (n_elements, 1). The returned gradient is w.r.t. those design variables,
    including the chain rule through the density filter and SIMP penalty
    (ATOMS Solver.system_solve, same as the first TO iteration).
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

    # Expand free DOF displacements to full nodal field (n_nodes, dim).
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
        description="Compute compliance and dC/drho for a NITO design via ATOMS."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "nito" / "Data",
        help="NITO data directory (shapes/topologies/BC/load/vfs .npy); default nito/Data",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Use <data-dir>/Test (test split from download.py)",
    )
    parser.add_argument("--index", type=int, default=0, help="Sample index in dataset")
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
        help="Density filter radius in physical units (default 1.5/shape.max())",
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
