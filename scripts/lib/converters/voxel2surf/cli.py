"""
voxel2surf CLI — load a NITO index, mesh it, write a tagged .vtp + log.

    PYTHONPATH=scripts python -m lib.converters.voxel2surf.cli \
        --index 119 --weights cotangent --cycles 2 --sub-levels 1 \
        --on-fail warn -o output/surfaces/small/119.vtp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from lib.converters.voxel2surf.main import Logger, Options, mesh_surface, write_vtp


def load_case(index, data_dir=None, *, test=False, density_cutoff=0.5):
    """NITO bundle → (vox, bc, load, shape) for ``index``."""
    from lib.nito_io import load_nito_arrays, load_sample, resolve_data_dir
    from lib.volume_check import binarize_topology

    dd = resolve_data_dir(data_dir, test=test)
    shape, rho, bc, load, _vf = load_sample(load_nito_arrays(dd), index)
    vox = binarize_topology(rho, shape, cutoff=density_cutoff)
    return vox, bc, load, np.asarray(shape, dtype=int)


def build_patches(vox, bc, load, shape, origin, spacing):
    """NITO BC/load rows → (support patches, load patches) via the shared parser."""
    from lib.converters.bc_patch import build_load_patches, build_patches_from_nito

    return (
        build_patches_from_nito(vox, bc, shape, origin, spacing),
        build_load_patches(load, shape),
    )


def _parse_args(argv):
    p = argparse.ArgumentParser(
        description="voxel2surf: cuberille + constrained Taubin → tagged .vtp",
    )
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--test", action="store_true")
    p.add_argument("--density-cutoff", type=float, default=0.5)
    p.add_argument("--no-bc", action="store_true", help="skip NITO BC parsing / tagging")

    g = p.add_argument_group("mesh")
    g.add_argument("--fair-mode", choices=("implicit", "taubin"), default="implicit",
                   help="implicit = energy + occupancy corridor (primary); taubin = fast explicit "
                        "band-limit (NO corridor, may collapse thin members — for previews/comparison)")
    g.add_argument("--target-faces", type=int, default=None,
                   help="final triangle budget; omit to skip decimation (keep refined resolution)")
    g.add_argument("--sub-levels", type=int, default=0,
                   help="unconditional midpoint subdivisions (each ×4 faces); independent of --target-faces")
    g.add_argument("--max-sub-levels", type=int, default=2, help="extra cap on the target-driven subdivisions")

    gi = p.add_argument_group("implicit (fair-mode=implicit)")
    gi.add_argument("--smooth-order", choices=("membrane", "thin-plate"), default="thin-plate",
                    help="energy: thin-plate (C1, non-shrinking — default) or membrane (C0, shrinks)")
    gi.add_argument("--data-weight", type=float, default=1e-2,
                    help="soft pull to the cuberille boundary (anti-shrink); higher = hugs more, lower = smoother")
    gi.add_argument("--corridor-voxels", type=float, default=1.5,
                    help="SDF corridor normal half-width (voxels); loose safety + thin-feature protection")
    gi.add_argument("--implicit-iters", type=int, default=12, help="primal-dual active-set rounds")

    gt = p.add_argument_group("taubin (fair-mode=taubin, fast fallback)")
    gt.add_argument("--iters", type=int, default=50, help="Taubin λ|μ iterations per fairing pass")
    gt.add_argument("--taubin-lambda", type=float, default=0.5,
                    help="λ (0.5 zeroes the mesh Nyquist mode = the voxel staircase)")
    gt.add_argument("--taubin-k-pb", type=float, default=0.1,
                    help="Taubin pass-band frequency; μ derived from λ and k_PB")
    gt.add_argument("--taubin-mu", type=float, default=None,
                    help="explicit signed μ (<0); overrides --taubin-k-pb")
    gt.add_argument("--weights", choices=("cotangent", "uniform"), default="cotangent")
    gt.add_argument("--symmetric", action="store_true", help="disable the asymmetric boundary coupling")
    gt.add_argument("--smooth-interp", action="store_true",
                    help="Taubin §4.2 smooth interpolation: fair unconstrained, then add an "
                         "exact smooth correction (fixed uniform kernel) instead of per-step projection")
    gt.add_argument("--smooth-kernel-iters", type=int, default=5,
                    help="locality of the §4.2 correction bump; keep small (F_mm conditioning blows up ~1e18 by 50)")
    g.add_argument(
        "--on-fail",
        choices=("warn", "raise"),
        default="warn",
        help="gate on self-intersection / overlap / watertight / BC residual",
    )

    p.add_argument("-o", "--output", type=Path, default=None)
    p.add_argument("--log-file", type=Path, default=None)
    p.add_argument("-q", "--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = _parse_args(argv)
    log = Logger(echo=not a.quiet, path=a.log_file)
    log(f"voxel2surf  index={a.index}")

    vox, bc, load, shape = load_case(
        a.index, a.data_dir, test=a.test, density_cutoff=a.density_cutoff,
    )
    origin, spacing = np.zeros(3, float), np.ones(3, float)

    bc_patches, load_patches = ([], [])
    if not a.no_bc:
        bc_patches, load_patches = build_patches(vox, bc, load, shape, origin, spacing)
        log(f"  BC: {len(bc_patches)} support patch(es), {len(load_patches)} load(s)")

    mu = a.taubin_mu if a.taubin_mu is not None else a.taubin_lambda / (a.taubin_lambda * a.taubin_k_pb - 1.0)
    opts = Options(
        density_cutoff=a.density_cutoff,
        with_bc=not a.no_bc,
        iters=a.iters,
        lamb=a.taubin_lambda,
        mu=mu,
        target_faces=a.target_faces,
        sub_levels=a.sub_levels,
        max_sub_levels=a.max_sub_levels,
        fair_mode=a.fair_mode,
        smooth_order=a.smooth_order,
        corridor_voxels=a.corridor_voxels,
        implicit_iters=a.implicit_iters,
        data_weight=a.data_weight,
        weights=a.weights,
        asymmetric=not a.symmetric,
        smooth_interp=a.smooth_interp,
        smooth_kernel_iters=a.smooth_kernel_iters,
        on_fail=a.on_fail,
    )

    verts, faces, cell_arrays, report = mesh_surface(
        vox, bc_patches, load_patches, shape, origin, spacing, opts, log,
    )

    if a.output is not None:
        write_vtp(a.output, verts, faces, cell_arrays=cell_arrays, load_patches=load_patches)
        log(f"  wrote {a.output}")

    log.close()
    return 1 if report.get("gates_failed") else 0


if __name__ == "__main__":
    sys.exit(main())
