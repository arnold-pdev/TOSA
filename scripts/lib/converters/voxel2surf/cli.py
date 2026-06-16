"""Command-line interface for staged voxel2surf pipelines."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

from lib.converters.voxel2surf.pipeline import save_surface, voxel_to_surface
from lib.converters.voxel2surf.recipes import get_recipe, list_recipes
from lib.converters.voxel2surf.run_io import (
    PipelineLogger,
    append_metrics_csv,
    update_manifest,
)
from lib.converters.voxel2surf.types import (
    PipelineOptions,
    RunMetrics,
    SurfaceOutputFormat,
    resolve_run_paths,
)
from lib.volume_check import format_volume_fraction_lines, volume_fraction_mesh


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Staged voxel2surf: extract, BC enforce, smooth, validate.",
    )
    p.add_argument("--index", type=int, default=None)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--test", action="store_true")
    p.add_argument(
        "--recipe",
        type=str,
        default="pyvista_laplacian",
        help="Pipeline recipe (pyvista_laplacian, pyvista_taubin, planar_cap_contour, …)",
    )
    p.add_argument("--list-recipes", action="store_true")
    p.add_argument("--format", choices=tuple(f.value for f in SurfaceOutputFormat), default=None)
    p.add_argument("-o", "--output", type=Path, default=None)
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Experiment run root (writes vtp/, logs/, metrics/, manifest.json)",
    )
    p.add_argument("--density-cutoff", type=float, default=0.5)
    p.add_argument("--no-bc", action="store_true")

    g = p.add_argument_group("extract")
    g.add_argument("--extractor", type=str, default=None, help="Override recipe extractor")
    g.add_argument("--iso-level", type=float, default=0.5)
    g.add_argument("--no-repair", action="store_true")
    g.add_argument("--meshfix-verbose", action="store_true")
    g.add_argument("--calibrate-vf", action="store_true")
    g.add_argument("--field-smooth-sigma", type=float, default=0.0)
    g.add_argument("--field-bc-mask", action="store_true")
    g.add_argument("--upsample-factor", type=int, default=1)

    g = p.add_argument_group("BC geometry")
    g.add_argument(
        "--bc-transfer",
        choices=("centroid", "cell_native"),
        default=None,
        help="BC label transfer backend",
    )
    g.add_argument("--bc-transfer-band-cells", type=float, default=1.0)
    g.add_argument(
        "--bc-regrow",
        choices=("none", "geodesic"),
        default=None,
        help="Geodesic BC patch regrow before smooth",
    )
    g.add_argument("--bc-oracle", action="store_true", help="Cuberille BC QA metrics")
    g.add_argument("--bc-strict-footprint", action="store_true")
    g.add_argument("--bc-plane-tol", type=float, default=1e-5)
    g.add_argument(
        "--on-bc-fail",
        choices=("raise", "warn"),
        default="raise",
        help="When labeled BC vertices deviate from NITO analytic planes",
    )
    g.add_argument(
        "--no-bc-claim-coplanar",
        action="store_true",
        help="Do not claim unlabeled triangles on NITO BC planes before snap",
    )
    g.add_argument(
        "--bc-assembly",
        choices=("monolithic", "shared_seam"),
        default=None,
        help="BC assembly backend (monolithic=MC BC faces, shared_seam=plane-cut caps)",
    )
    g.add_argument(
        "--bc-planar-cap",
        action="store_true",
        help="Build planar BC cap contours (audit only; caps are not assembled onto the skin)",
    )
    g.add_argument(
        "--bc-planar-cap-method",
        type=str,
        default="upsample_blur",
        help="Planar cap contour method (upsample_blur, native_contour)",
    )
    g.add_argument("--bc-planar-cap-upsample", type=int, default=4)
    g.add_argument("--bc-planar-cap-blur-sigma", type=float, default=1.0)
    g.add_argument("--bc-planar-cap-outward-buffer", type=float, default=0.25)

    g = p.add_argument_group("smooth")
    g.add_argument("--smoother", type=str, default=None, help="Override recipe smoother")
    g.add_argument("--subdivide-levels", type=int, default=0)
    g.add_argument("--laplacian-iters", type=int, default=None)
    g.add_argument("--laplacian-relaxation", type=float, default=0.5)
    g.add_argument("--taubin-iters", type=int, default=None)
    g.add_argument("--taubin-lambda", type=float, default=0.5)
    g.add_argument(
        "--taubin-k-pb",
        type=float,
        default=0.1,
        help="Taubin pass-band frequency; μ is derived from λ and k_PB",
    )
    g.add_argument(
        "--taubin-mu",
        type=float,
        default=None,
        help="Explicit signed Taubin μ (<0); overrides --taubin-k-pb",
    )
    g.add_argument(
        "--taubin-nu",
        type=float,
        default=None,
        help="Deprecated: positive magnitude of μ; use --taubin-mu or --taubin-k-pb",
    )
    g.add_argument("--no-constrain-bc-planes", action="store_true")
    g.add_argument("--laplacian-freeze-rings", type=int, default=0)
    g.add_argument("--smooth-free-only", action="store_true")
    g.add_argument("--subdivide-free-only", action="store_true")
    g.add_argument("--hc-iters", type=int, default=None)
    g.add_argument("--hc-lambda", type=float, default=0.5)
    g.add_argument("--hc-alpha", type=float, default=0.1)
    g.add_argument("--snap-to-sdf", action="store_true")
    g.add_argument("--snap-to-sdf-steps", type=int, default=3)

    g = p.add_argument_group("validation")
    g.add_argument("--load-surface-tol-cells", type=float, default=1.0)
    g.add_argument("--vf-tol-cells", type=float, default=2.0)
    g.add_argument("--no-load-check", action="store_true")
    g.add_argument(
        "--on-load-fail",
        choices=("raise", "warn"),
        default="raise",
    )

    p.add_argument("--log-file", type=Path, default=None)
    p.add_argument("--log-dir", type=Path, default=None)
    p.add_argument("-q", "--quiet", action="store_true")
    return p.parse_args(argv)


def _options_from_args(args: argparse.Namespace) -> PipelineOptions:
    recipe = get_recipe(args.recipe)
    taubin_mu = args.taubin_mu
    if taubin_mu is None and args.taubin_nu is not None:
        warnings.warn(
            "--taubin-nu is deprecated; use --taubin-mu (signed, <0) or --taubin-k-pb",
            DeprecationWarning,
            stacklevel=2,
        )
        taubin_mu = -abs(args.taubin_nu)
    opts = PipelineOptions(
        extractor=args.extractor or recipe.extractor,
        smoother=args.smoother or recipe.smoother,
        iso_level=args.iso_level,
        repair=not args.no_repair,
        meshfix_verbose=args.meshfix_verbose,
        subdivide_levels=args.subdivide_levels,
        laplacian_iters=8 if args.laplacian_iters is None else args.laplacian_iters,
        laplacian_relaxation=args.laplacian_relaxation,
        taubin_iters=10 if args.taubin_iters is None else args.taubin_iters,
        taubin_lambda=args.taubin_lambda,
        taubin_k_pb=args.taubin_k_pb,
        taubin_mu=taubin_mu,
        constrain_bc_planes=not args.no_constrain_bc_planes,
        laplacian_freeze_rings=args.laplacian_freeze_rings,
        calibrate_vf=args.calibrate_vf,
        field_smooth_sigma=args.field_smooth_sigma,
        field_bc_mask=args.field_bc_mask,
        upsample_factor=args.upsample_factor,
        snap_to_sdf=args.snap_to_sdf,
        snap_to_sdf_steps=args.snap_to_sdf_steps,
        load_surface_tol_cells=args.load_surface_tol_cells,
        vf_tol_cells=args.vf_tol_cells,
        check_loads=not args.no_load_check,
        on_load_fail=args.on_load_fail,
        bc_strict_footprint=args.bc_strict_footprint,
        bc_transfer=args.bc_transfer or "centroid",
        bc_transfer_band_cells=args.bc_transfer_band_cells,
        bc_regrow=args.bc_regrow or "none",
        bc_oracle=args.bc_oracle,
        bc_plane_tol=args.bc_plane_tol,
        on_bc_fail=args.on_bc_fail,
        bc_claim_coplanar=not args.no_bc_claim_coplanar,
        bc_assembly=args.bc_assembly or recipe.bc_assembly,
        bc_planar_cap=args.bc_planar_cap or ("bc_planar_cap" in recipe.stages),
        bc_planar_cap_method=args.bc_planar_cap_method,
        bc_planar_cap_upsample=args.bc_planar_cap_upsample,
        bc_planar_cap_blur_sigma=args.bc_planar_cap_blur_sigma,
        bc_planar_cap_outward_buffer=args.bc_planar_cap_outward_buffer,
        smooth_free_only=args.smooth_free_only,
        subdivide_free_only=args.subdivide_free_only,
        hc_iters=10 if args.hc_iters is None else args.hc_iters,
        hc_lambda=args.hc_lambda,
        hc_alpha=args.hc_alpha,
    )
    return opts


def main(argv: list[str] | None = None) -> None:
    from lib.converters.bc_patch import bcspecs_from_nito
    from lib.nito_io import load_nito_arrays, load_sample, resolve_data_dir
    from lib.surface_io import DEFAULT_SURFACE_DIR, vtp_path
    from lib.volume_check import binarize_topology, domain_volume

    args = _parse_args(argv)
    if args.list_recipes:
        for recipe in list_recipes():
            print(f"{recipe.id:24}  {recipe.description}")
        return

    if args.index is None:
        raise SystemExit("--index is required unless --list-recipes is set")

    recipe = get_recipe(args.recipe)
    verbose = not args.quiet
    data_dir = resolve_data_dir(args.data_dir, test=args.test)
    shape, rho, bc, load, vf_nito = load_sample(load_nito_arrays(data_dir), args.index)

    default_out = vtp_path(DEFAULT_SURFACE_DIR, args.index)
    out_path, log_path = resolve_run_paths(
        args.run_dir,
        args.index,
        output=args.output,
        log_file=args.log_file,
        log_dir=args.log_dir,
        default_output=default_out,
    )
    logger = PipelineLogger(echo_stdout=verbose, log_path=log_path)
    vox = binarize_topology(rho, shape, cutoff=args.density_cutoff)
    spacing = np.ones(int(shape.size), dtype=float)
    use_bc = not args.no_bc
    opts = _options_from_args(args)

    logger(f"voxel2surf  index={args.index}  recipe={args.recipe}")
    logger(f"  data_dir={data_dir}")
    logger(f"  shape={tuple(int(x) for x in shape)}  solid_voxels={int(np.count_nonzero(vox)):,}")
    logger(
        f"  pipeline: extractor={opts.extractor} smoother={opts.smoother} "
        f"subdivide={opts.subdivide_levels} "
        f"bc_assembly={opts.bc_assembly} "
        f"bc_planar_cap={opts.bc_planar_cap}"
        + (f" method={opts.bc_planar_cap_method}" if opts.bc_planar_cap else ""),
    )
    logger(f"  output={out_path.resolve()}")
    if args.run_dir is not None:
        logger(f"  run_dir={args.run_dir.expanduser().resolve()}")
    if opts.check_loads:
        logger(
            f"  load_surface_check=tol {opts.load_surface_tol_cells} cells "
            f"on_fail={opts.on_load_fail}",
        )
    if log_path is not None:
        logger(f"  log={log_path.resolve()}")

    result = voxel_to_surface(
        vox,
        bcspecs_from_nito(bc, shape) if use_bc else [],
        bc_rows=bc if use_bc else None,
        load_rows=None if args.no_load_check else load,
        grid_shape=shape,
        spacing=spacing,
        options=opts,
        recipe=recipe,
        log=logger,
    )

    logger("  write tagged VTK …")
    out = save_surface(result, out_path, fmt=args.format)
    vf_lines = format_volume_fraction_lines(
        vox=vox,
        mesh=result.mesh,
        shape=shape,
        spacing=spacing,
        vf_nito=vf_nito,
    )
    n_bc = int(np.count_nonzero(result.tri_labels >= 0))
    load_warn = ", load_check=WARN" if result.load_check_failed else ""
    summary = (
        f"done  {out}  ({len(result.mesh.vertices):,} vertices, "
        f"{len(result.patches)} BC patches, {n_bc:,} labeled triangles{load_warn})"
    )
    logger(summary)
    construction_sec = result.bc_audit.get("construction_sec")
    if construction_sec is not None:
        logger(f"  construction_sec={float(construction_sec):.3f}")
    for line in vf_lines:
        logger(line)
    logger.save()

    if not verbose:
        print(summary)
        for line in vf_lines:
            print(line)

    if args.run_dir is not None:
        from lib.volume_check import domain_volume, volume_fraction_voxels

        dom = domain_volume(shape, spacing)
        vf_mesh = volume_fraction_mesh(result.mesh, dom)
        vf_voxel = volume_fraction_voxels(vox)
        metrics = RunMetrics(
            index=args.index,
            recipe_id=recipe.id,
            vf_voxel=vf_voxel,
            vf_mesh=vf_mesh,
            vf_delta=None if vf_mesh is None else vf_mesh - vf_voxel,
            vf_nito=float(vf_nito),
            bc_plane_max_residual=float(
                result.bc_audit.get("bc_plane_max_residual", 0.0),
            ),
            bc_labeled_triangles=n_bc,
            bc_oracle_max_gap=result.bc_audit.get("bc_oracle_max_gap"),
            bc_min_patch_tris=result.bc_audit.get("bc_min_patch_tris"),
            free_dihedral_mean_deg=result.bc_audit.get("free_dihedral_mean_deg"),
            free_dihedral_p95_deg=result.bc_audit.get("free_dihedral_p95_deg"),
            axis_aligned_edge_frac=result.bc_audit.get("axis_aligned_edge_frac"),
            load_check_failed=result.load_check_failed,
            vertices=len(result.mesh.vertices),
            faces=len(result.mesh.faces),
            patches=len(result.patches),
        )
        run_root = args.run_dir.expanduser().resolve()
        append_metrics_csv(run_root, metrics)
        update_manifest(
            run_root,
            recipe_id=recipe.id,
            options=opts,
            index=args.index,
            metrics=metrics,
        )
