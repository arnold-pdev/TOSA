"""
ARCHIVED v2 pipeline (field upsample + VF calibration + optional Laplacian).

Active entry point: ``lib.converters.voxel2surf`` (PyVista contour + Laplacian).

voxel2surf v2: field-refinement surface pipeline from NITO voxel topologies.

Stages (optional steps controlled by keyword arguments):

    A  upsample field          (upsample_factor > 1)
    B  signed distance field   (always)
    C  field refine + VF cal   (field_smooth_sigma / calibrate_vf)
    D  marching cubes + repair (always)
    E  BC patches + transfer   (when bcspecs provided)
    F  subdivide + Laplacian   (subdivide_levels / laplacian_iters)
    G  flatten BC planes       (when BC patches exist)
    H  VF gate + load check    (enforce_vf_tol / check_loads)

v1 (Taubin + regrow) is preserved in ``voxel2surf_v1_archive.py``.

CLI::

    python scripts/lib/converters/voxel2surf.py --index 10 --data-dir nito/Data/3D
    python scripts/lib/converters/voxel2surf.py --index 10 --upsample-factor 2 \\
        --field-smooth-sigma 0.35 --subdivide-levels 0 --laplacian-iters 0
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

_SCRIPTS = Path(__file__).resolve().parents[2]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import numpy as np
import trimesh

from lib.meshing.field import (
    extract_surface_mesh,
    field_volume_fraction,
    refine_sdf_field,
    rescale_mesh_volume,
    signed_distance_field,
    upsample_field,
)
from lib.meshing.surface_smooth import (
    bc_interior_freeze_weights,
    constrained_laplacian_smooth,
    flatten_bc_planes,
    subdivide_mesh,
    vertex_patch_map,
)
from lib.volume_check import (
    PipelineStageReport,
    domain_volume,
    field_stage_report,
    format_volume_fraction_lines,
    mesh_stage_report,
    volume_fraction_voxels,
)

if TYPE_CHECKING:
    from lib.converters.bc_patch import BCSpec, Patch

LogFn = Callable[[str], None]


def _noop_log(_msg: str) -> None:
    pass


class PipelineLogger:
    """Capture pipeline messages; optionally echo to stdout and write to a file."""

    def __init__(self, *, echo_stdout: bool, log_path: Path | None) -> None:
        self._echo = echo_stdout
        self._path = log_path.expanduser().resolve() if log_path is not None else None
        self._lines: list[str] = []

    def __call__(self, msg: str) -> None:
        self._lines.append(msg)
        if self._echo:
            print(msg, flush=True)

    def save(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text("\n".join(self._lines) + "\n", encoding="utf-8")


@dataclass
class PipelineOptions:
    """Keyword-controlled optional processing for systematic exploration."""

    upsample_factor: int = 2
    upsample_order: int = 1
    field_smooth_sigma: float = 0.35
    calibrate_vf: bool = True
    vf_target: float | None = None
    vf_tol: float = 0.02
    enforce_vf_tol: bool = False
    repair: bool = True
    meshfix_verbose: bool = False
    subdivide_levels: int = 0
    laplacian_iters: int = 0
    laplacian_relaxation: float = 0.3
    constrain_bc_planes: bool = True
    laplacian_freeze_rings: int = 0
    vf_rescale: bool = True
    vf_rescale_min_drift: float = 0.005
    load_surface_tol_cells: float = 1.0
    check_loads: bool = True


@dataclass
class SurfaceResult:
    mesh: trimesh.Trimesh
    patches: list
    tri_labels: np.ndarray
    freeze: np.ndarray
    iso_level: float = 0.0
    vf_target: float = 0.0
    stage_reports: list[PipelineStageReport] = field(default_factory=list)


def _log_stage(log: LogFn, report: PipelineStageReport) -> None:
    log(report.format_line())


def transfer_patches(
    mesh: trimesh.Trimesh,
    patches: list[Patch],
    origin,
    spacing,
    vox_shape,
    band_cells: float = 1.0,
    cos_tol: float = 0.5,
) -> np.ndarray:
    """Per-triangle patch id (index into ``patches``), or -1 for free boundary."""
    origin = np.asarray(origin, float)
    spacing = np.asarray(spacing, float)
    vox_shape = np.asarray(vox_shape, dtype=int)
    ndim = int(vox_shape.size)
    centroids = mesh.triangles_center
    fnormals = mesh.face_normals
    cells = np.clip(
        np.floor((centroids - origin) / spacing).astype(int),
        0,
        vox_shape - 1,
    )

    labels = np.full(len(centroids), -1, dtype=int)
    for pid, patch in enumerate(patches):
        if not patch.is_bc:
            continue
        axis = patch.axis
        inplane = [i for i in range(ndim) if i != axis]
        foot = {tuple(f) for f in patch.faces}
        band = band_cells * spacing[axis]
        ref = (
            patch.plane_coord
            if patch.side == "min"
            else patch.plane_coord - spacing[axis]
        )
        near = np.abs(centroids[:, axis] - ref) < band
        aligned = fnormals @ np.asarray(patch.normal[:ndim]) > cos_tol
        in_foot = np.fromiter(
            (
                tuple(cells[i, j] for j in inplane) in foot
                for i in range(len(cells))
            ),
            dtype=bool,
            count=len(cells),
        )
        member = near & aligned & in_foot & (labels < 0)
        labels[member] = pid
    return labels


def _enabled_stages(opts: PipelineOptions, *, has_bc: bool) -> list[str]:
    stages: list[str] = []
    if opts.upsample_factor > 1:
        stages.append("A_upsample")
    stages.extend(["B_sdf", "C_field_refine", "D_extract"])
    if has_bc:
        stages.extend(["E_bc_patches", "E_bc_transfer"])
    if opts.subdivide_levels > 0:
        stages.append("F_subdivide")
    if opts.laplacian_iters > 0:
        stages.append("F_laplacian")
    if has_bc:
        stages.append("G_flatten")
    if opts.vf_rescale:
        stages.append("G_vf_rescale")
    stages.append("H_validate")
    return stages


def voxel_to_surface(
    vox: np.ndarray,
    bcspecs: list | None = None,
    *,
    bc_rows: np.ndarray | None = None,
    load_rows: np.ndarray | None = None,
    grid_shape: np.ndarray | None = None,
    origin=(0.0, 0.0, 0.0),
    spacing=(1.0, 1.0, 1.0),
    options: PipelineOptions | None = None,
    log: LogFn | None = None,
    **kwargs,
) -> SurfaceResult:
    """
    Build a tagged surface mesh from a binary voxel grid.

    Pass ``options=PipelineOptions(...)`` or individual keyword overrides
    (e.g. ``upsample_factor=1``, ``laplacian_iters=5``).
    """
    opts = options or PipelineOptions()
    for key, value in kwargs.items():
        if not hasattr(opts, key):
            raise TypeError(f"Unknown pipeline option: {key}")
        setattr(opts, key, value)

    spacing = np.asarray(spacing, float)
    origin = np.asarray(origin, float)
    coarse_shape = np.asarray(
        grid_shape if grid_shape is not None else vox.shape,
        dtype=int,
    )
    specs = list(bcspecs or [])
    log = _noop_log if log is None else log
    reports: list[PipelineStageReport] = []

    vf_target = (
        float(opts.vf_target)
        if opts.vf_target is not None
        else volume_fraction_voxels(vox)
    )
    domain_vol = domain_volume(coarse_shape, spacing)
    stages = _enabled_stages(opts, has_bc=bool(specs))
    n_stages = len(stages)
    step = 0

    def header(name: str, detail: str = "") -> None:
        nonlocal step
        step += 1
        suffix = f" — {detail}" if detail else ""
        log(f"  [{step}/{n_stages}] {name}{suffix}")

    # --- A: upsample ----------------------------------------------------- #
    vox_coarse = np.asarray(vox)
    if opts.upsample_factor > 1:
        header(
            "A upsample",
            f"factor={opts.upsample_factor} order={opts.upsample_order}",
        )
        vox_work = upsample_field(
            vox_coarse.astype(float),
            opts.upsample_factor,
            order=opts.upsample_order,
        )
        vox_work = (vox_work >= 0.5).astype(np.uint8)
        work_shape = np.array(vox_work.shape, dtype=int)
        work_spacing = spacing / opts.upsample_factor
        vf_up = volume_fraction_voxels(vox_work)
        report = field_stage_report(
            "A_upsample",
            volume_fraction=vf_up,
            vf_target=vf_target,
            details=(
                f"grid={tuple(int(x) for x in work_shape)}",
                f"vf_coarse={volume_fraction_voxels(vox_coarse):.6f}",
            ),
        )
        reports.append(report)
        _log_stage(log, report)
    else:
        vox_work = vox_coarse
        work_shape = coarse_shape
        work_spacing = spacing
        log("  [A upsample] skipped (upsample_factor=1)")

    # --- B: SDF ---------------------------------------------------------- #
    header("B signed distance field")
    sdf = signed_distance_field(vox_work, work_spacing)
    vf_sdf0 = field_volume_fraction(sdf, 0.0)
    report = field_stage_report(
        "B_sdf",
        volume_fraction=vf_sdf0,
        vf_target=vf_target,
        details=(
            f"grid={tuple(int(x) for x in work_shape)}",
            f"sdf_range=[{float(sdf.min()):.4g},{float(sdf.max()):.4g}]",
        ),
    )
    reports.append(report)
    _log_stage(log, report)

    # --- C: field refine ------------------------------------------------- #
    do_refine = opts.field_smooth_sigma > 0 or opts.calibrate_vf
    iso_level = 0.0
    if do_refine:
        header(
            "C field refine",
            f"sigma={opts.field_smooth_sigma} calibrate={opts.calibrate_vf}",
        )
        sdf, iso_level, meta = refine_sdf_field(
            sdf,
            vf_target,
            smooth_sigma=opts.field_smooth_sigma,
            calibrate=opts.calibrate_vf,
            vf_tol=min(opts.vf_tol, 1e-3),
        )
        report = field_stage_report(
            "C_field_refine",
            volume_fraction=float(meta["vf_calibrated"]),
            vf_target=vf_target,
            details=(
                f"iso_level={iso_level:.6g}",
                f"vf_input={meta['vf_input']:.6f}",
            ),
        )
        reports.append(report)
        _log_stage(log, report)
    else:
        iso_level = 0.0

    # --- D: extract ------------------------------------------------------ #
    header(
        "D marching cubes",
        f"iso={iso_level:.6g}" + (" + pymeshfix" if opts.repair else ""),
    )
    mesh, removed_pre = extract_surface_mesh(
        sdf,
        origin,
        work_spacing,
        iso_level=iso_level,
        repair=opts.repair,
        meshfix_verbose=opts.meshfix_verbose,
        domain_vol=domain_vol,
        vf_target=vf_target,
        log=log,
    )
    report = mesh_stage_report(mesh, domain_vol, "D_extract", vf_target=vf_target)
    reports.append(report)
    _log_stage(log, report)
    if removed_pre:
        log(f"        pre_meshfix removed {removed_pre:,} faces")

    # --- E: BC patches + transfer -------------------------------------- #
    bc_patches: list = []
    tri_labels = np.full(len(mesh.faces), -1, dtype=np.int32)
    if specs:
        from lib.converters.bc_patch import _exposed_faces, build_patch

        header("E BC patches", f"{len(specs)} spec(s) on coarse grid")
        exposed = _exposed_faces(vox_coarse)
        for spec in specs:
            patch = build_patch(
                vox_coarse,
                spec,
                origin,
                spacing,
                exposed=exposed,
                bc_rows=bc_rows,
                shape=coarse_shape,
            )
            if patch.faces:
                bc_patches.append(patch)
                log(
                    f"        patch {len(bc_patches) - 1}: axis={patch.axis} "
                    f"side={patch.side} cells={len(patch.faces)}",
                )
        header("E BC transfer", "coarse voxel footprints")
        tri_labels = transfer_patches(
            mesh,
            bc_patches,
            origin,
            spacing,
            coarse_shape,
        )
        n_bc_tris = int(np.count_nonzero(tri_labels >= 0))
        log(
            f"        {n_bc_tris:,} BC triangles, "
            f"{len(tri_labels) - n_bc_tris:,} free-boundary triangles",
        )
        report = mesh_stage_report(
            mesh,
            domain_vol,
            "E_bc_transfer",
            vf_target=vf_target,
        )
        report = PipelineStageReport(
            name="E_bc_transfer",
            volume_fraction=report.volume_fraction,
            vf_target=vf_target,
            watertight=report.watertight,
            bodies=report.bodies,
            degenerate_faces=report.degenerate_faces,
            n_vertices=report.n_vertices,
            n_faces=report.n_faces,
            details=(f"bc_triangles={n_bc_tris}",),
        )
        reports.append(report)
        _log_stage(log, report)

    patches = bc_patches

    # --- F: mesh refine -------------------------------------------------- #
    if opts.subdivide_levels > 0:
        header("F subdivide", f"levels={opts.subdivide_levels}")
        n_verts_before = len(mesh.vertices)
        n_faces_before = len(mesh.faces)
        mesh = subdivide_mesh(mesh, opts.subdivide_levels)
        if bc_patches:
            tri_labels = transfer_patches(
                mesh,
                bc_patches,
                origin,
                spacing,
                coarse_shape,
            )
        report = mesh_stage_report(
            mesh,
            domain_vol,
            "F_subdivide",
            vf_target=vf_target,
        )
        report = PipelineStageReport(
            name="F_subdivide",
            volume_fraction=report.volume_fraction,
            vf_target=vf_target,
            watertight=report.watertight,
            bodies=report.bodies,
            degenerate_faces=report.degenerate_faces,
            n_vertices=report.n_vertices,
            n_faces=report.n_faces,
            details=(
                f"verts {n_verts_before:,}->{len(mesh.vertices):,}",
                f"faces {n_faces_before:,}->{len(mesh.faces):,}",
            ),
        )
        reports.append(report)
        _log_stage(log, report)

    if opts.laplacian_iters > 0:
        header(
            "F Laplacian smooth",
            f"iters={opts.laplacian_iters} relax={opts.laplacian_relaxation}",
        )
        if bc_patches and np.any(tri_labels >= 0):
            smooth_weights, n_interior, n_frozen = bc_interior_freeze_weights(
                mesh,
                tri_labels,
                len(bc_patches),
                transition_rings=opts.laplacian_freeze_rings,
            )
            log(
                f"        freeze BC interiors: {n_interior:,} faces, "
                f"{n_frozen:,} vertices",
            )
        else:
            smooth_weights = np.zeros(len(mesh.vertices), dtype=float)
            n_interior, n_frozen = 0, 0

        constrained_laplacian_smooth(
            mesh,
            smooth_weights,
            bc_patches,
            tri_labels,
            iterations=opts.laplacian_iters,
            relaxation=opts.laplacian_relaxation,
            constrain_bc_planes=opts.constrain_bc_planes,
        )
        report = mesh_stage_report(
            mesh,
            domain_vol,
            "F_laplacian",
            vf_target=vf_target,
        )
        report = PipelineStageReport(
            name="F_laplacian",
            volume_fraction=report.volume_fraction,
            vf_target=vf_target,
            watertight=report.watertight,
            bodies=report.bodies,
            degenerate_faces=report.degenerate_faces,
            n_vertices=report.n_vertices,
            n_faces=report.n_faces,
            details=(
                f"frozen_verts={n_frozen}",
                f"constrain_planes={opts.constrain_bc_planes}",
            ),
        )
        reports.append(report)
        _log_stage(log, report)

    # --- G: flatten ------------------------------------------------------ #
    freeze = np.zeros(len(mesh.vertices), dtype=float)
    if bc_patches:
        header("G flatten BC planes")
        vmap = vertex_patch_map(mesh, tri_labels, len(patches))
        flatten_bc_planes(mesh, vmap, patches)
        freeze = (vmap >= 0).astype(float)
        n_patch_verts = int(np.count_nonzero(vmap >= 0))
        log(f"        {n_patch_verts:,} patch vertices flattened")
        report = mesh_stage_report(
            mesh,
            domain_vol,
            "G_flatten",
            vf_target=vf_target,
        )
        reports.append(report)
        _log_stage(log, report)

    if opts.vf_rescale and mesh.is_watertight:
        vf_before = float(mesh.volume) / domain_vol
        if abs(vf_before - vf_target) > opts.vf_rescale_min_drift:
            header("G vf rescale", f"{vf_before:.6f} -> target {vf_target:.6f}")
            rescale_mesh_volume(mesh, vf_target * domain_vol)
            vf_after = float(mesh.volume) / domain_vol
            report = mesh_stage_report(
                mesh,
                domain_vol,
                "G_vf_rescale",
                vf_target=vf_target,
            )
            reports.append(report)
            _log_stage(log, report)

    # --- H: validate ----------------------------------------------------- #
    header("H validate")
    final_report = mesh_stage_report(
        mesh,
        domain_vol,
        "H_final",
        vf_target=vf_target,
    )
    reports.append(final_report)
    _log_stage(log, final_report)

    if opts.enforce_vf_tol and final_report.volume_fraction is not None:
        drift = abs(final_report.volume_fraction - vf_target)
        if drift > opts.vf_tol + 1e-12:
            raise ValueError(
                f"Volume fraction drift {drift:.6f} exceeds vf_tol={opts.vf_tol} "
                f"(vf={final_report.volume_fraction:.6f}, target={vf_target:.6f})",
            )

    if (
        opts.check_loads
        and load_rows is not None
        and np.atleast_2d(load_rows).size > 0
    ):
        from lib.nito_physics import require_loads_on_surface

        log(f"        load surface check (tol={opts.load_surface_tol_cells} cells)")
        checks = require_loads_on_surface(
            mesh,
            load_rows,
            coarse_shape,
            spacing=spacing,
            max_distance_cells=opts.load_surface_tol_cells,
        )
        for check in checks:
            anchor = check.nito_anchor
            log(
                f"        load {check.index}: distance={check.distance:.4g} "
                f"nito=({anchor[0]:.3g}, {anchor[1]:.3g}, {anchor[2]:.3g})",
            )

    return SurfaceResult(
        mesh=mesh,
        patches=patches,
        tri_labels=tri_labels,
        freeze=freeze,
        iso_level=iso_level,
        vf_target=vf_target,
        stage_reports=reports,
    )


class SurfaceOutputFormat(str, Enum):
    """On-disk surface format (VTK PolyData only)."""

    VTP = "vtp"
    VTK = "vtk"


def save_surface(
    result: SurfaceResult,
    path: Path,
    *,
    fmt: SurfaceOutputFormat | str | None = None,
) -> Path:
    """Write a tagged VTK surface (.vtp default) for Gmsh, TetGen, and FEA."""
    from lib.meshing.surface_tags import write_surface_vtp

    fmt_str = None if fmt is None else str(fmt).lower().lstrip(".")
    return write_surface_vtp(result, path, fmt=fmt_str)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract a diagnostic surface from a voxel topology (v2 pipeline).",
    )
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--test", action="store_true")
    p.add_argument(
        "--format",
        choices=tuple(f.value for f in SurfaceOutputFormat),
        default=None,
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output path (default: public/vtp/<index>.vtp)",
    )
    p.add_argument("--density-cutoff", type=float, default=0.5)
    p.add_argument(
        "--no-bc",
        action="store_true",
        help="Skip NITO BC patch transfer (geometry-only surface)",
    )

    g = p.add_argument_group("field refinement (stages A–C)")
    g.add_argument(
        "--upsample-factor",
        type=int,
        default=2,
        help="Trilinear upsample per axis before SDF (1=off, default 2)",
    )
    g.add_argument("--upsample-order", type=int, default=1, choices=(0, 1, 3))
    g.add_argument(
        "--field-smooth-sigma",
        type=float,
        default=0.35,
        help="Gaussian SDF smooth width in upsampled cells (0=off)",
    )
    g.add_argument(
        "--no-calibrate-vf",
        action="store_true",
        help="Skip iso-level bisection for volume-fraction target",
    )
    g.add_argument("--vf-target", type=float, default=None)
    g.add_argument("--vf-tol", type=float, default=0.02)
    g.add_argument(
        "--enforce-vf-tol",
        action="store_true",
        help="Raise when final |vf - target| exceeds --vf-tol",
    )

    g = p.add_argument_group("extract (stage D)")
    g.add_argument("--no-repair", action="store_true")
    g.add_argument("--meshfix-verbose", action="store_true")

    g = p.add_argument_group("mesh refinement (stage F)")
    g.add_argument("--subdivide-levels", type=int, default=0)
    g.add_argument("--laplacian-iters", type=int, default=0)
    g.add_argument("--laplacian-relaxation", type=float, default=0.3)
    g.add_argument(
        "--no-constrain-bc-planes",
        action="store_true",
        help="Disable per-iteration BC plane projection during Laplacian",
    )
    g.add_argument("--laplacian-freeze-rings", type=int, default=0)
    g.add_argument(
        "--no-vf-rescale",
        action="store_true",
        help="Skip uniform volume rescale after Laplacian smooth",
    )
    g.add_argument("--vf-rescale-min-drift", type=float, default=0.005)

    g = p.add_argument_group("validation (stage H)")
    g.add_argument("--load-surface-tol-cells", type=float, default=1.0)
    g.add_argument("--no-load-check", action="store_true")

    p.add_argument("--log-file", type=Path, default=None)
    p.add_argument("--log-dir", type=Path, default=None)
    p.add_argument("-q", "--quiet", action="store_true")
    return p.parse_args(argv)


def _resolve_log_path(args: argparse.Namespace) -> Path | None:
    if args.log_file is not None:
        return args.log_file.expanduser()
    if args.log_dir is not None:
        return args.log_dir.expanduser() / f"{args.index}.log"
    return None


def _vox_from_rho(rho: np.ndarray, shape: np.ndarray, cutoff: float) -> np.ndarray:
    vox = np.asarray(rho, dtype=float).reshape(tuple(int(x) for x in shape), order="C")
    return (vox >= cutoff).astype(np.uint8)


def _load_bcspecs(bc: np.ndarray, shape: np.ndarray, *, use_bc: bool) -> list:
    if not use_bc:
        return []
    from lib.converters.bc_patch import bcspecs_from_nito

    return bcspecs_from_nito(bc, shape)


def _options_from_args(args: argparse.Namespace) -> PipelineOptions:
    return PipelineOptions(
        upsample_factor=args.upsample_factor,
        upsample_order=args.upsample_order,
        field_smooth_sigma=args.field_smooth_sigma,
        calibrate_vf=not args.no_calibrate_vf,
        vf_target=args.vf_target,
        vf_tol=args.vf_tol,
        enforce_vf_tol=args.enforce_vf_tol,
        repair=not args.no_repair,
        meshfix_verbose=args.meshfix_verbose,
        subdivide_levels=args.subdivide_levels,
        laplacian_iters=args.laplacian_iters,
        laplacian_relaxation=args.laplacian_relaxation,
        constrain_bc_planes=not args.no_constrain_bc_planes,
        laplacian_freeze_rings=args.laplacian_freeze_rings,
        vf_rescale=not args.no_vf_rescale,
        vf_rescale_min_drift=args.vf_rescale_min_drift,
        load_surface_tol_cells=args.load_surface_tol_cells,
        check_loads=not args.no_load_check,
    )


def main(argv: list[str] | None = None) -> None:
    from lib.nito_io import load_nito_arrays, load_sample, resolve_data_dir
    from lib.surface_io import DEFAULT_SURFACE_DIR, vtp_path

    args = _parse_args(argv)
    verbose = not args.quiet
    log_path = _resolve_log_path(args)
    logger = PipelineLogger(echo_stdout=verbose, log_path=log_path)
    data_dir = resolve_data_dir(args.data_dir, test=args.test)
    shape, rho, bc, load, vf_nito = load_sample(load_nito_arrays(data_dir), args.index)

    out_path = args.output or vtp_path(DEFAULT_SURFACE_DIR, args.index)
    vox = _vox_from_rho(rho, shape, args.density_cutoff)
    spacing = np.ones(int(shape.size), dtype=float)
    use_bc = not args.no_bc
    opts = _options_from_args(args)

    logger(f"voxel2surf v2  index={args.index}")
    logger(f"  data_dir={data_dir}")
    logger(f"  shape={tuple(int(x) for x in shape)}  solid_voxels={int(np.count_nonzero(vox)):,}")
    logger(f"  density_cutoff={args.density_cutoff}  bc_patches={'on' if use_bc else 'off'}")
    logger(
        f"  field: upsample={opts.upsample_factor} sigma={opts.field_smooth_sigma} "
        f"calibrate_vf={opts.calibrate_vf}",
    )
    logger(
        f"  mesh: subdivide={opts.subdivide_levels} laplacian_iters={opts.laplacian_iters}",
    )
    if opts.check_loads:
        logger(f"  load_surface_check=tol {opts.load_surface_tol_cells} cells")
    logger(f"  output={out_path.resolve()}")
    if log_path is not None:
        logger(f"  log={log_path.resolve()}")

    result = voxel_to_surface(
        vox,
        _load_bcspecs(bc, shape, use_bc=use_bc),
        bc_rows=bc if use_bc else None,
        load_rows=None if args.no_load_check else load,
        grid_shape=shape,
        spacing=spacing,
        options=opts,
        log=logger,
    )

    logger("  write tagged VTK …")
    out = save_surface(result, out_path, fmt=args.format)

    n_patches = len(result.patches)
    n_labeled = int(np.count_nonzero(result.tri_labels >= 0))
    vf_lines = format_volume_fraction_lines(
        vox=vox,
        mesh=result.mesh,
        shape=shape,
        spacing=spacing,
        vf_nito=vf_nito,
    )
    summary = (
        f"done  {out}  ({result.mesh.vertices.shape[0]:,} vertices, "
        f"{n_patches} BC patches, {n_labeled:,} labeled triangles, "
        f"iso={result.iso_level:.6g})",
    )
    logger(summary)
    for line in vf_lines:
        logger(line)
    logger.save()

    if not verbose:
        print(summary)
        for line in vf_lines:
            print(line)


if __name__ == "__main__":
    main()
