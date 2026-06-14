#!/usr/bin/env python3
"""
Check volume fraction for a NITO voxel grid, STL, or VTP surface mesh.

Examples:

    # Voxel solid fraction from NITO index
    python scripts/volume/check.py voxel --index 119 --data-dir nito/Data/3D

    # STL / VTP against the same design-domain box (index supplies shape + nito vf)
    python scripts/volume/check.py stl --index 119
    python scripts/volume/check.py vtp --index 119
    python scripts/volume/check.py vtp --file public/vtp/119.vtp --index 119

    # Surface file without NITO index (explicit domain box)
    python scripts/volume/check.py vtp --file mesh.vtp --shape 64 64 64

    # Compare voxel, STL, and VTP for one index
    python scripts/volume/check.py compare --index 119 --data-dir nito/Data/3D
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from lib.paths import ensure_scripts_on_path

ensure_scripts_on_path()

from lib.nito_io import load_nito_arrays, load_sample, resolve_data_dir
from lib.stl_common import DEFAULT_OUTPUT_DIR as DEFAULT_STL_DIR, stl_path
from lib.surface_io import DEFAULT_SURFACE_DIR, vtp_path
from lib.volume_check import (
    VolumeReport,
    binarize_topology,
    check_surface_file,
    check_voxels,
    default_spacing,
    format_compare_lines,
)


def _load_nito_context(
    *,
    index: int,
    data_dir: Path | None,
    test: bool,
    density_cutoff: float,
) -> tuple:
    data_dir = resolve_data_dir(data_dir, test=test)
    shape, rho, _bc, _load, vf = load_sample(load_nito_arrays(data_dir), index)
    vox = binarize_topology(rho, shape, cutoff=density_cutoff)
    spacing = default_spacing(shape)
    return shape, spacing, vox, vf, data_dir


def _resolve_surface_path(
    *,
    kind: str,
    index: int | None,
    path: Path | None,
    stl_dir: Path,
    surface_dir: Path,
) -> Path:
    if path is not None:
        return path.expanduser().resolve()
    if index is None:
        raise SystemExit(f"--file is required for {kind} when --index is omitted")
    if kind == "stl":
        candidate = stl_path(stl_dir.resolve(), index)
    else:
        candidate = vtp_path(surface_dir.resolve(), index)
    if not candidate.is_file():
        raise SystemExit(f"Surface not found: {candidate}")
    return candidate


def _print_report(report: VolumeReport) -> None:
    for line in report.format_lines():
        print(line)


def _domain_from_args(args: argparse.Namespace) -> tuple:
    shape = tuple(int(x) for x in args.shape) if args.shape is not None else None
    spacing = (
        np.asarray(args.spacing, dtype=float)
        if args.spacing is not None
        else None
    )
    vf_nito = None
    if args.index is not None:
        shape_n, spacing_n, _vox, vf_nito, data_dir = _load_nito_context(
            index=args.index,
            data_dir=args.data_dir,
            test=args.test,
            density_cutoff=args.density_cutoff,
        )
        if shape is None:
            shape = tuple(int(x) for x in shape_n)
        if spacing is None:
            spacing = spacing_n
        print(f"data_dir={data_dir}")
    if shape is None:
        raise SystemExit("--shape is required when --index is not provided")
    if spacing is None:
        spacing = default_spacing(shape)
    return shape, spacing, vf_nito


def cmd_voxel(args: argparse.Namespace) -> None:
    if args.index is None:
        raise SystemExit("voxel check requires --index")
    shape, spacing, vox, vf_nito, data_dir = _load_nito_context(
        index=args.index,
        data_dir=args.data_dir,
        test=args.test,
        density_cutoff=args.density_cutoff,
    )
    print(f"data_dir={data_dir}")
    _print_report(
        check_voxels(vox, shape, spacing, index=args.index, vf_nito=vf_nito)
    )


def cmd_surface(args: argparse.Namespace, *, kind: str) -> None:
    shape, spacing, vf_nito = _domain_from_args(args)
    path = _resolve_surface_path(
        kind=kind,
        index=args.index,
        path=args.file,
        stl_dir=args.stl_dir,
        surface_dir=args.surface_dir,
    )
    _print_report(
        check_surface_file(
            path,
            np.asarray(shape, dtype=int),
            np.asarray(spacing, dtype=float),
            index=args.index,
            vf_nito=vf_nito,
        )
    )


def cmd_compare(args: argparse.Namespace) -> None:
    if args.index is None:
        raise SystemExit("compare requires --index")

    shape, spacing, vox, vf_nito, data_dir = _load_nito_context(
        index=args.index,
        data_dir=args.data_dir,
        test=args.test,
        density_cutoff=args.density_cutoff,
    )
    print(f"data_dir={data_dir}")
    reports: list[VolumeReport] = [
        check_voxels(vox, shape, spacing, index=args.index, vf_nito=vf_nito)
    ]

    stl = stl_path(args.stl_dir.resolve(), args.index)
    if stl.is_file():
        reports.append(
            check_surface_file(
                stl,
                shape,
                spacing,
                index=args.index,
                vf_nito=vf_nito,
            )
        )
    else:
        print(f"skip stl (missing): {stl}")

    vtp = vtp_path(args.surface_dir.resolve(), args.index)
    if vtp.is_file():
        reports.append(
            check_surface_file(
                vtp,
                shape,
                spacing,
                index=args.index,
                vf_nito=vf_nito,
            )
        )
    else:
        print(f"skip vtp (missing): {vtp}")

    print()
    for report in reports:
        _print_report(report)
        print()
    print("comparison:")
    for line in format_compare_lines(reports, baseline="voxel"):
        print(f"  {line}")


def _add_domain_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--index", type=int, default=None)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--test", action="store_true", help="Use nito/Data/Test/")
    p.add_argument(
        "--density-cutoff",
        type=float,
        default=0.5,
        help="Voxels with density >= cutoff are solid (voxel / compare only)",
    )
    p.add_argument(
        "--shape",
        type=int,
        nargs="+",
        default=None,
        help="Design-domain shape (required without --index)",
    )
    p.add_argument(
        "--spacing",
        type=float,
        nargs="+",
        default=None,
        help="Voxel spacing (default: 1 per axis)",
    )
    p.add_argument("--stl-dir", type=Path, default=DEFAULT_STL_DIR)
    p.add_argument("--surface-dir", type=Path, default=DEFAULT_SURFACE_DIR)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check volume fraction for voxel / STL / VTP.")
    sub = p.add_subparsers(dest="command", required=True)

    pv = sub.add_parser("voxel", help="NITO voxel solid fraction")
    _add_domain_args(pv)
    pv.set_defaults(func=cmd_voxel)

    ps = sub.add_parser("stl", help="STL enclosed volume vs design domain")
    _add_domain_args(ps)
    ps.add_argument("--file", type=Path, default=None, help="STL path (default: public/stl/<index>.stl)")
    ps.set_defaults(func=lambda a: cmd_surface(a, kind="stl"))

    pvtp = sub.add_parser("vtp", help="VTP enclosed volume vs design domain")
    _add_domain_args(pvtp)
    pvtp.add_argument("--file", type=Path, default=None, help="VTP path (default: public/vtp/<index>.vtp)")
    pvtp.set_defaults(func=lambda a: cmd_surface(a, kind="vtp"))

    pc = sub.add_parser("compare", help="Voxel + STL + VTP for one NITO index")
    _add_domain_args(pc)
    pc.set_defaults(func=cmd_compare)

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
