#!/usr/bin/env python3
"""
PyVista viewer for NITO surface meshes (.vtp distribution format).

Colors the surface by a point scalar (default: z coordinate, or an array stored
in the .vtp). Intended to extend to shape-derivative sensitivity fields from
scripts/surface/sensitivity/main.py.

Examples:

    python scripts/surface/visualize.py --index 0
    python scripts/surface/visualize.py --surface-file public/vtp/0.vtp --scalar dJ_dn
    python scripts/surface/visualize.py --index 0 --save output/figures/surface_0_z.png --no-show
    python scripts/surface/visualize.py --surface-file public/vtp/0.vtp --list-scalars
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyvista as pv

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from lib.paths import ensure_scripts_on_path

ensure_scripts_on_path()

from lib.surface_io import (
    COORD_SCALAR_FIELDS,
    DEFAULT_SURFACE_DIR,
    load_surface,
    point_scalar_names,
    resolve_scalar_field,
    resolve_surface_path,
)


def plot_surface(
    mesh: pv.PolyData,
    *,
    scalar_name: str,
    title: str,
    cmap: str,
    show: bool,
    save_path: Path | None,
) -> None:
    mesh = mesh.compute_normals(cell_normals=False, inplace=False)

    off_screen = save_path is not None and not show
    plotter = pv.Plotter(off_screen=off_screen)
    plotter.set_background("white")
    plotter.add_mesh(
        mesh,
        scalars=scalar_name,
        cmap=cmap,
        smooth_shading=True,
        show_scalar_bar=True,
        scalar_bar_args={"title": scalar_name, "vertical": True},
        ambient=0.35,
        diffuse=0.65,
        specular=0.25,
    )
    plotter.add_axes()
    plotter.add_text(title, position="upper_left", font_size=10, color="black")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if show:
            print(f"Opening viewer (saving screenshot to {save_path} on close) …")
            plotter.show(screenshot=str(save_path), auto_close=False)
        else:
            plotter.screenshot(str(save_path))
            plotter.close()
        print(f"Saved {save_path}")
    elif show:
        print("Opening PyVista viewer (close the window to exit) …")
        plotter.show(auto_close=False)
    else:
        plotter.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize a .vtp surface colored by point scalars."
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--index", type=int, help="NITO dataset index → <surface-dir>/<index>.vtp"
    )
    src.add_argument(
        "--surface-file", type=Path, help="Path to .vtp (or legacy .stl)"
    )
    p.add_argument(
        "--surface-dir",
        type=Path,
        default=DEFAULT_SURFACE_DIR,
        help=f"VTP distribution directory (default: {DEFAULT_SURFACE_DIR})",
    )
    p.add_argument(
        "--scalar",
        type=str,
        default="z",
        help="Point array name, or x/y/z coordinate (default: z)",
    )
    p.add_argument(
        "--list-scalars",
        action="store_true",
        help="Print point_data array names and exit",
    )
    p.add_argument("--cmap", type=str, default="turbo", help="Colormap name")
    p.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Screenshot path (e.g. output/figures/surface_0_z.png)",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Skip interactive window (use with --save)",
    )
    p.add_argument("--title", type=str, default=None, help="Plot title override")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    path = resolve_surface_path(
        index=args.index,
        surface_file=args.surface_file,
        surface_dir=args.surface_dir,
    )
    print(f"Loading {path} …")
    mesh = load_surface(path)
    print(f"  {mesh.n_points:,} points, {mesh.n_cells:,} cells")

    if args.list_scalars:
        names = point_scalar_names(mesh)
        coord = ", ".join(sorted(COORD_SCALAR_FIELDS))
        print("Point arrays:", names or "(none — use x/y/z coordinates)")
        print(f"Coordinate scalars always available: {coord}")
        return

    mesh, scalar_name = resolve_scalar_field(mesh, args.scalar)

    title = args.title or f"{path.name} — {scalar_name}"
    show = not args.no_show
    if show and args.save is None and not sys.stdout.isatty():
        print(
            "Note: non-interactive terminal — if no window appears, try:\n"
            "  python scripts/surface/visualize.py --index 0 "
            "--save output/figures/surface_0_z.png --no-show"
        )

    plot_surface(
        mesh,
        scalar_name=scalar_name,
        title=title,
        cmap=args.cmap,
        show=show,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
