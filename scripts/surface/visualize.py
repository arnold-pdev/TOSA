#!/usr/bin/env python3
"""
PyVista viewer for NITO / voxel2surf surface meshes (.vtp).

Colors the surface by point scalars (x/y/z, sensitivity fields) or by VTK
boundary tags from voxel2surf (patch_id → boundary type per triangle).

Examples:

    python scripts/surface/visualize.py --index 0
    python scripts/surface/visualize.py --surface-file out/119.vtp --scalar boundary
    python scripts/surface/visualize.py --surface-file out/119.vtp --scalar patch_id
    python scripts/surface/visualize.py --index 119 --scalar boundary --overlay-voxels --data-dir nito/Data/3D
    python scripts/surface/visualize.py --index 119 --surface-file output/surfaces/validation/.../extract_only.vtp --scalar boundary --overlay-voxels --data-dir nito/Data/3D
    # Interactive keys: v=voxels+BC faces  s=surface  l=load markers
    python scripts/surface/visualize.py --surface-file public/vtp/119.vtp --scalar z --data-dir nito/Data/3D
    python scripts/surface/visualize.py --surface-file out/119.vtp --list-scalars
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh
from matplotlib import colors

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from lib.paths import ensure_scripts_on_path

ensure_scripts_on_path()

from lib.meshing.surface_tags import (
    PATCH_ID_KEY,
    PatchMetadata,
    merge_patch_display_order,
    nito_patch_display_order,
    patch_categorical_cmap,
    patch_display_color_hex,
    patch_display_order,
    patch_label,
    prepare_patch_display,
    try_read_surface_tags,
)
from lib.converters.bc_patch import build_patches_from_nito, project_point_to_mesh
from lib.nito_io import load_nito_arrays, load_sample, resolve_data_dir
from lib.nito_physics import load_anchor_positions, marker_sphere_radius
from lib.surface_io import (
    COORD_SCALAR_FIELDS,
    DEFAULT_SURFACE_DIR,
    cell_scalar_names,
    has_patch_tags,
    load_surface,
    point_scalar_names,
    resolve_scalar_field,
    resolve_surface_path,
)

BOUNDARY_SCALARS = frozenset(
    {"boundary", "boundary_type", "patches", "patch_id", "patch_display"}
)


def _surface_load_markers(mesh: pv.PolyData, load: np.ndarray, shape: np.ndarray) -> np.ndarray:
    """NITO load anchors projected onto the displayed surface mesh."""
    anchors = load_anchor_positions(load, shape)
    if anchors.size == 0:
        return anchors
    faces = np.asarray(mesh.faces, dtype=int).reshape(-1, 4)[:, 1:4]
    tm = trimesh.Trimesh(
        vertices=np.asarray(mesh.points, dtype=float),
        faces=faces,
        process=False,
    )
    projected = []
    for pt in anchors:
        surf, _dist, _ = project_point_to_mesh(tm, pt)
        projected.append(surf)
    return np.asarray(projected, dtype=float)


def _nito_bc_patch_count(
    shape: np.ndarray,
    rho: np.ndarray,
    bc: np.ndarray,
    *,
    threshold: float = 0.5,
) -> int:
    """Number of NITO BC patches (matches voxel2surf / bc_face_overlays indexing)."""
    shape = np.asarray(shape, dtype=int).ravel()
    ndim = int(shape.size)
    origin = np.zeros(3, dtype=float)
    spacing = np.ones(3, dtype=float)
    design = np.asarray(rho, dtype=float).reshape(tuple(shape), order="C")
    vox = (design >= threshold).astype(np.uint8)
    patches = build_patches_from_nito(
        vox,
        bc,
        shape,
        origin[:ndim],
        spacing[:ndim],
    )
    return sum(1 for patch in patches if patch.is_bc)


def _resolve_patch_display_order(
    mesh: pv.PolyData,
    *,
    shape: np.ndarray | None = None,
    rho: np.ndarray | None = None,
    bc: np.ndarray | None = None,
    load: np.ndarray | None = None,
    threshold: float = 0.5,
    n_bc: int | None = None,
) -> list[int]:
    """Match boundary scalar coloring: tagged surface first, else NITO patch layout."""
    if has_patch_tags(mesh):
        pid = mesh.cell_data[PATCH_ID_KEY]
        if n_bc is None and shape is not None and rho is not None and bc is not None:
            n_bc = _nito_bc_patch_count(shape, rho, bc, threshold=threshold)
        if n_bc:
            return merge_patch_display_order(pid, n_bc=n_bc)
        return patch_display_order(pid)
    if shape is None or rho is None or bc is None or load is None:
        return [-1]
    if n_bc is None:
        n_bc = _nito_bc_patch_count(shape, rho, bc, threshold=threshold)
    return nito_patch_display_order(n_bc, 0)


def voxel_overlay_mesh(
    shape: np.ndarray,
    rho: np.ndarray,
    *,
    threshold: float = 0.5,
    origin=(0.0, 0.0, 0.0),
    spacing=(1.0, 1.0, 1.0),
) -> pv.UnstructuredGrid:
    """Semi-transparent voxel solid in the voxel2surf design frame."""
    shape = np.asarray(shape, dtype=int).ravel()
    spacing = np.asarray(spacing, dtype=float)
    origin = np.asarray(origin, dtype=float)
    design = np.asarray(rho, dtype=float).reshape(tuple(shape), order="C")
    grid = pv.ImageData()
    grid.dimensions = np.array(shape, dtype=int) + 1
    grid.origin = tuple(float(x) for x in origin)
    grid.spacing = tuple(float(x) for x in spacing)
    # VTK ImageData: x (shape[0]) varies fastest — match voxel/visualize.py.
    grid.cell_data["rho"] = design.ravel(order="F")
    return grid.threshold(threshold, scalars="rho")


def _inplane_to_cell(
    axis: int,
    side: str,
    inplane: tuple[int, ...],
    shape: np.ndarray,
) -> tuple[int, ...]:
    shape = np.asarray(shape, dtype=int).ravel()
    ndim = int(shape.size)
    slab = 0 if side == "min" else int(shape[axis]) - 1
    inplane_axes = [i for i in range(ndim) if i != axis]
    cell = [0] * ndim
    cell[axis] = slab
    for k, ax in enumerate(inplane_axes):
        cell[ax] = int(inplane[k])
    return tuple(cell)


def _voxel_face_quad(
    cell: tuple[int, ...],
    axis: int,
    side: str,
    origin: np.ndarray,
    spacing: np.ndarray,
) -> np.ndarray:
    """Four corners of the outward voxel face (3D; 2D cells get unit z extent)."""
    origin = np.asarray(origin, dtype=float).ravel()
    spacing = np.asarray(spacing, dtype=float).ravel()
    ndim = len(cell)
    if ndim < 3:
        origin = np.pad(origin, (0, 3 - ndim))
        spacing = np.pad(spacing, (0, 3 - ndim), constant_values=1.0)
    lo = origin[:3] + np.array(cell[:3], dtype=float) * spacing[:3]
    hi = lo + spacing[:3]
    if axis == 0:
        x = lo[0] if side == "min" else hi[0]
        return np.array(
            [
                [x, lo[1], lo[2]],
                [x, hi[1], lo[2]],
                [x, hi[1], hi[2]],
                [x, lo[1], hi[2]],
            ],
            dtype=float,
        )
    if axis == 1:
        y = lo[1] if side == "min" else hi[1]
        return np.array(
            [
                [lo[0], y, lo[2]],
                [hi[0], y, lo[2]],
                [hi[0], y, hi[2]],
                [lo[0], y, hi[2]],
            ],
            dtype=float,
        )
    z = lo[2] if side == "min" else hi[2]
    return np.array(
        [
            [lo[0], lo[1], z],
            [hi[0], lo[1], z],
            [hi[0], hi[1], z],
            [lo[0], hi[1], z],
        ],
        dtype=float,
    )


def bc_face_overlays(
    rho: np.ndarray,
    bc: np.ndarray,
    shape: np.ndarray,
    *,
    patch_order: Sequence[int],
    threshold: float = 0.5,
    origin=(0.0, 0.0, 0.0),
    spacing=(1.0, 1.0, 1.0),
) -> list[tuple[pv.PolyData, str, str]]:
    """Colored exterior voxel faces on NITO BC patches (for transparent overlay)."""
    shape = np.asarray(shape, dtype=int).ravel()
    ndim = int(shape.size)
    origin = np.asarray(origin, dtype=float).ravel()
    spacing = np.asarray(spacing, dtype=float).ravel()
    if ndim < 3:
        origin = np.pad(origin, (0, 3 - ndim))
        spacing = np.pad(spacing, (0, 3 - ndim), constant_values=1.0)

    design = np.asarray(rho, dtype=float).reshape(tuple(shape), order="C")
    vox = (design >= threshold).astype(np.uint8)
    patches = build_patches_from_nito(
        vox,
        bc,
        shape,
        origin[:ndim],
        spacing[:ndim],
    )

    overlays: list[tuple[pv.PolyData, str, str]] = []
    for idx, patch in enumerate(patches):
        if not patch.is_bc or not patch.faces:
            continue
        pts: list[list[float]] = []
        faces: list[int] = []
        pid = 0
        for inplane in patch.faces:
            cell = _inplane_to_cell(patch.axis, patch.side, inplane, shape)
            quad = _voxel_face_quad(cell, patch.axis, patch.side, origin, spacing)
            pts.extend(quad.tolist())
            faces.extend([4, pid, pid + 1, pid + 2, pid + 3])
            pid += 4
        if not pts:
            continue
        color = patch_display_color_hex(idx, patch_order)
        meta = PatchMetadata(
            flags=np.asarray(patch.flags, dtype=float),
            axis=int(patch.axis),
            side=str(patch.side),
        )
        label = patch_label(idx, meta)
        overlays.append((pv.PolyData(np.asarray(pts), faces), color, label))
    return overlays


def _resolve_nito_index(args: argparse.Namespace, path: Path) -> int | None:
    if args.index is not None:
        return int(args.index)
    stem = path.stem
    if stem.isdigit():
        return int(stem)
    return None


def _add_marker_spheres(
    plotter: pv.Plotter,
    points: np.ndarray,
    *,
    color: str,
    radius: float,
    label: str,
) -> object | None:
    if points.size == 0:
        return None
    cloud = pv.PolyData(np.asarray(points, dtype=float))
    geom = pv.Sphere(radius=radius, theta_resolution=16, phi_resolution=16)
    return plotter.add_mesh(
        cloud.glyph(geom=geom, scale=False),
        color=color,
        label=label,
    )


_TOGGLE_HELP = "Keys: [v] voxels+BC  [s] surface  [t] tri skeleton  [l] loads"


def _register_layer_toggles(
    plotter: pv.Plotter,
    layers: dict[str, list],
) -> None:
    """Keyboard visibility toggles (interactive viewer only)."""

    def _toggle(group: str) -> None:
        actors = layers.get(group)
        if not actors:
            return
        visible = not actors[0].GetVisibility()
        for actor in actors:
            actor.SetVisibility(visible)
        plotter.render()

    plotter.add_key_event("v", lambda: _toggle("voxels"))
    plotter.add_key_event("s", lambda: _toggle("surface"))
    plotter.add_key_event("t", lambda: _toggle("tri_skeleton"))
    plotter.add_key_event("l", lambda: _toggle("loads"))
    plotter.add_text(
        _TOGGLE_HELP,
        position="lower_right",
        font_size=8,
        color="dimgray",
        name="toggle_help",
    )


def plot_surface(
    mesh: pv.PolyData,
    *,
    scalar_name: str,
    title: str,
    cmap: str | colors.Colormap,
    show: bool,
    save_path: Path | None,
    scalar_location: str = "point",
    legend_labels: list[str] | None = None,
    categorical: bool = False,
    voxel_overlay: pv.UnstructuredGrid | None = None,
    voxel_opacity: float = 0.25,
    bc_faces: list[tuple[pv.PolyData, str, str]] | None = None,
    load_points: np.ndarray | None = None,
    marker_radius: float = 1.0,
    bc_face_opacity: float = 0.9,
    tri_skeleton_mesh: pv.PolyData | None = None,
    tri_skeleton_scalar: str | None = None,
    tri_skeleton_labels: list[str] | None = None,
    tri_skeleton_opacity: float = 0.95,
    tri_skeleton_line_width: float = 2.0,
) -> None:
    mesh = mesh.compute_normals(cell_normals=False, inplace=False)

    off_screen = save_path is not None and not show
    plotter = pv.Plotter(off_screen=off_screen)
    plotter.set_background("white")
    layers: dict[str, list] = {
        "voxels": [],
        "surface": [],
        "tri_skeleton": [],
        "loads": [],
    }

    if voxel_overlay is not None:
        layers["voxels"].append(
            plotter.add_mesh(
                voxel_overlay,
                color="tan",
                opacity=voxel_opacity,
                show_edges=True,
                label="voxels",
            )
        )

    if bc_faces:
        for face_mesh, color, label in bc_faces:
            layers["voxels"].append(
                plotter.add_mesh(
                    face_mesh,
                    color=color,
                    opacity=bc_face_opacity,
                    show_edges=True,
                    edge_color="white",
                    label=label,
                )
            )

    kwargs: dict = dict(
        scalars=scalar_name,
        cmap=cmap,
        smooth_shading=True,
        show_scalar_bar=not categorical,
        ambient=0.35,
        diffuse=0.65,
        specular=0.25,
    )
    if categorical and legend_labels:
        n = len(legend_labels)
        kwargs["cmap"] = cmap if isinstance(cmap, colors.Colormap) else patch_categorical_cmap(n)
        kwargs["clim"] = [-0.5, n - 0.5]
        kwargs["show_scalar_bar"] = False
    else:
        kwargs["scalar_bar_args"] = {"title": scalar_name, "vertical": True}

    layers["surface"].append(plotter.add_mesh(mesh, **kwargs))

    if tri_skeleton_mesh is not None and tri_skeleton_scalar:
        n_labels = len(tri_skeleton_labels or [])
        layers["tri_skeleton"].append(
            plotter.add_mesh(
                tri_skeleton_mesh,
                scalars=tri_skeleton_scalar,
                preference="cell",
                style="wireframe",
                cmap=patch_categorical_cmap(n_labels),
                clim=[-0.5, max(n_labels - 0.5, 0.5)],
                show_scalar_bar=False,
                line_width=tri_skeleton_line_width,
                opacity=tri_skeleton_opacity,
                render_lines_as_tubes=True,
                label="tri skeleton",
            )
        )

    if load_points is not None:
        actor = _add_marker_spheres(
            plotter,
            load_points,
            color="dodgerblue",
            radius=marker_radius,
            label="load anchors",
        )
        if actor is not None:
            layers["loads"].append(actor)

    if (
        tri_skeleton_mesh is not None
        or (bc_faces and len(bc_faces))
        or (load_points is not None and load_points.size)
    ):
        plotter.add_legend()

    plotter.add_axes()
    plotter.add_text(title, position="upper_left", font_size=10, color="black")

    legend_blocks: list[str] = []
    if categorical and legend_labels:
        legend_blocks.append(
            "\n".join(f"{i}: {label}" for i, label in enumerate(legend_labels))
        )
    if tri_skeleton_labels and (
        not categorical or tri_skeleton_labels != legend_labels
    ):
        legend_blocks.append(
            "tri skeleton:\n"
            + "\n".join(f"{i}: {label}" for i, label in enumerate(tri_skeleton_labels))
        )
    if legend_blocks:
        plotter.add_text(
            "\n\n".join(legend_blocks),
            position="lower_left",
            font_size=8,
            color="black",
        )

    interactive = show and not off_screen
    if interactive:
        _register_layer_toggles(plotter, layers)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if show:
            print(f"Opening viewer (saving screenshot to {save_path} on close) …")
            if interactive:
                print(f"  {_TOGGLE_HELP}")
            plotter.show(screenshot=str(save_path), auto_close=False)
        else:
            plotter.screenshot(str(save_path))
            plotter.close()
        print(f"Saved {save_path}")
    elif show:
        print("Opening PyVista viewer (close the window to exit) …")
        if interactive:
            print(f"  {_TOGGLE_HELP}")
        plotter.show(auto_close=False)
    else:
        plotter.close()


def _resolve_boundary_display(
    path: Path,
    scalar: str,
    *,
    n_bc: int = 0,
):
    tags = try_read_surface_tags(path)
    if tags is None:
        raise ValueError(
            f"{path} has no cell_data[{PATCH_ID_KEY!r}]. "
            "Run voxel2surf to produce a tagged VTK surface, e.g.\n"
            "  PYTHONPATH=scripts python -m lib.converters.voxel2surf.cli --index N -o out/N.vtp"
        )

    if n_bc > 0:
        order = merge_patch_display_order(tags.patch_id, n_bc=n_bc)
    else:
        order = patch_display_order(tags.patch_id)

    if scalar == "patch_id":
        mesh = tags.mesh.copy()
        mesh.cell_data["patch_id_display"] = np.asarray(tags.patch_id, dtype=np.int32)
        labels = [
            f"patch_id {v}" if v >= 0 else "free boundary (patch_id -1)"
            for v in order
        ]
        remap = {v: i for i, v in enumerate(order)}
        mesh.cell_data["patch_display"] = np.array(
            [remap[int(p)] for p in tags.patch_id], dtype=np.int32
        )
        return mesh, "patch_display", labels, True

    mesh, scalar_name, labels = prepare_patch_display(tags, order=order)
    return mesh, scalar_name, labels, True


def _resolve_tri_skeleton_display(
    path: Path,
    *,
    n_bc: int = 0,
) -> tuple[pv.PolyData, str, list[str]]:
    """Wireframe overlay colored with the same BC patch palette as boundary mode."""
    tags = try_read_surface_tags(path)
    if tags is None:
        raise ValueError(
            f"{path} has no cell_data[{PATCH_ID_KEY!r}]. "
            "--tri-skeleton requires a tagged voxel2surf VTK surface."
        )

    if n_bc > 0:
        order = merge_patch_display_order(tags.patch_id, n_bc=n_bc)
    else:
        order = patch_display_order(tags.patch_id)
    mesh, scalar_name, labels = prepare_patch_display(tags, order=order)
    return mesh, scalar_name, labels


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize a VTK surface (point scalars or BC patch tags)."
    )
    p.add_argument(
        "--index",
        type=int,
        default=None,
        help="NITO dataset index (loads <surface-dir>/<index>.vtp, or NITO voxels with --surface-file)",
    )
    p.add_argument(
        "--surface-file",
        type=Path,
        default=None,
        help="Path to .vtp / .vtk from voxel2surf (optional with --index for voxel overlay)",
    )
    p.add_argument(
        "--surface-dir",
        type=Path,
        default=DEFAULT_SURFACE_DIR,
        help=f"VTP directory (default: {DEFAULT_SURFACE_DIR})",
    )
    p.add_argument(
        "--scalar",
        type=str,
        default="z",
        help=(
            "Color field: point x/y/z array, point_data name, or boundary tag "
            f"{sorted(BOUNDARY_SCALARS)} from voxel2surf cell_data"
        ),
    )
    p.add_argument(
        "--list-scalars",
        action="store_true",
        help="Print point_data and cell_data array names and exit",
    )
    p.add_argument(
        "--tri-skeleton",
        action="store_true",
        help="Overlay the triangle wireframe colored by BC patch_id (requires tagged .vtp)",
    )
    p.add_argument(
        "--tri-skeleton-opacity",
        type=float,
        default=0.95,
        help="Opacity of the BC-colored triangle skeleton overlay (default 0.95)",
    )
    p.add_argument(
        "--tri-skeleton-line-width",
        type=float,
        default=2.0,
        help="Line width for the triangle skeleton overlay (default 2.0)",
    )
    p.add_argument("--cmap", type=str, default="turbo", help="Colormap (point mode)")
    p.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Screenshot path (e.g. output/figures/surface_0_boundary.png)",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Skip interactive window (use with --save)",
    )
    p.add_argument("--title", type=str, default=None, help="Plot title override")
    p.add_argument(
        "--overlay-voxels",
        action="store_true",
        help="Overlay binarized NITO topology voxels (requires NITO index + --data-dir)",
    )
    p.add_argument("--data-dir", type=Path, default=None, help="NITO data root for voxel overlay")
    p.add_argument("--test", action="store_true", help="Use nito/Data/Test/ with --overlay-voxels")
    p.add_argument(
        "--density-cutoff",
        type=float,
        default=0.5,
        help="Solid threshold for --overlay-voxels (match voxel2surf)",
    )
    p.add_argument(
        "--voxel-opacity",
        type=float,
        default=0.25,
        help="Opacity of voxel overlay (default 0.25)",
    )
    p.add_argument(
        "--no-markers",
        action="store_true",
        help="Skip NITO load anchor spheres",
    )
    p.add_argument(
        "--no-bc-faces",
        action="store_true",
        help="Skip BC-colored faces on the voxel overlay",
    )
    p.add_argument(
        "--bc-face-opacity",
        type=float,
        default=0.9,
        help="Opacity of BC-colored voxel faces (default 0.9)",
    )
    p.add_argument(
        "--marker-radius",
        type=float,
        default=None,
        help="Sphere radius for load markers (default: 4%% of max shape)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.index is None and args.surface_file is None:
        raise SystemExit("Provide --index and/or --surface-file")
    path = resolve_surface_path(
        index=args.index,
        surface_file=args.surface_file,
        surface_dir=args.surface_dir,
    )
    print(f"Loading {path} …")
    mesh = load_surface(path)
    print(f"  {mesh.n_points:,} points, {mesh.n_cells:,} cells")
    if has_patch_tags(mesh):
        print(f"  tagged surface ({PATCH_ID_KEY} on cells)")

    if args.list_scalars:
        pts = point_scalar_names(mesh)
        cells = cell_scalar_names(mesh)
        coord = ", ".join(sorted(COORD_SCALAR_FIELDS))
        print("Point arrays:", pts or "(none)")
        print("Cell arrays:", cells or "(none)")
        print(f"Coordinate scalars always available: {coord}")
        if has_patch_tags(mesh):
            print(f"Boundary scalars: {', '.join(sorted(BOUNDARY_SCALARS))}")
        return

    nito_index = _resolve_nito_index(args, path)
    nito_sample: tuple | None = None
    n_bc = 0
    want_bc_faces = not args.no_bc_faces and nito_index is not None
    want_load_markers = not args.no_markers and nito_index is not None
    want_voxels = args.overlay_voxels or want_bc_faces
    need_nito = want_voxels or want_load_markers

    if need_nito and nito_index is None:
        if args.overlay_voxels or want_bc_faces:
            raise SystemExit(
                "NITO overlay/BC faces require --index (NITO sample id) plus --data-dir. "
                "With validation VTPs named by recipe, pass both --index 119 and --surface-file."
            )
    elif nito_index is not None:
        data_dir = resolve_data_dir(args.data_dir, test=args.test)
        nito_sample = load_sample(load_nito_arrays(data_dir), nito_index)
        n_bc = _nito_bc_patch_count(
            nito_sample[0],
            nito_sample[1],
            nito_sample[2],
            threshold=args.density_cutoff,
        )

    scalar = args.scalar.lower()
    categorical = False
    legend_labels: list[str] | None = None
    cmap: str | colors.Colormap = args.cmap

    if scalar in BOUNDARY_SCALARS:
        mesh, scalar_name, legend_labels, categorical = _resolve_boundary_display(
            path,
            "patch_id" if scalar == "patch_id" else "boundary",
            n_bc=n_bc,
        )
        title = args.title or f"{path.name} — boundary type"
    elif scalar in mesh.cell_data:
        scalar_name = scalar
        categorical = scalar in {"patch_id", "patch_display", "facet_marker"}
        title = args.title or f"{path.name} — {scalar_name} (cell)"
    else:
        mesh, scalar_name = resolve_scalar_field(mesh, args.scalar)
        title = args.title or f"{path.name} — {scalar_name}"

    show = not args.no_show
    if show and args.save is None and not sys.stdout.isatty():
        print(
            "Note: non-interactive terminal — if no window appears, try:\n"
            "  python scripts/surface/visualize.py --index 0 "
            "--save output/figures/surface_0_z.png --no-show"
        )

    overlay = None
    bc_faces: list[tuple[pv.PolyData, str, str]] | None = None
    load_points = None
    marker_radius = 1.0
    tri_skeleton_mesh = None
    tri_skeleton_scalar = None
    tri_skeleton_labels = None

    if nito_sample is not None:
        shape, rho, bc, load, _vf = nito_sample
        marker_radius = (
            float(args.marker_radius)
            if args.marker_radius is not None
            else marker_sphere_radius(shape)
        )
        spacing = np.ones(max(3, int(shape.size)), dtype=float)
        if want_load_markers:
            load_points = _surface_load_markers(mesh, load, shape)
            print(
                f"  load markers: {load_points.shape[0]} anchor(s) "
                f"(index={nito_index}, snapped to surface)"
            )
        if want_voxels:
            overlay = voxel_overlay_mesh(
                shape,
                rho,
                threshold=args.density_cutoff,
                spacing=spacing[: int(shape.size)],
            )
            print(
                f"  voxel overlay: {int(overlay.n_cells):,} solid cells "
                f"(cutoff={args.density_cutoff})"
            )
        if want_bc_faces:
            patch_order = _resolve_patch_display_order(
                mesh,
                shape=shape,
                rho=rho,
                bc=bc,
                load=load,
                threshold=args.density_cutoff,
                n_bc=n_bc,
            )
            bc_faces = bc_face_overlays(
                rho,
                bc,
                shape,
                patch_order=patch_order,
                threshold=args.density_cutoff,
                spacing=spacing[: int(shape.size)],
            )
            print(f"  BC face overlays: {len(bc_faces)} patch(es)")

    if args.tri_skeleton:
        try:
            (
                tri_skeleton_mesh,
                tri_skeleton_scalar,
                tri_skeleton_labels,
            ) = _resolve_tri_skeleton_display(path, n_bc=n_bc)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        print(
            f"  tri skeleton overlay: {tri_skeleton_mesh.n_cells:,} triangles "
            f"colored by {PATCH_ID_KEY}"
        )

    plot_surface(
        mesh,
        scalar_name=scalar_name,
        title=title,
        cmap=cmap,
        show=show,
        save_path=args.save,
        scalar_location="cell" if categorical or scalar_name in mesh.cell_data else "point",
        legend_labels=legend_labels,
        categorical=categorical,
        voxel_overlay=overlay,
        voxel_opacity=args.voxel_opacity,
        bc_faces=bc_faces,
        load_points=load_points,
        marker_radius=marker_radius,
        bc_face_opacity=args.bc_face_opacity,
        tri_skeleton_mesh=tri_skeleton_mesh,
        tri_skeleton_scalar=tri_skeleton_scalar,
        tri_skeleton_labels=tri_skeleton_labels,
        tri_skeleton_opacity=args.tri_skeleton_opacity,
        tri_skeleton_line_width=args.tri_skeleton_line_width,
    )


if __name__ == "__main__":
    main()
