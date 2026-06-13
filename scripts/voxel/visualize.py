#!/usr/bin/env python3
"""
Visualize one NITO sample: topology, boundary conditions, and loads.

- 2D samples: Matplotlib (flat imshow + BC/load markers, no 3D camera).
- 3D samples: PyVista (interactive 3D).

Examples:

    conda activate tosa
    python scripts/voxel/visualize.py --index 0 --test
    python scripts/voxel/visualize.py --index 0 --test --with-gradient
    python scripts/voxel/visualize.py --index 0 --test --with-gradient --gradient-binary
    # NITO ML design (after voxel/predict.py):
    python scripts/voxel/visualize.py --index 0 --test \\
        --rho-file output/nito_predictions/0/rho_pred.npy --no-show \\
        --save output/figures/sample_0_nito_pred.png
    python scripts/voxel/visualize.py --index 0 --test --with-gradient \\
        --atoms-results output/atoms_results/nito_pred/0 --no-show \\
        --save output/figures/sample_0_nito_pred_grad.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.patches import Patch, Polygon

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from lib.paths import ensure_scripts_on_path

ensure_scripts_on_path()

from lib.nito_io import (
    REPO_ROOT,
    load_nito_arrays,
    load_sample,
    resolve_data_dir,
    shape_ndim,
)


def _default_title(index: int, shape: np.ndarray, vf: float) -> str:
    nd = shape_ndim(shape)
    return (
        f"NITO sample {index}  {nd}D  "
        f"shape={tuple(int(x) for x in shape)}  vf={vf:.3f}"
    )


def physical_points_xy(rows: np.ndarray, shape: np.ndarray) -> np.ndarray:
    """Normalized BC/load rows → physical XY."""
    rows = np.atleast_2d(np.asarray(rows, dtype=float))
    scale = float(np.max(shape))
    return rows[:, :2] * scale


def _bc_constraint_bits(flags_row: np.ndarray) -> tuple[int, int]:
    """Per-axis flags: 1 = fixed, 0 = free (ATOMS convention)."""
    f = np.asarray(flags_row, dtype=float).ravel()
    return (1 if f[0] > 0.5 else 0, 1 if f[1] > 0.5 else 0)


def _bc_marker_size(
    shape: np.ndarray,
    bc: np.ndarray,
    *,
    fraction_of_cell: float = 0.7,
    spacing_fraction: float = 0.5,
    min_fraction: float = 0.24,
    max_fraction: float = 1.0,
    scale_multiplier: float = 1.0,
) -> float:
    """
    Edge length for BC glyphs in plot data coordinates.

    imshow uses extent [0, nelx] × [0, nely], so one voxel is 1×1 — size is *not*
    proportional to nelx/nely. We start at a fraction of a cell, then shrink if BC
    points are densely spaced (typical edge discretizations).
    """
    cell = 1.0
    size = fraction_of_cell * cell

    bc = np.atleast_2d(np.asarray(bc, dtype=float))
    if bc.shape[0] > 1:
        pos = physical_points_xy(bc, shape)
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        sep = np.linalg.norm(diff, axis=2)
        upper = sep[np.triu_indices(len(pos), k=1)]
        if upper.size and np.any(upper > 1e-9):
            spacing = float(np.percentile(upper, 15))
            size = min(size, spacing_fraction * spacing)

    size = float(np.clip(size, min_fraction * cell, max_fraction * cell))
    return size * scale_multiplier


def _equilateral_triangle_xy(
    cx: float, cy: float, direction: tuple[float, float], edge_length: float
) -> np.ndarray:
    """Equilateral triangle centered at (cx, cy); vertex points along direction (+axis)."""
    d = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(d)
    if norm < 1e-12:
        raise ValueError("direction must be non-zero")
    d /= norm
    perp = np.array([-d[1], d[0]])
    s = edge_length
    altitude = s * np.sqrt(3) / 2
    center = np.array([cx, cy])
    tip = center + d * (2.0 / 3.0) * altitude
    base_c = center - d * (altitude / 3.0)
    left = base_c + perp * (s / 2.0)
    right = base_c - perp * (s / 2.0)
    return np.vstack([tip, left, right])


def _square_xy(cx: float, cy: float, edge_length: float) -> np.ndarray:
    h = edge_length / 2.0
    c = np.array([cx, cy])
    return np.array(
        [
            c + [-h, -h],
            c + [h, -h],
            c + [h, h],
            c + [-h, h],
        ]
    )


def draw_bc_markers_2d(
    ax: plt.Axes,
    bc: np.ndarray,
    shape: np.ndarray,
    *,
    edge_length: float | None = None,
    color: str = "crimson",
    edgecolor: str = "white",
    linewidth: float = 0.6,
    zorder: int = 3,
) -> list[Patch]:
    """
    Draw 2D BC glyphs from per-axis constraint flags (cols 2:4 for 2D rows).

    - (0, 0): free in both — nothing drawn
    - (1, 1): fixed x & y — square
    - (0, 1) or (1, 0): roller — triangle pointing along the free axis (+x or +y)
    """
    bc = np.atleast_2d(np.asarray(bc, dtype=float))
    if bc.shape[1] < 4:
        return []

    pos = physical_points_xy(bc, shape)
    size = edge_length if edge_length is not None else _bc_marker_size(shape, bc)
    kinds: set[str] = set()
    legend_patches: list[Patch] = []

    for k in range(pos.shape[0]):
        fx, fy = _bc_constraint_bits(bc[k, 2:4])
        cx, cy = float(pos[k, 0]), float(pos[k, 1])

        if fx == 0 and fy == 0:
            continue
        if fx == 1 and fy == 1:
            verts = _square_xy(cx, cy, size)
            kinds.add("clamp")
        elif fx == 0 and fy == 1:
            verts = _equilateral_triangle_xy(cx, cy, (1.0, 0.0), size)
            kinds.add("roller")
        elif fx == 1 and fy == 0:
            verts = _equilateral_triangle_xy(cx, cy, (0.0, 1.0), size)
            kinds.add("roller")
        else:
            continue

        ax.add_patch(
            Polygon(
                verts,
                closed=True,
                facecolor=color,
                edgecolor=edgecolor,
                linewidth=linewidth,
                zorder=zorder,
            )
        )

    if "clamp" in kinds:
        legend_patches.append(
            Patch(
                facecolor=color,
                edgecolor=edgecolor,
                linewidth=linewidth,
                label="BC: fixed x & y",
            )
        )
    if "roller" in kinds:
        legend_patches.append(
            Patch(
                facecolor=color,
                edgecolor=edgecolor,
                linewidth=linewidth,
                label="BC: roller (△ → free axis)",
            )
        )
    return legend_patches


def physical_points_xyz(rows: np.ndarray, shape: np.ndarray) -> np.ndarray:
    """Normalized BC/load rows → physical XYZ for PyVista."""
    dim = int(np.asarray(shape, dtype=int).size)
    rows = np.atleast_2d(np.asarray(rows, dtype=float))
    scale = float(np.max(shape))
    pts = rows[:, :dim] * scale
    if dim == 2:
        pts = np.column_stack([pts, np.zeros(len(pts), dtype=float)])
    return np.asarray(pts, dtype=float)


def load_atoms_results(results_dir: Path) -> dict[str, np.ndarray | float]:
    """Load outputs written by scripts/voxel/sensitivity/compliance.py."""
    results_dir = Path(results_dir)
    required = ("dC_drho.npy", "shape.npy")
    missing = [n for n in required if not (results_dir / n).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing in {results_dir}: {', '.join(missing)}. "
            "Run: python scripts/voxel/sensitivity/compliance.py --index <i> --test"
        )
    out: dict[str, np.ndarray | float] = {
        "dC_drho": np.load(results_dir / "dC_drho.npy"),
        "shape": np.load(results_dir / "shape.npy"),
    }
    if (results_dir / "rho.npy").exists():
        out["rho"] = np.load(results_dir / "rho.npy")
    if (results_dir / "compliance.npy").exists():
        out["compliance"] = float(np.load(results_dir / "compliance.npy"))
    return out


def _reshape_field_2d(field: np.ndarray, shape: np.ndarray) -> np.ndarray:
    return np.asarray(field, dtype=float).reshape(
        (int(shape[0]), int(shape[1])), order="C"
    )


def mask_binary_gradient(
    rho: np.ndarray,
    dC_drho: np.ndarray,
    shape: np.ndarray,
    *,
    rho_lo: float = 0.01,
    rho_hi: float = 0.99,
) -> np.ndarray:
    """
    Keep ∂C/∂ρ only where it matches a feasible binary flip.

    Void (ρ ≤ rho_lo): negative → filling would lower compliance.
    Solid (ρ ≥ rho_hi): positive → removal would lower compliance.
    Other cells are NaN (masked in the plot).
    """
    R = _reshape_field_2d(rho, shape)
    G = _reshape_field_2d(dC_drho, shape)
    void = R <= rho_lo
    solid = R >= rho_hi
    show = (void & (G < 0)) | (solid & (G > 0))
    out = np.where(show, G, np.nan)
    return out.ravel(order="C")


def _imshow_2d_field(
    ax: plt.Axes,
    field: np.ndarray,
    shape: np.ndarray,
    *,
    cmap: str,
    title: str,
    cbar_label: str,
    vmin: float | None = None,
    vmax: float | None = None,
    bad_color: str | None = None,
) -> None:
    nx, ny = int(shape[0]), int(shape[1])
    data = _reshape_field_2d(field, shape).T
    cmap_obj = plt.get_cmap(cmap).copy()
    if bad_color is not None:
        cmap_obj.set_bad(color=bad_color)
    im = ax.imshow(
        np.ma.masked_invalid(data),
        origin="lower",
        extent=[0, nx, 0, ny],
        aspect="equal",
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label=cbar_label, fraction=0.046)


def _overlay_bcs_loads_2d(
    ax: plt.Axes,
    bc: np.ndarray,
    load: np.ndarray,
    shape: np.ndarray,
    *,
    bc_marker_size: float | None,
    bc_marker_scale: float,
) -> list:
    edge_len = bc_marker_size
    if edge_len is None:
        edge_len = _bc_marker_size(shape, bc, scale_multiplier=bc_marker_scale)
    legend_handles = draw_bc_markers_2d(ax, bc, shape, edge_length=edge_len)

    load_arr = np.atleast_2d(np.asarray(load, dtype=float))
    load_pts = physical_points_xy(load_arr, shape)
    if load_pts.size:
        ax.scatter(
            load_pts[:, 0],
            load_pts[:, 1],
            c="dodgerblue",
            s=60,
            marker="*",
            edgecolors="white",
            linewidths=0.5,
            zorder=4,
        )
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                linestyle="none",
                marker="*",
                color="dodgerblue",
                markeredgecolor="white",
                markeredgewidth=0.5,
                markersize=10,
                label="Load point",
            )
        )
    return legend_handles


def plot_sample_2d(
    index: int,
    shape: np.ndarray,
    rho: np.ndarray,
    bc: np.ndarray,
    load: np.ndarray,
    vf: float,
    *,
    dC_drho: np.ndarray | None = None,
    compliance: float | None = None,
    gradient_binary: bool = False,
    gradient_rho_lo: float = 0.01,
    gradient_rho_hi: float = 0.99,
    threshold: float = 0.5,
    title: str | None = None,
    show: bool = True,
    save_path: Path | None = None,
    bc_marker_size: float | None = None,
    bc_marker_scale: float = 1.0,
) -> None:
    shape = np.asarray(shape, dtype=int).ravel()
    if title is None:
        title = _default_title(index, shape, vf)
    if compliance is not None:
        title = f"{title}  C={compliance:.4e}"

    if dC_drho is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        g = np.asarray(dC_drho, dtype=float).reshape(-1)
        if gradient_binary:
            g_plot = mask_binary_gradient(
                rho,
                dC_drho,
                shape,
                rho_lo=gradient_rho_lo,
                rho_hi=gradient_rho_hi,
            )
            shown = g_plot[np.isfinite(g_plot)]
            lim = float(np.max(np.abs(shown))) if shown.size else 1.0
            grad_title = r"$\partial C / \partial \rho$ (binary flips)"
            cbar_label = r"$\partial C / \partial \rho$"
            bad_color = "#d9d9d9"
        else:
            g_plot = g
            lim = float(np.max(np.abs(g))) or 1.0
            grad_title = r"$\partial C / \partial \rho$"
            cbar_label = r"$\partial C / \partial \rho$"
            bad_color = None
        _imshow_2d_field(
            axes[1],
            g_plot,
            shape,
            cmap="RdBu_r",
            title=grad_title,
            cbar_label=cbar_label,
            vmin=-lim,
            vmax=lim,
            bad_color=bad_color,
        )
        _overlay_bcs_loads_2d(
            axes[1], bc, load, shape,
            bc_marker_size=bc_marker_size,
            bc_marker_scale=bc_marker_scale,
        )
        if gradient_binary:
            axes[1].legend(
                handles=[
                    Patch(facecolor=plt.cm.RdBu_r(0.05), edgecolor="k", label="Void: fill helps"),
                    Patch(facecolor=plt.cm.RdBu_r(0.95), edgecolor="k", label="Solid: remove helps"),
                ],
                loc="upper right",
                fontsize=8,
            )

    ax0 = axes[0]
    _imshow_2d_field(
        ax0,
        rho,
        shape,
        cmap="Greys",
        title=r"$\rho$" if dC_drho is not None else title,
        cbar_label=r"$\rho$",
        vmin=0.0,
        vmax=1.0,
    )
    legend_handles = _overlay_bcs_loads_2d(
        ax0, bc, load, shape,
        bc_marker_size=bc_marker_size,
        bc_marker_scale=bc_marker_scale,
    )
    if dC_drho is None and legend_handles:
        ax0.legend(handles=legend_handles, loc="upper right")
    elif dC_drho is not None:
        fig.suptitle(title, fontsize=11)
        axes[0].set_title(r"$\rho$")

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def topology_mesh_3d(
    shape: np.ndarray, rho: np.ndarray, *, threshold: float = 0.5
) -> pv.UnstructuredGrid:
    shape = np.asarray(shape, dtype=int).ravel()
    design = rho.reshape(shape, order="C")
    grid = pv.ImageData(dimensions=np.array(shape) + 1)
    grid.cell_data["rho"] = design.flatten(order="F")
    return grid.threshold(threshold, scalars="rho")


def plot_sample_3d(
    index: int,
    shape: np.ndarray,
    rho: np.ndarray,
    bc: np.ndarray,
    load: np.ndarray,
    vf: float,
    *,
    threshold: float = 0.5,
    title: str | None = None,
    show: bool = True,
    save_path: Path | None = None,
) -> None:
    off_screen = save_path is not None and not show
    solid = topology_mesh_3d(shape, rho, threshold=threshold)
    bc_pts = physical_points_xyz(bc, shape)
    load_pts = physical_points_xyz(load, shape)

    if title is None:
        title = _default_title(index, shape, vf)

    plotter = pv.Plotter(off_screen=off_screen)
    plotter.add_text(title, font_size=10)
    plotter.enable_depth_peeling()

    plotter.add_mesh(
        solid,
        show_edges=True,
        color="tan",
        opacity=0.45,
        label=f"Topology (rho >= {threshold})",
    )
    if bc_pts.size:
        plotter.add_mesh(
            pv.PolyData(bc_pts),
            color="red",
            point_size=18,
            render_points_as_spheres=True,
            label="BCs",
        )
    if load_pts.size:
        plotter.add_mesh(
            pv.PolyData(load_pts),
            color="dodgerblue",
            point_size=18,
            render_points_as_spheres=True,
            label="Loads",
        )

    plotter.add_legend()
    plotter.add_axes_at_origin()
    plotter.show_grid()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if show:
            plotter.show(screenshot=str(save_path))
        else:
            plotter.screenshot(str(save_path))
            plotter.close()
        print(f"Saved {save_path}")
    elif show:
        plotter.show()
    else:
        plotter.close()


def plot_sample(
    index: int,
    shape: np.ndarray,
    rho: np.ndarray,
    bc: np.ndarray,
    load: np.ndarray,
    vf: float,
    *,
    threshold: float = 0.5,
    title: str | None = None,
    show: bool = True,
    save_path: Path | None = None,
    backend: str | None = None,
    bc_marker_size: float | None = None,
    bc_marker_scale: float = 1.0,
    dC_drho: np.ndarray | None = None,
    compliance: float | None = None,
    gradient_binary: bool = False,
    gradient_rho_lo: float = 0.01,
    gradient_rho_hi: float = 0.99,
) -> None:
    nd = shape_ndim(shape)
    if backend is None:
        backend = "matplotlib" if nd == 2 else "pyvista"
    if backend == "matplotlib":
        if nd != 2:
            raise ValueError("Matplotlib backend only supports 2D samples")
        plot_sample_2d(
            index, shape, rho, bc, load, vf,
            dC_drho=dC_drho,
            compliance=compliance,
            gradient_binary=gradient_binary,
            gradient_rho_lo=gradient_rho_lo,
            gradient_rho_hi=gradient_rho_hi,
            threshold=threshold,
            title=title,
            show=show,
            save_path=save_path,
            bc_marker_size=bc_marker_size,
            bc_marker_scale=bc_marker_scale,
        )
    elif backend == "pyvista":
        if dC_drho is not None:
            raise ValueError(
                "Gradient visualization is 2D-only; use default backend on a 2D sample."
            )
        plot_sample_3d(
            index, shape, rho, bc, load, vf,
            threshold=threshold, title=title, show=show, save_path=save_path,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize a NITO dataset sample.")
    p.add_argument("--index", type=int, default=0, help="Sample index")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="NITO data root (default: nito/Data)",
    )
    p.add_argument("--test", action="store_true", help="Use <data-dir>/Test/")
    p.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Save figure to this path (e.g. output/figures/sample_0.png)",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive window (use with --save)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Contour / threshold for solid region",
    )
    p.add_argument(
        "--backend",
        choices=("matplotlib", "pyvista"),
        default=None,
        help="Force viewer (default: matplotlib for 2D, pyvista for 3D)",
    )
    dim = p.add_mutually_exclusive_group()
    dim.add_argument("--only-2d", action="store_true", help="Error if sample is not 2D")
    dim.add_argument("--only-3d", action="store_true", help="Error if sample is not 3D")
    p.add_argument(
        "--bc-marker-size",
        type=float,
        default=None,
        help="BC glyph edge length in plot units (1 unit = one cell); overrides auto sizing",
    )
    p.add_argument(
        "--bc-marker-scale",
        type=float,
        default=1.0,
        help="Multiplier on auto BC glyph size (default 1.0)",
    )
    p.add_argument(
        "--with-gradient",
        action="store_true",
        help="Overlay dC/drho from sensitivity run (output/atoms_results/<index>/)",
    )
    p.add_argument(
        "--gradient-binary",
        action="store_true",
        help="Mask gradient: void & dC/drho<0 (fill), solid & dC/drho>0 (remove); implies --with-gradient",
    )
    p.add_argument(
        "--gradient-rho-lo",
        type=float,
        default=0.01,
        help="Treat rho <= this as void for --gradient-binary (default 0.01)",
    )
    p.add_argument(
        "--gradient-rho-hi",
        type=float,
        default=0.99,
        help="Treat rho >= this as solid for --gradient-binary (default 0.99)",
    )
    p.add_argument(
        "--atoms-results",
        type=Path,
        default=None,
        help="Directory with dC_drho.npy (default: output/atoms_results/<index>)",
    )
    p.add_argument(
        "--rho-file",
        type=Path,
        default=None,
        help="Design field for left panel (e.g. output/nito_predictions/<i>/rho_pred.npy); "
        "default: dataset topology or rho.npy from --atoms-results",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = resolve_data_dir(args.data_dir, test=args.test)
    data = load_nito_arrays(data_dir)
    n = len(data["shapes"])
    if not (0 <= args.index < n):
        raise IndexError(f"index {args.index} out of range [0, {n})")

    shape, rho, bc, load, vf = load_sample(data, args.index)
    if args.rho_file is not None:
        rho_path = Path(args.rho_file)
        if not rho_path.is_file():
            raise FileNotFoundError(f"--rho-file not found: {rho_path}")
        rho = np.load(rho_path, allow_pickle=True).astype(float).reshape(-1, 1)
        if rho.size != int(np.prod(shape)):
            raise ValueError(
                f"--rho-file length {rho.size} != shape.prod() {int(np.prod(shape))}"
            )

    nd = shape_ndim(shape)
    if args.only_2d and nd != 2:
        raise ValueError(f"index {args.index} is {nd}D, not 2D (shape={shape})")
    if args.only_3d and nd != 3:
        raise ValueError(f"index {args.index} is {nd}D, not 3D (shape={shape})")

    if args.gradient_binary and not args.with_gradient:
        args.with_gradient = True

    dC_drho = None
    compliance = None
    if args.with_gradient:
        results_dir = args.atoms_results
        if results_dir is None:
            results_dir = REPO_ROOT / "output" / "atoms_results" / str(args.index)
        sens = load_atoms_results(results_dir)
        dC_drho = np.asarray(sens["dC_drho"])
        compliance = sens.get("compliance")
        if args.rho_file is None and "rho" in sens:
            rho = np.asarray(sens["rho"]).reshape(-1, 1)
        shape = np.asarray(sens["shape"], dtype=int)

    plot_sample(
        args.index,
        shape,
        rho,
        bc,
        load,
        vf,
        threshold=args.threshold,
        show=not args.no_show,
        save_path=args.save,
        backend=args.backend,
        bc_marker_size=args.bc_marker_size,
        bc_marker_scale=args.bc_marker_scale,
        dC_drho=dC_drho,
        compliance=compliance,
        gradient_binary=args.gradient_binary,
        gradient_rho_lo=args.gradient_rho_lo,
        gradient_rho_hi=args.gradient_rho_hi,
    )


if __name__ == "__main__":
    main()
