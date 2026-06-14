"""VTP surface meshes for distribution and PyVista visualization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv

from lib.paths import REPO_ROOT
from lib.meshing.surface_tags import PATCH_ID_KEY, VTK_SUFFIXES

DEFAULT_SURFACE_DIR = REPO_ROOT / "public" / "vtp"
COORD_SCALAR_FIELDS = frozenset({"x", "y", "z"})


def vtp_path(surface_dir: Path, index: int) -> Path:
    return surface_dir / f"{index}.vtp"


def resolve_surface_path(
    *,
    index: int | None,
    surface_file: Path | None,
    surface_dir: Path,
) -> Path:
    if surface_file is not None:
        path = surface_file.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Surface file not found: {path}")
        if path.suffix.lower() not in {".vtp", ".vtk", ".stl", ".ply"}:
            raise ValueError(
                f"Unsupported surface format: {path.suffix}. "
                f"Use VTK PolyData ({', '.join(sorted(VTK_SUFFIXES))}) for the FEA path."
            )
        return path
    if index is None:
        raise ValueError("Provide --index or --surface-file")
    path = vtp_path(surface_dir.resolve(), index)
    if not path.is_file():
        raise FileNotFoundError(
            f"Surface not found: {path}\n"
            "Expected tagged VTK PolyData (.vtp) from voxel2surf or convert."
        )
    return path


def load_surface(path: Path) -> pv.PolyData:
    mesh = pv.read(path)
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    if mesh.n_cells == 0:
        raise ValueError(f"No surface cells in {path}")
    return mesh


def point_scalar_names(mesh: pv.PolyData) -> list[str]:
    return list(mesh.point_data.keys())


def cell_scalar_names(mesh: pv.PolyData) -> list[str]:
    return list(mesh.cell_data.keys())


def has_patch_tags(mesh: pv.PolyData) -> bool:
    return PATCH_ID_KEY in mesh.cell_data


def resolve_scalar_field(mesh: pv.PolyData, field: str) -> tuple[pv.PolyData, str]:
    """
    Return (mesh, array_name) for coloring.

    If `field` is x/y/z and not already on the mesh, attach from point coordinates.
    Otherwise use an existing point_data array.
    """
    if field in mesh.point_data:
        return mesh, field

    if field in COORD_SCALAR_FIELDS:
        axis = {"x": 0, "y": 1, "z": 2}[field]
        out = mesh.copy()
        out.point_data[field] = np.asarray(out.points, dtype=float)[:, axis]
        return out, field

    available = point_scalar_names(mesh)
    raise ValueError(
        f"Scalar field {field!r} not found on mesh. "
        f"Available point arrays: {available or '(none)'}"
    )


def save_vtp(mesh: pv.PolyData, path: Path) -> None:
    """Write a VTK PolyData (.vtp) for distribution."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.save(path)
