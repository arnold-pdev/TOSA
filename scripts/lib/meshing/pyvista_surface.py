"""PyVista ImageData contour extraction (no SDF pre-blur)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import trimesh

LogFn = Callable[[str], None]


def voxels_to_imagedata(
    vox: np.ndarray,
    origin,
    spacing,
    *,
    scalar_name: str = "rho",
) -> object:
    """Build a PyVista ImageData grid from a C-ordered voxel array (VTK F-flatten)."""
    import pyvista as pv

    vox = np.asarray(vox)
    shape = np.asarray(vox.shape, dtype=int)
    origin = np.asarray(origin, dtype=float)
    spacing = np.asarray(spacing, dtype=float)
    grid = pv.ImageData(dimensions=shape + 1)
    grid.origin = tuple(float(x) for x in origin)
    grid.spacing = tuple(float(x) for x in spacing)
    grid.cell_data[scalar_name] = vox.astype(float).ravel(order="F")
    return grid


def contour_to_trimesh(
    poly,
    *,
    process: bool = False,
) -> trimesh.Trimesh:
    """Convert a PyVista PolyData surface to Trimesh (triangles only)."""
    import pyvista as pv

    if not isinstance(poly, pv.PolyData):
        raise TypeError(f"Expected PolyData, got {type(poly).__name__}")
    work = poly.triangulate().clean()
    raw = np.asarray(work.faces, dtype=np.int64)
    if raw.size == 0:
        raise ValueError("Contour produced an empty surface")
    if raw.size % 4 != 0:
        raise ValueError(f"Expected triangle vtk faces (n,i,j,k); got {raw.size} ints")
    faces = raw.reshape(-1, 4)[:, 1:4]
    return trimesh.Trimesh(
        vertices=np.asarray(work.points, dtype=float),
        faces=np.asarray(faces, dtype=np.int64),
        process=process,
    )


def extract_contour_mesh(
    vox: np.ndarray,
    origin,
    spacing,
    *,
    iso_level: float = 0.5,
    repair: bool = True,
    meshfix_verbose: bool = False,
    log: LogFn | None = None,
) -> trimesh.Trimesh:
    """
    ImageData → solid iso-surface at ``iso_level`` → optional pymeshfix.

    Cell ``threshold`` + ``extract_surface`` on voxel data (no SDF pre-blur).
    Aligns with the NITO / marching-cubes frame; avoids point ``contour`` bias.
    """
    grid = voxels_to_imagedata(vox, origin, spacing)
    solid = grid.threshold(
        (float(iso_level) - 1e-9, 1.0 + 1e-9),
        scalars="rho",
        preference="cell",
    )
    poly = solid.extract_surface(algorithm="dataset_surface")
    mesh = contour_to_trimesh(poly)
    mesh = _pre_meshfix_cleanup(mesh)
    if log is not None:
        log(
            f"        contour iso={iso_level}: "
            f"{len(mesh.vertices):,} verts, {len(mesh.faces):,} faces",
        )
    if repair:
        mesh = _pymeshfix(mesh, verbose=meshfix_verbose, log=log)
    return mesh


def _pre_meshfix_cleanup(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.merge_vertices()
    trimesh.repair.fix_normals(mesh)
    return mesh


def _pymeshfix(
    mesh: trimesh.Trimesh,
    *,
    verbose: bool = False,
    log: LogFn | None = None,
) -> trimesh.Trimesh:
    try:
        import pymeshfix
    except ImportError:
        if log is not None:
            log("        pymeshfix not installed — skipping surface repair")
        return mesh
    v_in = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
    f_in = np.ascontiguousarray(mesh.faces, dtype=np.int32)
    v, f = pymeshfix.clean_from_arrays(
        v_in,
        f_in,
        joincomp=True,
        verbose=verbose,
    )
    out = trimesh.Trimesh(vertices=v, faces=f, process=False)
    trimesh.repair.fix_normals(out)
    if log is not None:
        log(
            f"        pymeshfix: {len(out.vertices):,} verts, "
            f"{len(out.faces):,} faces",
        )
    return out
