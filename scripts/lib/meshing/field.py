"""Voxel field upsample, SDF, VF calibration, and iso-surface extraction."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import trimesh
from scipy import ndimage
from skimage import measure

from lib.volume_check import format_mesh_stage_lines, volume_fraction_mesh

LogFn = Callable[[str], None]
MC_PAD_VOXELS = 1


def upsample_field(
    field: np.ndarray,
    factor: int,
    *,
    order: int = 1,
) -> np.ndarray:
    """
    Trilinear (order=1) upsample of a voxel/density grid by ``factor`` per axis.

    ``factor <= 1`` returns the input unchanged.
    """
    field = np.asarray(field, dtype=float)
    if factor <= 1:
        return field
    return ndimage.zoom(field, zoom=factor, order=order)


def signed_distance_field(vox: np.ndarray, spacing) -> np.ndarray:
    """Signed distance: negative inside solid, positive in void, ~0 on boundary."""
    spacing = np.asarray(spacing, float)
    solid = np.asarray(vox, dtype=float) > 0.5
    d_in = ndimage.distance_transform_edt(solid, sampling=spacing)
    d_out = ndimage.distance_transform_edt(~solid, sampling=spacing)
    return d_out - d_in


def field_volume_fraction(sdf: np.ndarray, iso_level: float) -> float:
    """Solid fraction proxy: fraction of grid cells with SDF below ``iso_level``."""
    sdf = np.asarray(sdf, dtype=float)
    return float(np.mean(sdf < iso_level))


def refine_sdf_field(
    sdf: np.ndarray,
    vf_target: float,
    *,
    smooth_sigma: float = 0.0,
    calibrate: bool = True,
    max_bisect_iters: int = 24,
    vf_tol: float = 1e-4,
) -> tuple[np.ndarray, float, dict[str, float]]:
    """
    Optional Gaussian smooth then bisect iso-level so field VF matches ``vf_target``.

    Returns ``(field, iso_level, meta)`` where ``meta`` has diagnostic scalars.
    """
    field = np.asarray(sdf, dtype=float)
    meta: dict[str, float] = {
        "vf_input": field_volume_fraction(field, 0.0),
        "smooth_sigma": float(smooth_sigma),
    }
    if smooth_sigma > 0:
        field = ndimage.gaussian_filter(field, sigma=smooth_sigma)

    iso_level = 0.0
    if calibrate:
        lo = float(field.min())
        hi = float(field.max())
        if lo >= hi:
            lo, hi = lo - 1.0, hi + 1.0
        iso_level = 0.5 * (lo + hi)
        for _ in range(max_bisect_iters):
            mid = 0.5 * (lo + hi)
            vf_mid = field_volume_fraction(field, mid)
            iso_level = mid
            if abs(vf_mid - vf_target) <= vf_tol:
                break
            # Higher iso_level → more cells with field < level → higher VF.
            if vf_mid < vf_target:
                lo = mid
            else:
                hi = mid
        iso_level = float(np.clip(iso_level, lo, hi))
        meta["vf_calibrated"] = field_volume_fraction(field, iso_level)
        meta["iso_level"] = iso_level
    else:
        meta["vf_calibrated"] = field_volume_fraction(field, iso_level)
        meta["iso_level"] = iso_level

    return field, iso_level, meta


def _pad_sdf_for_marching_cubes(
    sdf: np.ndarray,
    spacing: np.ndarray,
    *,
    pad: int = MC_PAD_VOXELS,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad SDF with exterior void so MC does not open at the grid boundary."""
    spacing = np.asarray(spacing, dtype=float)
    sdf = np.asarray(sdf, dtype=float)
    if pad <= 0:
        return sdf, np.zeros_like(spacing)
    void_val = float(np.max(sdf[sdf > 0])) if np.any(sdf > 0) else 1.0
    void_val = max(void_val, float(np.max(spacing)))
    padded = np.pad(sdf, pad, mode="constant", constant_values=void_val)
    return padded, -pad * spacing


def _pre_meshfix_cleanup(mesh: trimesh.Trimesh) -> int:
    """Remove degenerate/duplicate faces and merge vertices before MeshFix."""
    before = len(mesh.faces)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.merge_vertices()
    trimesh.repair.fix_normals(mesh)
    return before - len(mesh.faces)


def extract_surface_mesh(
    sdf: np.ndarray,
    origin,
    spacing,
    *,
    iso_level: float = 0.0,
    repair: bool = True,
    meshfix_verbose: bool = False,
    domain_vol: float | None = None,
    vf_target: float | None = None,
    log: LogFn | None = None,
) -> tuple[trimesh.Trimesh, int]:
    """
    Padded marching cubes + optional pymeshfix.

    Returns ``(mesh, n_faces_removed_pre_meshfix)``.
    """
    origin = np.asarray(origin, float)
    spacing = np.asarray(spacing, float)
    field = np.asarray(sdf, dtype=float)

    field, origin_shift = _pad_sdf_for_marching_cubes(field, spacing)
    mc_origin = origin + origin_shift
    level = float(iso_level)
    fmin, fmax = float(field.min()), float(field.max())
    if level <= fmin or level >= fmax:
        level = float(
            np.clip(
                level,
                fmin + 1e-6 * max(abs(fmin), 1.0),
                fmax - 1e-6 * max(abs(fmax), 1.0),
            ),
        )

    verts, faces, _normals, _vals = measure.marching_cubes(
        field,
        level=level,
        spacing=tuple(spacing),
        method="lewiner",
        allow_degenerate=False,
        gradient_direction="ascent",
    )
    verts = verts + mc_origin
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    removed = _pre_meshfix_cleanup(mesh)

    if log is not None and domain_vol is not None:
        for line in format_mesh_stage_lines(
            mesh,
            domain_vol,
            "marching_cubes",
            vf_reference=vf_target,
        ):
            log(f"        {line.strip()}")
        if removed:
            log(
                f"        pre_meshfix cleanup removed {removed:,} "
                "degenerate/duplicate faces",
            )

    if repair:
        try:
            import pymeshfix

            v_in = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
            f_in = np.ascontiguousarray(mesh.faces, dtype=np.int32)
            v, f = pymeshfix.clean_from_arrays(
                v_in,
                f_in,
                joincomp=True,
                verbose=meshfix_verbose,
            )
            mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
            trimesh.repair.fix_normals(mesh)
            if log is not None and domain_vol is not None:
                for line in format_mesh_stage_lines(
                    mesh,
                    domain_vol,
                    "pymeshfix",
                    vf_reference=vf_target,
                ):
                    log(f"        {line.strip()}")
        except ImportError:
            if log is not None:
                log("        pymeshfix not installed — skipping surface repair")

    return mesh, removed


def rescale_mesh_volume(
    mesh: trimesh.Trimesh,
    target_solid_volume: float,
) -> bool:
    """
    Uniformly scale mesh about its centroid to match ``target_solid_volume``.

    Returns True when rescale was applied.
    """
    if not mesh.is_watertight or target_solid_volume <= 0:
        return False
    current = float(mesh.volume)
    if current <= 0:
        return False
    scale = (target_solid_volume / current) ** (1.0 / 3.0)
    if abs(scale - 1.0) < 1e-9:
        return False
    centroid = mesh.centroid
    mesh.vertices = (mesh.vertices - centroid) * scale + centroid
    return True


def mesh_volume_fraction(mesh: trimesh.Trimesh, domain_vol: float) -> float | None:
    """Convenience wrapper around volume_check."""
    return volume_fraction_mesh(mesh, domain_vol)
