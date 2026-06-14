"""
Volume-fraction checks for NITO voxels and surface meshes (STL / VTP).

Domain volume is the design-box volume from shape × spacing (NITO convention:
unit spacing unless overridden). Surface volume fraction uses the enclosed volume
of a watertight triangle mesh divided by that domain volume.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh


def domain_volume(shape: np.ndarray, spacing: np.ndarray) -> float:
    """Physical volume of the design-domain voxel box."""
    shape = np.asarray(shape, dtype=float)
    spacing = np.asarray(spacing, dtype=float)
    return float(np.prod(shape) * np.prod(spacing))


def volume_fraction_voxels(vox: np.ndarray) -> float:
    """Solid fraction from the binarized voxel grid."""
    vox = np.asarray(vox)
    return float(np.count_nonzero(vox)) / float(vox.size)


def volume_fraction_mesh(mesh: trimesh.Trimesh, domain_vol: float) -> float | None:
    """Solid fraction from the volume enclosed by a watertight surface mesh."""
    if domain_vol <= 0 or not mesh.is_watertight:
        return None
    return float(mesh.volume) / domain_vol


def solid_volume_mesh(mesh: trimesh.Trimesh) -> float | None:
    """Enclosed volume of a watertight mesh, or None."""
    if not mesh.is_watertight:
        return None
    return float(mesh.volume)


def default_spacing(shape: np.ndarray) -> np.ndarray:
    """Unit spacing along each design-domain axis (NITO voxel2surf default)."""
    return np.ones(int(np.asarray(shape, dtype=int).size), dtype=float)


def binarize_topology(
    rho: np.ndarray,
    shape: np.ndarray,
    *,
    cutoff: float = 0.5,
) -> np.ndarray:
    """Binarize a NITO topology row into a uint8 voxel grid."""
    vox = np.asarray(rho, dtype=float).reshape(
        tuple(int(x) for x in shape),
        order="C",
    )
    return (vox >= cutoff).astype(np.uint8)


def trimesh_from_path(path: Path) -> trimesh.Trimesh:
    """Load STL, VTP, VTK, PLY, etc. as a single Trimesh."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Surface file not found: {path}")

    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError(f"No triangle mesh in {path}")
        loaded = trimesh.util.concatenate(geoms) if len(geoms) > 1 else geoms[0]
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"Expected a triangle mesh in {path}, got {type(loaded).__name__}")
    return loaded


@dataclass(frozen=True)
class VolumeReport:
    """Volume-fraction summary for one geometry source."""

    source: str  # "voxel" | "stl" | "vtp" | "surface"
    path: Path | None
    index: int | None
    shape: tuple[int, ...]
    spacing: tuple[float, ...]
    domain_volume: float
    volume_fraction: float | None
    solid_volume: float | None
    vf_nito: float | None
    watertight: bool | None
    n_vertices: int | None
    n_faces: int | None

    def format_lines(self) -> list[str]:
        """Human-readable report lines."""
        lines = [f"source={self.source}"]
        if self.index is not None:
            lines.append(f"  index={self.index}")
        if self.path is not None:
            lines.append(f"  path={self.path}")
        lines.append(f"  shape={self.shape}  spacing={self.spacing}")
        lines.append(f"  domain_volume={self.domain_volume:.6g}")
        if self.volume_fraction is not None:
            lines.append(f"  volume_fraction={self.volume_fraction:.6f}")
        else:
            reason = "mesh not watertight" if self.watertight is False else "n/a"
            lines.append(f"  volume_fraction=n/a ({reason})")
        if self.solid_volume is not None:
            lines.append(f"  solid_volume={self.solid_volume:.6g}")
        if self.watertight is not None:
            lines.append(f"  watertight={self.watertight}")
        if self.n_vertices is not None:
            lines.append(
                f"  mesh={self.n_vertices:,} vertices, {self.n_faces:,} triangles"
            )
        if self.vf_nito is not None:
            lines.append(f"  nito_target={self.vf_nito:.6f}")
            if self.volume_fraction is not None:
                lines.append(
                    f"  delta_vs_nito={self.volume_fraction - self.vf_nito:+.6f}"
                )
        return lines


def check_voxels(
    vox: np.ndarray,
    shape: np.ndarray,
    spacing: np.ndarray | None = None,
    *,
    index: int | None = None,
    vf_nito: float | None = None,
) -> VolumeReport:
    """Volume fraction from a binarized voxel grid."""
    shape_arr = tuple(int(x) for x in np.asarray(shape, dtype=int).ravel())
    spacing_arr = np.asarray(spacing if spacing is not None else default_spacing(shape), dtype=float)
    dom = domain_volume(shape_arr, spacing_arr)
    vf = volume_fraction_voxels(vox)
    return VolumeReport(
        source="voxel",
        path=None,
        index=index,
        shape=shape_arr,
        spacing=tuple(float(x) for x in spacing_arr),
        domain_volume=dom,
        volume_fraction=vf,
        solid_volume=vf * dom,
        vf_nito=vf_nito,
        watertight=None,
        n_vertices=None,
        n_faces=None,
    )


def check_surface_mesh(
    mesh: trimesh.Trimesh,
    shape: np.ndarray,
    spacing: np.ndarray | None = None,
    *,
    source: str = "surface",
    path: Path | None = None,
    index: int | None = None,
    vf_nito: float | None = None,
) -> VolumeReport:
    """Volume fraction from a surface mesh against the design-domain box."""
    shape_arr = tuple(int(x) for x in np.asarray(shape, dtype=int).ravel())
    spacing_arr = np.asarray(spacing if spacing is not None else default_spacing(shape), dtype=float)
    dom = domain_volume(shape_arr, spacing_arr)
    watertight = bool(mesh.is_watertight)
    solid = solid_volume_mesh(mesh)
    vf = volume_fraction_mesh(mesh, dom)
    return VolumeReport(
        source=source,
        path=path,
        index=index,
        shape=shape_arr,
        spacing=tuple(float(x) for x in spacing_arr),
        domain_volume=dom,
        volume_fraction=vf,
        solid_volume=solid,
        vf_nito=vf_nito,
        watertight=watertight,
        n_vertices=len(mesh.vertices),
        n_faces=len(mesh.faces),
    )


def check_surface_file(
    path: Path,
    shape: np.ndarray,
    spacing: np.ndarray | None = None,
    *,
    index: int | None = None,
    vf_nito: float | None = None,
) -> VolumeReport:
    """Volume fraction from an on-disk surface mesh (STL, VTP, …)."""
    path = path.expanduser().resolve()
    suffix = path.suffix.lower().lstrip(".")
    source = suffix if suffix in {"stl", "vtp", "vtk", "ply"} else "surface"
    mesh = trimesh_from_path(path)
    return check_surface_mesh(
        mesh,
        shape,
        spacing,
        source=source,
        path=path,
        index=index,
        vf_nito=vf_nito,
    )


def mesh_body_count(mesh: trimesh.Trimesh) -> int:
    """Connected surface components (including non-watertight pieces)."""
    return len(mesh.split(only_watertight=False))


def mesh_degenerate_face_count(mesh: trimesh.Trimesh) -> int:
    """Triangles with near-zero area."""
    if len(mesh.faces) == 0:
        return 0
    return int(len(mesh.faces) - len(mesh.nondegenerate_faces()))


@dataclass(frozen=True)
class PipelineStageReport:
    """Compact stats for one voxel2surf pipeline stage."""

    name: str
    volume_fraction: float | None = None
    vf_target: float | None = None
    watertight: bool | None = None
    bodies: int | None = None
    degenerate_faces: int | None = None
    n_vertices: int | None = None
    n_faces: int | None = None
    details: tuple[str, ...] = ()

    def format_line(self) -> str:
        """Single log line for this stage."""
        parts = [f"  [{self.name}]"]
        if self.volume_fraction is not None:
            parts.append(f"vf={self.volume_fraction:.6f}")
            if self.vf_target is not None:
                parts.append(f"delta_vs_target={self.volume_fraction - self.vf_target:+.6f}")
        elif self.vf_target is not None:
            parts.append(f"vf_target={self.vf_target:.6f}")
        if self.watertight is not None:
            parts.append(f"watertight={self.watertight}")
        if self.bodies is not None:
            parts.append(f"bodies={self.bodies}")
        if self.degenerate_faces is not None:
            parts.append(f"degenerate={self.degenerate_faces}")
        if self.n_vertices is not None:
            parts.append(f"verts={self.n_vertices:,}")
        if self.n_faces is not None:
            parts.append(f"faces={self.n_faces:,}")
        if self.details:
            parts.append(" ".join(self.details))
        return "  ".join(parts)


def mesh_stage_report(
    mesh: trimesh.Trimesh,
    domain_vol: float,
    name: str,
    *,
    vf_target: float | None = None,
) -> PipelineStageReport:
    """Build a pipeline report for a surface mesh stage."""
    vf = volume_fraction_mesh(mesh, domain_vol)
    return PipelineStageReport(
        name=name,
        volume_fraction=vf,
        vf_target=vf_target,
        watertight=bool(mesh.is_watertight),
        bodies=mesh_body_count(mesh),
        degenerate_faces=mesh_degenerate_face_count(mesh),
        n_vertices=len(mesh.vertices),
        n_faces=len(mesh.faces),
    )


def field_stage_report(
    name: str,
    *,
    volume_fraction: float | None = None,
    vf_target: float | None = None,
    details: Sequence[str] = (),
) -> PipelineStageReport:
    """Build a pipeline report for a field-only stage (no mesh yet)."""
    return PipelineStageReport(
        name=name,
        volume_fraction=volume_fraction,
        vf_target=vf_target,
        details=tuple(details),
    )


def format_mesh_stage_lines(
    mesh: trimesh.Trimesh,
    domain_vol: float,
    stage: str,
    *,
    vf_reference: float | None = None,
) -> list[str]:
    """Volume-fraction and topology stats for an intermediate surface mesh."""
    vf = volume_fraction_mesh(mesh, domain_vol)
    watertight = bool(mesh.is_watertight)
    bodies = mesh_body_count(mesh)
    degenerate = mesh_degenerate_face_count(mesh)
    stats = (
        f"watertight={watertight}  bodies={bodies}  degenerate={degenerate}  "
        f"verts={len(mesh.vertices):,}  faces={len(mesh.faces):,}"
    )
    if vf is not None:
        line = f"  volume_fraction  {stage}={vf:.6f}  {stats}"
        if vf_reference is not None:
            line += f"  delta_vs_voxels={vf - vf_reference:+.6f}"
        return [line]
    return [f"  volume_fraction  {stage}=n/a  {stats}"]


def format_volume_fraction_lines(
    *,
    vox: np.ndarray,
    mesh: trimesh.Trimesh,
    shape: np.ndarray,
    spacing: np.ndarray,
    vf_nito: float,
) -> list[str]:
    """Compact before/after lines for voxel2surf CLI output."""
    dom = domain_volume(shape, spacing)
    vf_before = volume_fraction_voxels(vox)
    vf_after = volume_fraction_mesh(mesh, dom)
    watertight = bool(mesh.is_watertight)
    lines = [f"  volume_fraction  before (voxels)={vf_before:.6f}"]
    if vf_after is not None:
        lines.append(
            f"  volume_fraction  after (surface)={vf_after:.6f}  "
            f"delta={vf_after - vf_before:+.6f}"
        )
    else:
        bodies = mesh_body_count(mesh)
        lines.append(
            f"  volume_fraction  after (surface)=n/a  "
            f"watertight={watertight}  body_count={bodies}"
        )
    lines.append(f"  volume_fraction  nito_target={float(vf_nito):.6f}")
    return lines


def format_compare_lines(
    reports: list[VolumeReport],
    *,
    baseline: str = "voxel",
) -> list[str]:
    """Compare multiple VolumeReport rows (e.g. voxel vs vtp)."""
    lines: list[str] = []
    by_source = {r.source: r for r in reports}
    base = by_source.get(baseline)
    if base is not None and base.volume_fraction is not None:
        lines.append(f"baseline ({baseline}) volume_fraction={base.volume_fraction:.6f}")
    for report in reports:
        if report.source == baseline:
            continue
        if report.volume_fraction is None:
            lines.append(f"  vs {report.source}: n/a")
            continue
        if base is not None and base.volume_fraction is not None:
            delta = report.volume_fraction - base.volume_fraction
            lines.append(
                f"  vs {report.source}: {report.volume_fraction:.6f}  delta={delta:+.6f}"
            )
        else:
            lines.append(f"  {report.source}: {report.volume_fraction:.6f}")
    nito = next((r.vf_nito for r in reports if r.vf_nito is not None), None)
    if nito is not None:
        lines.append(f"  nito_target={nito:.6f}")
    return lines
