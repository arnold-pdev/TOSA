"""Shared types for the voxel2surf staged pipeline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import trimesh

from lib.converters.bc_patch import BCSpec, Patch
from lib.nito_physics import LoadCheckPolicy, LoadSurfaceCheck
from lib.volume_check import PipelineStageReport

LogFn = Callable[[str], None]


@dataclass
class PipelineOptions:
    """Keyword-controlled optional steps for systematic exploration."""

    extractor: str = "pyvista_binary"
    smoother: str = "laplacian_bc"
    iso_level: float = 0.5
    repair: bool = True
    meshfix_verbose: bool = False
    subdivide_levels: int = 0
    laplacian_iters: int = 8
    laplacian_relaxation: float = 0.5
    taubin_iters: int = 10
    taubin_lambda: float = 0.5
    taubin_nu: float = 0.53
    constrain_bc_planes: bool = True
    laplacian_freeze_rings: int = 0
    calibrate_vf: bool = False
    field_smooth_sigma: float = 0.0
    load_surface_tol_cells: float = 1.0
    vf_tol_cells: float = 2.0
    check_loads: bool = True
    on_load_fail: LoadCheckPolicy = "raise"
    bc_strict_footprint: bool = False


@dataclass
class SurfaceState:
    """Mutable pipeline state passed between stages."""

    mesh: trimesh.Trimesh
    vox: np.ndarray
    shape: np.ndarray
    origin: np.ndarray
    spacing: np.ndarray
    specs: list[BCSpec] = field(default_factory=list)
    patches: list[Patch] = field(default_factory=list)
    tri_labels: np.ndarray | None = None
    freeze: np.ndarray | None = None
    stage_reports: list[PipelineStageReport] = field(default_factory=list)
    bc_audit: dict[str, float] = field(default_factory=dict)
    load_checks: list[LoadSurfaceCheck] = field(default_factory=list)
    load_check_failed: bool = False
    extract_meta: dict[str, float] = field(default_factory=dict)

    @property
    def domain_shape(self) -> np.ndarray:
        return self.shape


@dataclass
class PipelineContext:
    """Immutable per-run context."""

    options: PipelineOptions
    log: LogFn
    recipe_id: str
    domain_vol: float
    vf_voxel: float
    bc_rows: np.ndarray | None = None
    load_rows: np.ndarray | None = None


@dataclass
class SurfaceResult:
    mesh: trimesh.Trimesh
    patches: list
    tri_labels: np.ndarray
    freeze: np.ndarray
    stage_reports: list[PipelineStageReport] = field(default_factory=list)
    load_checks: list[LoadSurfaceCheck] = field(default_factory=list)
    load_check_failed: bool = False
    bc_audit: dict[str, float] = field(default_factory=dict)
    validation: dict[str, float] = field(default_factory=dict)


class SurfaceOutputFormat(str, Enum):
    VTP = "vtp"
    VTK = "vtk"


@dataclass
class RunMetrics:
    index: int
    recipe_id: str
    vf_voxel: float
    vf_mesh: float | None
    vf_delta: float | None
    vf_nito: float | None
    bc_plane_max_residual: float
    bc_labeled_triangles: int
    load_check_failed: bool
    vertices: int
    faces: int
    patches: int

    def as_csv_row(self) -> dict[str, str | int | float]:
        return {
            "index": self.index,
            "recipe_id": self.recipe_id,
            "vf_voxel": self.vf_voxel,
            "vf_mesh": "" if self.vf_mesh is None else self.vf_mesh,
            "vf_delta": "" if self.vf_delta is None else self.vf_delta,
            "vf_nito": "" if self.vf_nito is None else self.vf_nito,
            "bc_plane_max_residual": self.bc_plane_max_residual,
            "bc_labeled_triangles": self.bc_labeled_triangles,
            "load_check_failed": int(self.load_check_failed),
            "vertices": self.vertices,
            "faces": self.faces,
            "patches": self.patches,
        }


def state_to_result(state: SurfaceState) -> SurfaceResult:
    tri_labels = state.tri_labels
    if tri_labels is None:
        tri_labels = np.full(len(state.mesh.faces), -1, dtype=np.int32)
    freeze = state.freeze
    if freeze is None:
        freeze = np.zeros(len(state.mesh.vertices), dtype=float)
    return SurfaceResult(
        mesh=state.mesh,
        patches=state.patches,
        tri_labels=np.asarray(tri_labels, dtype=np.int32),
        freeze=np.asarray(freeze, dtype=float),
        stage_reports=list(state.stage_reports),
        load_checks=list(state.load_checks),
        load_check_failed=state.load_check_failed,
        bc_audit=dict(state.bc_audit),
    )


def resolve_run_paths(
    run_dir: Path | None,
    index: int,
    *,
    output: Path | None,
    log_file: Path | None,
    log_dir: Path | None,
    default_output: Path,
) -> tuple[Path, Path | None]:
    """Resolve VTP output and per-index log path from --run-dir or legacy flags."""
    if run_dir is not None:
        root = run_dir.expanduser().resolve()
        out = output or (root / "vtp" / f"{index}.vtp")
        log_path = log_file or (root / "logs" / f"{index}.log")
        for sub in ("vtp", "logs", "metrics", "figures"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        return out, log_path
    out = output or default_output
    if log_file is not None:
        log_path = log_file.expanduser()
    elif log_dir is not None:
        log_path = log_dir.expanduser() / f"{index}.log"
    else:
        log_path = None
    return out, log_path
