"""Staged voxel → tagged surface pipeline orchestrator."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import trimesh

from lib.converters.bc_patch import BCSpec
from lib.converters.voxel2surf.recipes import Recipe, get_recipe
from lib.converters.voxel2surf.stages import run_stage
from lib.converters.voxel2surf.types import (
    PipelineContext,
    PipelineOptions,
    SurfaceResult,
    SurfaceState,
    state_to_result,
)
from lib.volume_check import domain_volume, volume_fraction_voxels

LogFn = Callable[[str], None]


def _noop_log(_msg: str) -> None:
    pass


def _apply_recipe_defaults(opts: PipelineOptions, recipe: Recipe) -> PipelineOptions:
    opts.extractor = recipe.extractor
    opts.smoother = recipe.smoother
    return opts


def run_pipeline(
    state: SurfaceState,
    ctx: PipelineContext,
    recipe: Recipe,
) -> SurfaceState:
    for stage in recipe.stages:
        run_stage(stage, state, ctx)
    return state


def voxel_to_surface(
    vox: np.ndarray,
    bcspecs: list[BCSpec] | None = None,
    *,
    bc_rows: np.ndarray | None = None,
    load_rows: np.ndarray | None = None,
    grid_shape: np.ndarray | None = None,
    origin=(0.0, 0.0, 0.0),
    spacing=(1.0, 1.0, 1.0),
    options: PipelineOptions | None = None,
    recipe: str | Recipe = "v3_default",
    log: LogFn | None = None,
    **kwargs,
) -> SurfaceResult:
    """
    Build a tagged surface via a named recipe and staged pipeline.

    Override options with ``options=PipelineOptions(...)`` or keyword args.
    """
    opts = options or PipelineOptions()
    for key, value in kwargs.items():
        if not hasattr(opts, key):
            raise TypeError(f"Unknown pipeline option: {key}")
        setattr(opts, key, value)

    if isinstance(recipe, str):
        recipe_obj = get_recipe(recipe)
    else:
        recipe_obj = recipe
    opts = _apply_recipe_defaults(opts, recipe_obj)

    spacing = np.asarray(spacing, float)
    origin = np.asarray(origin, float)
    shape = np.asarray(grid_shape if grid_shape is not None else vox.shape, dtype=int)
    log_fn = _noop_log if log is None else log

    ctx = PipelineContext(
        options=opts,
        log=log_fn,
        recipe_id=recipe_obj.id,
        domain_vol=domain_volume(shape, spacing),
        vf_voxel=volume_fraction_voxels(vox),
        bc_rows=bc_rows,
        load_rows=load_rows,
    )

    log_fn(f"voxel2surf  recipe={recipe_obj.id}")
    log_fn(
        f"  extract={opts.extractor} smooth={opts.smoother} "
        f"iso={opts.iso_level}",
    )

    # Bootstrap mesh placeholder; extract stage replaces it.
    placeholder = trimesh.Trimesh(
        vertices=np.zeros((0, 3), dtype=float),
        faces=np.zeros((0, 3), dtype=np.int64),
        process=False,
    )
    state = SurfaceState(
        mesh=placeholder,
        vox=np.asarray(vox),
        shape=shape,
        origin=origin,
        spacing=spacing,
        specs=list(bcspecs or []),
    )

    run_pipeline(state, ctx, recipe_obj)
    return state_to_result(state)


def save_surface(result: SurfaceResult, path, *, fmt: str | None = None):
    from lib.meshing.surface_tags import write_surface_vtp
    from pathlib import Path

    path = Path(path)
    fmt_str = None if fmt is None else str(fmt).lower().lstrip(".")
    return write_surface_vtp(result, path, fmt=fmt_str)
