"""Voxel ↔ surface conversion utilities."""

from __future__ import annotations

__all__ = [
    "PipelineOptions",
    "SurfaceOutputFormat",
    "SurfaceResult",
    "save_surface",
    "voxel_to_surface",
]


def __getattr__(name: str):
    if name in __all__:
        import lib.converters.voxel2surf as _voxel2surf

        return getattr(_voxel2surf, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
