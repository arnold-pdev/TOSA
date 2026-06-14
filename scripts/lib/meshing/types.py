"""Mesh backend and artifact types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class MeshBackend(str, Enum):
    GMSH = "gmsh"
    TETGEN = "tetgen"


@dataclass(frozen=True)
class VolumeMesh:
    """On-disk volume mesh for a single sample."""

    index: int
    backend: MeshBackend
    path: Path  # .msh or .xdmf
    meta_path: Path | None = None
