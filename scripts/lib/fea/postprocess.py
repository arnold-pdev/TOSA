"""Boundary shape-derivative post-processing."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lib.fea.types import FEASolution


def boundary_eps_sigma_contraction(
    solution: FEASolution,
    surface_path: Path,
    *,
    output_path: Path,
) -> Path:
    """
    Compute ε:σ on exterior faces and write scalar fields to a .vtp surface.

    Returns the path to the enriched surface file.
    """
    raise NotImplementedError(
        "Boundary shape-derivative post-processing is not yet implemented."
    )
