"""FEA types and on-disk artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MaterialProperties:
    youngs_modulus: float = 1.0
    poisson_ratio: float = 0.33


@dataclass(frozen=True)
class FEASolution:
    """Displacement solution and optional cached paths."""

    index: int
    displacement_path: Path
    compliance: float | None = None
    shape_derivative_path: Path | None = None


@dataclass(frozen=True)
class ShapeDerivativeOnSurface:
    """Shape-derivative scalar field transferred to the design surface."""

    index: int
    field_name: str
    values: np.ndarray
    surface_path: Path
    output_path: Path
