"""Load NITO .npy bundles from nito/Data/ (shared by TOSA scripts)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lib.paths import REPO_ROOT

DEFAULT_DATA_DIR = REPO_ROOT / "nito" / "Data"

REQUIRED_FILES = (
    "shapes.npy",
    "topologies.npy",
    "boundary_conditions.npy",
    "loads.npy",
    "vfs.npy",
)


def resolve_data_dir(data_dir: Path | None, *, test: bool) -> Path:
    root = (data_dir or DEFAULT_DATA_DIR).resolve()
    return root / "Test" if test else root


def load_nito_arrays(data_dir: Path) -> dict[str, np.ndarray]:
    missing = [name for name in REQUIRED_FILES if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing in {data_dir}: {', '.join(missing)}. "
            "Run: ./scripts/fetch/data_2d.sh --test-only"
        )
    return {
        "shapes": np.load(data_dir / "shapes.npy", allow_pickle=True),
        "topologies": np.load(data_dir / "topologies.npy", allow_pickle=True),
        "boundary_conditions": np.load(
            data_dir / "boundary_conditions.npy", allow_pickle=True
        ),
        "loads": np.load(data_dir / "loads.npy", allow_pickle=True),
        "vfs": np.load(data_dir / "vfs.npy", allow_pickle=True),
    }


def load_sample(
    data: dict[str, np.ndarray], index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    shape = np.asarray(data["shapes"][index], dtype=int)
    rho = np.asarray(data["topologies"][index], dtype=float)
    bc = np.asarray(data["boundary_conditions"][index], dtype=float)
    load = np.asarray(data["loads"][index], dtype=float)
    vf = float(data["vfs"][index])
    return shape, rho, bc, load, vf


def shape_ndim(shape: np.ndarray) -> int:
    """Return 2 or 3 from a NITO shape vector (nelx, nely[, nelz])."""
    n = int(np.asarray(shape, dtype=int).size)
    if n not in (2, 3):
        raise ValueError(f"Expected 2D or 3D shape, got size {n}: {shape}")
    return n


def indices_by_ndim(shapes: np.ndarray) -> dict[int, np.ndarray]:
    """Map dimension (2 or 3) → sorted array of sample indices."""
    ndim = np.fromiter(
        (shape_ndim(s) for s in shapes), dtype=int, count=len(shapes)
    )
    return {int(d): np.flatnonzero(ndim == d) for d in np.unique(ndim)}


def summarize_dimensions(shapes: np.ndarray) -> dict:
    """Summary stats for 2D vs 3D (and any other) samples."""
    by_dim = indices_by_ndim(shapes)
    n = len(shapes)
    counts = {d: len(idxs) for d, idxs in by_dim.items()}
    n_elem = np.array([int(np.prod(shapes[i])) for i in range(n)])

    def _subset_stats(idxs: np.ndarray) -> dict:
        if idxs.size == 0:
            return {"count": 0}
        sub = n_elem[idxs]
        i_min = int(idxs[int(np.argmin(sub))])
        i_max = int(idxs[int(np.argmax(sub))])
        return {
            "count": int(idxs.size),
            "n_elements_min": int(sub.min()),
            "n_elements_max": int(sub.max()),
            "smallest_index": i_min,
            "largest_index": i_max,
            "smallest_shape": tuple(int(x) for x in shapes[i_min]),
            "largest_shape": tuple(int(x) for x in shapes[i_max]),
        }

    unique_templates: dict[int, list] = {}
    for d, idxs in by_dim.items():
        if d not in (2, 3):
            continue
        templates = {}
        for i in idxs:
            key = tuple(int(x) for x in shapes[i])
            templates.setdefault(key, []).append(int(i))
        unique_templates[d] = [
            {"shape": k, "count": len(v), "example_index": v[0]}
            for k, v in sorted(templates.items(), key=lambda x: -len(x[1]))
        ]

    return {
        "n_samples": n,
        "counts_by_ndim": counts,
        "dim_2d": _subset_stats(by_dim.get(2, np.array([], dtype=int))),
        "dim_3d": _subset_stats(by_dim.get(3, np.array([], dtype=int))),
        "unique_shapes": unique_templates,
        "indices_2d": by_dim.get(2, np.array([], dtype=int)).tolist(),
        "indices_3d": by_dim.get(3, np.array([], dtype=int)).tolist(),
    }
