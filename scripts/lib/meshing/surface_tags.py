"""
Canonical VTK surface tag schema (voxel2surf → Gmsh / TetGen / FEA).

VTK PolyData (.vtp / .vtk) is the only supported surface interchange format.

Per-triangle cell_data
----------------------
  patch_id      int32   -1 = free (design) boundary; 0…n-1 = BC patch index
  facet_marker  int32   meshing physical-group / TetGen marker (see below)
  fix_x/y/z     uint8   NITO constraint flags broadcast per triangle

FieldData (patch catalogue)
---------------------------
  n_patches     int
  patch_flags   (n_patches, 3) float — NITO cols dim:2×dim, zero-padded to 3D
  patch_axis    (n_patches,) int
  patch_side    (n_patches,) int — 0=min, 1=max

Facet marker convention
-----------------------
  1             free exterior (shape-derivative trace lives here)
  2 + patch_id  BC patch i (patch_id >= 0)
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PATCH_ID_KEY = "patch_id"
FACET_MARKER_KEY = "facet_marker"
FREE_BOUNDARY_ID = -1
FREE_FACET_MARKER = 1
BC_FACET_MARKER_OFFSET = 2

FIELD_N_PATCHES = "n_patches"
FIELD_PATCH_FLAGS = "patch_flags"
FIELD_PATCH_AXIS = "patch_axis"
FIELD_PATCH_SIDE = "patch_side"

FIX_KEYS = ("fix_x", "fix_y", "fix_z")
SIDE_TO_INT = {"min": 0, "max": 1}
INT_TO_SIDE = {0: "min", 1: "max"}

VTK_SUFFIXES = {".vtp", ".vtk"}


@dataclass(frozen=True)
class PatchMetadata:
    flags: np.ndarray
    axis: int
    side: str


@dataclass(frozen=True)
class SurfaceTags:
    """Tagged surface read from or destined for a VTK PolyData file."""

    mesh: object  # pyvista.PolyData
    patch_id: np.ndarray
    facet_markers: np.ndarray
    patches: list[PatchMetadata]
    fix_per_cell: np.ndarray  # (n_cells, 3)


def _max_dim(patches: list) -> int:
    if not patches:
        return 3
    return max(3, max(len(np.asarray(p.flags).ravel()) for p in patches))


def facet_marker_from_patch_id(patch_id: np.ndarray) -> np.ndarray:
    """Map patch_id (-1 free, 0+ BC) to meshing facet markers."""
    pid = np.asarray(patch_id, dtype=np.int32)
    markers = np.full(pid.shape, FREE_FACET_MARKER, dtype=np.int32)
    bc = pid >= 0
    markers[bc] = BC_FACET_MARKER_OFFSET + pid[bc]
    return markers


def fix_per_cell_from_patches(
    patch_id: np.ndarray, patches: list, *, dim: int = 3
) -> np.ndarray:
    n = len(patch_id)
    out = np.zeros((n, dim), dtype=np.uint8)
    for i, patch in enumerate(patches):
        mask = patch_id == i
        if not np.any(mask):
            continue
        flags = np.asarray(patch.flags, dtype=float).ravel()
        out[mask, : min(dim, len(flags))] = (flags[:dim] > 0.5).astype(np.uint8)
    return out


def encode_surface_vtp(result) -> object:
    """
    Build a PyVista PolyData with the canonical surface tag schema.

    `result` is a voxel2surf.SurfaceResult.
    """
    import pyvista as pv

    mesh = result.mesh
    faces = np.hstack(
        [np.full((len(mesh.faces), 1), 3, dtype=np.int64), mesh.faces]
    ).ravel()
    poly = pv.PolyData(np.asarray(mesh.vertices, dtype=float), faces)

    patch_id = np.asarray(result.tri_labels, dtype=np.int32)
    patches = [
        PatchMetadata(
            flags=np.asarray(p.flags, dtype=float),
            axis=int(p.axis),
            side=str(p.side),
        )
        for p in result.patches
    ]
    dim = _max_dim(result.patches)
    facet_markers = facet_marker_from_patch_id(patch_id)
    fix = fix_per_cell_from_patches(patch_id, result.patches, dim=dim)

    poly.cell_data[PATCH_ID_KEY] = patch_id
    poly.cell_data[FACET_MARKER_KEY] = facet_markers
    for j, key in enumerate(FIX_KEYS[:dim]):
        poly.cell_data[key] = fix[:, j]
    if dim < 3:
        for key in FIX_KEYS[dim:]:
            poly.cell_data[key] = np.zeros(len(patch_id), dtype=np.uint8)

    poly.point_data["freeze"] = np.asarray(result.freeze, dtype=float)

    n = len(patches)
    poly.field_data[FIELD_N_PATCHES] = np.array([n], dtype=np.int32)
    if n:
        flag_rows = []
        for p in patches:
            row = np.zeros(3, dtype=float)
            f = np.asarray(p.flags, dtype=float).ravel()
            row[: min(3, len(f))] = f[:3]
            flag_rows.append(row)
        poly.field_data[FIELD_PATCH_FLAGS] = np.stack(flag_rows, axis=0)
        poly.field_data[FIELD_PATCH_AXIS] = np.array(
            [p.axis for p in patches], dtype=np.int32
        )
        poly.field_data[FIELD_PATCH_SIDE] = np.array(
            [SIDE_TO_INT.get(p.side, 0) for p in patches], dtype=np.int32
        )
    else:
        poly.field_data[FIELD_PATCH_FLAGS] = np.zeros((0, 3), dtype=float)
        poly.field_data[FIELD_PATCH_AXIS] = np.zeros(0, dtype=np.int32)
        poly.field_data[FIELD_PATCH_SIDE] = np.zeros(0, dtype=np.int32)

    return poly


def read_surface_tags(path: Path) -> SurfaceTags:
    """Load a tagged VTK surface and decode meshing / FEA boundary metadata."""
    import pyvista as pv

    path = path.expanduser().resolve()
    if path.suffix.lower() not in VTK_SUFFIXES:
        raise ValueError(
            f"Surface tags require VTK PolyData ({', '.join(sorted(VTK_SUFFIXES))}); "
            f"got {path.suffix!r}"
        )

    mesh = pv.read(path)
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    if mesh.n_cells == 0:
        raise ValueError(f"No surface cells in {path}")

    if PATCH_ID_KEY not in mesh.cell_data:
        raise KeyError(
            f"{path} missing cell_data[{PATCH_ID_KEY!r}]. "
            "Run voxel2surf with BC patches (default) to produce a tagged surface."
        )

    patch_id = np.asarray(mesh.cell_data[PATCH_ID_KEY], dtype=np.int32)
    if FACET_MARKER_KEY in mesh.cell_data:
        facet_markers = np.asarray(mesh.cell_data[FACET_MARKER_KEY], dtype=np.int32)
    else:
        facet_markers = facet_marker_from_patch_id(patch_id)

    n_patches = int(mesh.field_data.get(FIELD_N_PATCHES, [0])[0])
    patches: list[PatchMetadata] = []
    if n_patches and FIELD_PATCH_FLAGS in mesh.field_data:
        flags = np.asarray(mesh.field_data[FIELD_PATCH_FLAGS], dtype=float)
        axes = np.asarray(mesh.field_data.get(FIELD_PATCH_AXIS, []), dtype=int)
        sides = np.asarray(mesh.field_data.get(FIELD_PATCH_SIDE, []), dtype=int)
        for i in range(n_patches):
            patches.append(
                PatchMetadata(
                    flags=flags[i],
                    axis=int(axes[i]) if i < len(axes) else 0,
                    side=INT_TO_SIDE.get(int(sides[i]), "min") if i < len(sides) else "min",
                )
            )
    else:
        n_bc = int(np.max(patch_id)) + 1 if np.any(patch_id >= 0) else 0
        for _ in range(n_bc):
            patches.append(PatchMetadata(flags=np.zeros(3), axis=0, side="min"))

    dim = 3
    if all(k in mesh.cell_data for k in FIX_KEYS):
        fix = np.column_stack(
            [np.asarray(mesh.cell_data[k], dtype=np.uint8) for k in FIX_KEYS]
        )
    else:
        fix = fix_per_cell_from_patches(patch_id, patches, dim=dim)

    return SurfaceTags(
        mesh=mesh,
        patch_id=patch_id,
        facet_markers=facet_markers,
        patches=patches,
        fix_per_cell=fix,
    )


def physical_groups_by_marker(tags: SurfaceTags) -> dict[int, list[int]]:
    """Map facet_marker → sorted triangle indices (for Gmsh physical groups)."""
    groups: dict[int, list[int]] = {}
    for i, marker in enumerate(tags.facet_markers.tolist()):
        groups.setdefault(int(marker), []).append(i)
    return {k: sorted(v) for k, v in sorted(groups.items())}


def marker_names(tags: SurfaceTags) -> dict[int, str]:
    """Human-readable names for facet markers."""
    names = {FREE_FACET_MARKER: "boundary_free"}
    for i in range(len(tags.patches)):
        names[BC_FACET_MARKER_OFFSET + i] = f"bc_patch_{i}"
    return names


_AXIS_NAMES = "xyz"


def constraint_kind_name(flags: np.ndarray) -> str:
    """Describe NITO constraint flags as a short boundary-type label."""
    bits = tuple(1 if x > 0.5 else 0 for x in np.asarray(flags, dtype=float).ravel()[:3])
    n_fixed = sum(bits)
    if n_fixed == 0:
        return "free"
    if n_fixed == len(bits) and all(bits):
        return "clamp"
    if len(bits) == 2:
        if bits == (1, 1):
            return "clamp"
        if bits == (1, 0):
            return "roller (+y)"
        if bits == (0, 1):
            return "roller (+x)"
    fixed = "".join(_AXIS_NAMES[i] for i, b in enumerate(bits) if b)
    free = "".join(_AXIS_NAMES[i] for i, b in enumerate(bits) if not b)
    if free:
        return f"fix {fixed}, free {free}"
    return f"fix {fixed}"


def patch_label(patch_idx: int, meta: PatchMetadata | None) -> str:
    if patch_idx < 0:
        return "free boundary"
    if meta is None:
        return f"BC patch {patch_idx}"
    if meta.axis < 0:
        return f"load patch {patch_idx}"
    kind = constraint_kind_name(meta.flags)
    axis = _AXIS_NAMES[meta.axis] if 0 <= meta.axis < 3 else str(meta.axis)
    return f"patch {patch_idx}: {kind} ({axis}-{meta.side})"


PATCH_CMAP_NAME = "tab20"


def patch_display_order(patch_id: np.ndarray | Iterable[int]) -> list[int]:
    """Sorted unique patch_id values (same order as boundary scalar coloring)."""
    return sorted({int(v) for v in np.asarray(patch_id).ravel()})


def merge_patch_display_order(
    patch_id: np.ndarray | Iterable[int],
    *,
    n_bc: int = 0,
) -> list[int]:
    """
    Union of patch_ids on the surface and all NITO BC patch indices.

    BC voxel overlays enumerate every NITO patch even when transfer tagged zero
    surface triangles (e.g. a thin slab with no iso-surface intersection).
    """
    ids = {int(v) for v in np.asarray(patch_id).ravel()}
    if n_bc > 0:
        ids.update(range(int(n_bc)))
    return sorted(ids)


def patch_categorical_cmap(n: int):
    """Matplotlib ListedColormap for categorical patch display."""
    from matplotlib import cm, colors

    n = max(int(n), 1)
    try:
        base = cm.colormaps[PATCH_CMAP_NAME].resampled(n)
    except AttributeError:
        base = cm.get_cmap(PATCH_CMAP_NAME, n)
    return colors.ListedColormap(base(np.linspace(0, 1, n)))


def patch_display_rgba(patch_id: int, order: Sequence[int]) -> tuple[float, float, float, float]:
    """RGBA in [0, 1] for a patch_id using the shared categorical palette."""
    order = [int(v) for v in order]
    cmap = patch_categorical_cmap(len(order))
    idx = order.index(int(patch_id))
    return tuple(float(x) for x in cmap(idx))


def patch_display_color_hex(patch_id: int, order: Sequence[int]) -> str:
    """Hex color for a patch_id (for PyVista mesh color=...)."""
    from matplotlib import colors as mcolors

    rgba = patch_display_rgba(patch_id, order)
    return mcolors.to_hex(rgba[:3])


def nito_patch_display_order(n_bc: int, n_load: int) -> list[int]:
    """patch_id order for an untagged voxel2surf surface (free + BC + loads)."""
    order = [-1]
    n = int(n_bc) + int(n_load)
    if n > 0:
        order.extend(range(n))
    return order


def prepare_patch_display(
    tags: SurfaceTags,
    *,
    order: Sequence[int] | None = None,
) -> tuple[object, str, list[str]]:
    """
    Attach a contiguous cell scalar for categorical coloring by boundary patch.

    Returns (mesh, scalar_name, legend_labels).
    """
    mesh = tags.mesh.copy()
    pid = np.asarray(tags.patch_id, dtype=np.int32)
    if order is None:
        order = patch_display_order(pid)
    else:
        order = [int(v) for v in order]
    remap = {v: i for i, v in enumerate(order)}
    mesh.cell_data["patch_display"] = np.array(
        [remap[int(p)] for p in pid], dtype=np.int32
    )
    labels = [
        patch_label(
            v,
            tags.patches[v] if v >= 0 and v < len(tags.patches) else None,
        )
        for v in order
    ]
    return mesh, "patch_display", labels


def try_read_surface_tags(path: Path) -> SurfaceTags | None:
    """Load tags when patch_id is present; return None for untagged legacy surfaces."""
    if path.suffix.lower() not in VTK_SUFFIXES:
        return None
    import pyvista as pv

    mesh = pv.read(path)
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    if PATCH_ID_KEY not in mesh.cell_data:
        return None
    return read_surface_tags(path)


def write_surface_vtp(result, path: Path, *, fmt: str | None = None) -> Path:
    """Encode and write a tagged VTK surface."""
    path = path.expanduser().resolve()
    suffix = path.suffix.lower()
    if fmt is not None:
        fmt = str(fmt).lower().lstrip(".")
        path = path.with_suffix(f".{fmt}")
    elif suffix not in VTK_SUFFIXES:
        path = path.with_suffix(".vtp")
    path.parent.mkdir(parents=True, exist_ok=True)
    encode_surface_vtp(result).save(path)
    return path
