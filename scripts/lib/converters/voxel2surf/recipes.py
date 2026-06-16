"""Named pipeline recipes (extractor + smoother + stage list)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass


@dataclass(frozen=True)
class Recipe:
    id: str
    extractor: str
    smoother: str
    stages: tuple[str, ...]
    description: str = ""
    bc_assembly: str = "monolithic"


def _monolithic_stages(
    *,
    with_smooth: bool,
    with_regrow: bool = False,
    with_snap: bool = False,
    with_planar_cap: bool = False,
    pre_bc_build: bool = False,
) -> tuple[str, ...]:
    if pre_bc_build:
        stages: list[str] = ["bc_build", "extract", "bc_transfer", "bc_enforce"]
    else:
        stages = ["extract", "bc_build", "bc_transfer", "bc_enforce"]
    if with_planar_cap:
        stages.append("bc_planar_cap")
    if with_regrow:
        stages.append("bc_regrow")
    if with_smooth:
        stages.extend(["subdivide", "smooth"])
        if with_snap:
            stages.append("snap")
        stages.append("bc_enforce")
    stages.append("validate")
    return tuple(stages)


def _shared_seam_stages(*, with_smooth: bool = True) -> tuple[str, ...]:
    stages = ["extract", "bc_build", "bc_transfer", "bc_assembly", "bc_enforce"]
    if with_smooth:
        stages.extend(["subdivide", "smooth", "bc_enforce"])
    stages.append("validate")
    return tuple(stages)


RECIPES: dict[str, Recipe] = {
    "pyvista_laplacian": Recipe(
        id="pyvista-laplacian",
        extractor="pyvista_binary",
        smoother="laplacian_bc",
        stages=_monolithic_stages(with_smooth=True),
        description="PyVista binary extract + BC pin + Laplacian smooth",
    ),
    "pyvista_taubin": Recipe(
        id="pyvista-taubin",
        extractor="pyvista_binary",
        smoother="taubin_bc",
        stages=_monolithic_stages(with_smooth=True),
        description="PyVista binary extract + BC pin + Taubin smooth",
    ),
    "pyvista_shared_seam_taubin": Recipe(
        id="pyvista-shared-seam-taubin",
        extractor="pyvista_binary",
        smoother="taubin_bc",
        stages=_shared_seam_stages(with_smooth=True),
        description="PyVista extract + shared-seam BC caps + Taubin (critique a)",
        bc_assembly="shared_seam",
    ),
    "pyvista_taubin_regrow": Recipe(
        id="pyvista-taubin-regrow",
        extractor="pyvista_binary",
        smoother="taubin_bc",
        stages=_monolithic_stages(with_smooth=True, with_regrow=True),
        description="PyVista extract + geodesic BC regrow + Taubin",
    ),
    "pyvista_hc": Recipe(
        id="pyvista-hc",
        extractor="pyvista_binary",
        smoother="hc_bc",
        stages=_monolithic_stages(with_smooth=True),
        description="PyVista extract + Humphrey–Carlson smooth + BC pin",
    ),
    "sdf_lewiner_calibrated": Recipe(
        id="sdf-lewiner-calibrated-taubin",
        extractor="sdf_lewiner",
        smoother="taubin_bc",
        stages=_monolithic_stages(with_smooth=True),
        description="SDF + VF isolevel calibration + Lewiner MC + Taubin",
    ),
    "sdf_field_smooth_taubin": Recipe(
        id="sdf-field-smooth-taubin",
        extractor="sdf_field_smooth",
        smoother="tangential_bc",
        stages=_monolithic_stages(with_smooth=True),
        description="Gaussian SDF smooth + VF cal + Lewiner MC + tangential Taubin (critique c)",
    ),
    "sdf_bc_planes_taubin": Recipe(
        id="sdf-bc-planes-taubin",
        extractor="sdf_bc_planes",
        smoother="taubin_bc",
        stages=_monolithic_stages(with_smooth=True, pre_bc_build=True),
        description="BC-plane pinned SDF + Lewiner MC + Taubin (critique b lite)",
    ),
    "sdf_masked_taubin": Recipe(
        id="sdf-masked-taubin",
        extractor="sdf_masked",
        smoother="taubin_bc",
        stages=_monolithic_stages(with_smooth=True, pre_bc_build=True),
        description="BC-masked SDF smooth + Lewiner MC + Taubin",
    ),
    "sdf_lewiner_raw": Recipe(
        id="sdf-lewiner-raw-taubin",
        extractor="sdf_lewiner",
        smoother="taubin_bc",
        stages=_monolithic_stages(with_smooth=True),
        description="SDF Lewiner MC (no VF calibrate) + Taubin",
    ),
    "sdf_lewiner_snap": Recipe(
        id="sdf-lewiner-snap-taubin",
        extractor="sdf_lewiner",
        smoother="taubin_bc",
        stages=_monolithic_stages(with_smooth=True, with_snap=True),
        description="SDF + Taubin + SDF vertex snap for VF",
    ),
    "extract_baseline": Recipe(
        id="extract-no-smooth",
        extractor="pyvista_binary",
        smoother="none",
        stages=("extract", "bc_build", "bc_transfer", "bc_enforce", "validate"),
        description="No smooth — stairstep extract and BC enforcement only",
    ),
    "planar_cap_contour": Recipe(
        id="planar-cap-contour",
        extractor="pyvista_binary",
        smoother="none",
        stages=_monolithic_stages(with_smooth=False, with_planar_cap=True),
        description="Monolithic extract + planar cap contour audit (caps not assembled)",
    ),
}

# Deprecated CLI aliases (removed in a future release).
_ALIASES: dict[str, str] = {
    "v3_default": "pyvista_laplacian",
    "v3_laplacian": "pyvista_laplacian",
    "v3_taubin": "pyvista_taubin",
    "v3_regrow": "pyvista_taubin_regrow",
    "v3_hc": "pyvista_hc",
    "sdf_vf_match": "sdf_lewiner_calibrated",
    "sdf_masked": "sdf_masked_taubin",
    "sdf_raw_taubin": "sdf_lewiner_raw",
    "sdf_snap": "sdf_lewiner_snap",
    "extract_only": "extract_baseline",
    "pyvista_laplacian_cap_inline": "pyvista_laplacian",
    "pyvista_taubin_cap_inline": "pyvista_taubin",
    "cuberille_stitch_taubin": "extract_baseline",
    "hybrid_bc_iso": "extract_baseline",
}


def get_recipe(name: str) -> Recipe:
    key = name.strip().lower().replace("-", "_")
    if key in _ALIASES:
        canonical = _ALIASES[key]
        warnings.warn(
            f"Recipe {name!r} is deprecated; use {canonical!r} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        key = canonical
    if key not in RECIPES:
        choices = ", ".join(sorted(RECIPES))
        raise ValueError(f"Unknown recipe {name!r}. Choose from: {choices}")
    return RECIPES[key]


def list_recipes() -> list[Recipe]:
    return list(RECIPES.values())
