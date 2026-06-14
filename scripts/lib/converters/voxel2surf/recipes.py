"""Named pipeline recipes (extractor + smoother + stage list)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Recipe:
    id: str
    extractor: str
    smoother: str
    stages: tuple[str, ...]
    description: str = ""


def _base_bc_stages(*, with_smooth: bool) -> tuple[str, ...]:
    stages = ["extract", "bc_build", "bc_transfer", "bc_enforce"]
    if with_smooth:
        stages.extend(["subdivide", "smooth", "bc_enforce"])
    stages.append("validate")
    return tuple(stages)


RECIPES: dict[str, Recipe] = {
    "v3_default": Recipe(
        id="v3-pyvista-laplacian",
        extractor="pyvista_binary",
        smoother="laplacian_bc",
        stages=_base_bc_stages(with_smooth=True),
        description="PyVista binary extract + BC pin + Laplacian smooth",
    ),
    "v3_laplacian": Recipe(
        id="v3-pyvista-laplacian",
        extractor="pyvista_binary",
        smoother="laplacian_bc",
        stages=_base_bc_stages(with_smooth=True),
        description="Alias for v3_default",
    ),
    "v3_taubin": Recipe(
        id="v3-pyvista-taubin",
        extractor="pyvista_binary",
        smoother="taubin_bc",
        stages=_base_bc_stages(with_smooth=True),
        description="PyVista binary extract + BC pin + Taubin smooth",
    ),
    "sdf_vf_match": Recipe(
        id="sdf-lewiner-taubin",
        extractor="sdf_lewiner",
        smoother="taubin_bc",
        stages=_base_bc_stages(with_smooth=True),
        description="SDF + VF isolevel calibration + Lewiner MC + Taubin",
    ),
    "extract_only": Recipe(
        id="extract-only",
        extractor="pyvista_binary",
        smoother="none",
        stages=("extract", "bc_build", "bc_transfer", "bc_enforce", "validate"),
        description="No smooth — baseline extract and BC enforcement",
    ),
}


def get_recipe(name: str) -> Recipe:
    key = name.strip().lower()
    if key not in RECIPES:
        choices = ", ".join(sorted(RECIPES))
        raise ValueError(f"Unknown recipe {name!r}. Choose from: {choices}")
    return RECIPES[key]


def list_recipes() -> list[Recipe]:
    seen: set[str] = set()
    out: list[Recipe] = []
    for recipe in RECIPES.values():
        if recipe.id in seen:
            continue
        seen.add(recipe.id)
        out.append(recipe)
    return out
