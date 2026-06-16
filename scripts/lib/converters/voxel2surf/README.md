# voxel2surf

Convert NITO voxel topologies into tagged VTK surfaces (`.vtp`) for FEA meshing.

**Entry point:** `python scripts/lib/converters/voxel2surf.py --index N`

**Package:** `scripts/lib/converters/voxel2surf/`

---

## Quick start

```bash
# List named pipelines
python scripts/lib/converters/voxel2surf.py --list-recipes

# Default production path
python scripts/lib/converters/voxel2surf.py --index 119 --recipe pyvista_taubin

# Experiment run directory (VTP, logs, metrics CSV, manifest)
python scripts/lib/converters/voxel2surf.py --index 119 \
  --recipe pyvista_taubin \
  --run-dir output/surfaces/runs/pyvista-taubin-119
```

Canonical probe index: **119**.

**Preferred production path:** `pyvista_taubin` (VF-neutral mesh smooth) or `pyvista_laplacian` (lighter smooth).

---

## Current status

### Production default (monolithic BC)

The default pipeline keeps BC faces on the extracted iso-surface:

```
extract → bc_build → bc_transfer → bc_enforce → [smooth] → bc_enforce → validate
```

Triangles are labeled by voxel footprint (`bc_transfer`), coplanar faces are claimed on NITO planes (`bc_claim_coplanar`), then hard-snapped (`bc_enforce`). Smoothing freezes BC vertices and re-projects them each iteration.

On index **119** this path typically yields:

| Metric | Typical value |
|--------|----------------|
| `bc_plane_max_residual` | 0 |
| `bc_footprint_coverage` | ~99.8% (with coplanar claim) |
| `bc_labeled_triangles` | ~3,051 |
| Load checks | pass |
| Watertight | yes (single body) |

### Removed: cap weld / stitch assembly

Earlier work built planar caps separately and welded them to the free skin (`stitch`, rim weld, coplanar strip). That path was **removed** — independent rim tessellation cannot be made watertight by tolerance snapping. Planar contouring (`bc_planar_cap`) remains for **audit only** (caps are not assembled onto the skin).

### Experimental: critique-proposal backends

Three alternative backends are wired as **parallel stage slots** (same recipe/stage pattern as extract × smooth) for validation comparison:

| Recipe | Stage | Status on 119 |
|--------|-------|----------------|
| `pyvista_shared_seam_taubin` | `bc_assembly=shared_seam` | Runs; **not watertight** (`body_count=5`) — rim chain + merge need edge-split work |
| `sdf_field_smooth_taubin` | `sdf_field_smooth` + `tangential_bc` | Runs; VF calibration needs tuning (~+2% delta) |
| `sdf_bc_planes_taubin` | `sdf_bc_planes` extract | Runs; incremental BC-at-field step (~−1% VF delta) |

Compare with `./batch/run_validation_critique_methods.sh` (dataset `critique-methods-v1`).

Full dual contouring / OpenVDB surfacing is **not** implemented; `sdf_bc_planes` pins the SDF on BC slabs before Lewiner MC.

---

## Recipes

Recipes are named by what they do — no version prefixes.

### Production

| CLI `--recipe` | What it does |
|----------------|--------------|
| `pyvista_laplacian` | PyVista iso-surface + BC pin + Laplacian smooth |
| `pyvista_taubin` | **Preferred** — PyVista + BC pin + Taubin smooth |
| `pyvista_taubin_regrow` | PyVista + geodesic BC regrow + Taubin |
| `pyvista_hc` | PyVista + Humphrey–Carlson smooth + BC pin |
| `sdf_lewiner_calibrated` | SDF Lewiner MC + VF isolevel calibration + Taubin |
| `sdf_masked_taubin` | BC-masked SDF field + Lewiner MC + Taubin |
| `sdf_lewiner_raw` | SDF Lewiner MC (no VF calibration) + Taubin |
| `sdf_lewiner_snap` | SDF + Taubin + post-smooth SDF vertex snap |
| `extract_baseline` | Extract + BC enforce only (no smooth) |

### Experimental (critique proposals)

| CLI `--recipe` | What it does |
|----------------|--------------|
| `pyvista_shared_seam_taubin` | Plane-cut seam + Delaunay caps + merge + Taubin |
| `sdf_field_smooth_taubin` | Gaussian SDF smooth + VF cal + tangential Taubin |
| `sdf_bc_planes_taubin` | BC-plane pinned SDF (`bc_build` before extract) + Taubin |

### QA / audit

| CLI `--recipe` | What it does |
|----------------|--------------|
| `planar_cap_contour` | Monolithic extract + planar cap contour audit (caps not on skin) |

Deprecated aliases (`v3_taubin`, `extract_only`, `cuberille_stitch_taubin`, …) still resolve with a warning.

---

## Pipeline architecture

Stages run in recipe order. Each stage exposes swappable backends via recipe defaults or CLI flags.

| Module | Stages |
|--------|--------|
| `stages/extract.py` | `extract` |
| `stages/bc.py` | `bc_build`, `bc_transfer`, `bc_enforce`, `bc_planar_cap` |
| `stages/bc_assembly.py` | `bc_assembly` |
| `stages/refine.py` | `subdivide`, `smooth`, `snap`, `bc_regrow` |
| `stages/validate.py` | `validate` |

### Monolithic path (default)

```
extract → bc_build → bc_transfer → bc_enforce
  → [bc_planar_cap] → [bc_regrow] → [subdivide] → [smooth] → [snap] → bc_enforce → validate
```

`bc_assembly` is omitted (or `monolithic` = no-op).

### Shared-seam path (`pyvista_shared_seam_taubin`)

```
extract → bc_build → bc_transfer → bc_assembly → bc_enforce
  → [subdivide] → [smooth] → bc_enforce → validate
```

`bc_assembly=shared_seam` strips MC BC caps, builds planar caps on plane-cut rim polylines, merges with the free skin (`lib/meshing/shared_seam.py`).

### SDF BC-planes path (`sdf_bc_planes_taubin`, `sdf_masked_taubin`)

```
bc_build → extract → bc_transfer → bc_enforce → …
```

Patches must exist before extract so the SDF can be pinned or masked at BC slabs.

### Parallel backends

| Stage | Option | Backends |
|-------|--------|----------|
| **extract** | `--extractor` | `pyvista_binary`, `sdf_lewiner`, `sdf_masked`, `sdf_field_smooth`, `sdf_bc_planes` |
| **bc_assembly** | `--bc-assembly` | `monolithic` (default), `shared_seam` |
| **smooth** | `--smoother` | `laplacian_bc`, `taubin_bc`, `hc_bc`, `tangential_bc`, `none` |

Registries: `stages/extract.py`, `stages/bc_assembly.py` (`BC_ASSEMBLY_RUNNERS`), `stages/refine.py`.

Implementation entry points for critique methods:

- `lib/meshing/shared_seam.py` — seam extraction + cap merge
- `lib/converters/bc_planar_cap.py` — `triangulate_cap_on_plane`, contour methods
- `lib/meshing/field.py` — SDF, BC Dirichlet pin, masked smooth
- `lib/meshing/surface_smooth.py` — `tangential_taubin_smooth`

---

## Stage reference

### `extract`

Build the initial iso-surface from the voxel solid.

| Option | CLI flag | Default | Description |
|--------|----------|---------|-------------|
| `extractor` | `--extractor` | recipe | See parallel backends table |
| `iso_level` | `--iso-level` | `0.5` | PyVista contour level |
| `repair` | `--no-repair` | `True` | trimesh / MeshFix cleanup |
| `meshfix_verbose` | `--meshfix-verbose` | `False` | MeshFix log noise |
| `upsample_factor` | `--upsample-factor` | `1` | Voxel upsample before SDF |
| `field_smooth_sigma` | `--field-smooth-sigma` | `0.0` | Gaussian SDF smooth (σ in voxels); `sdf_field_smooth` defaults to 0.35 |
| `field_bc_mask` | `--field-bc-mask` | `False` | Mask SDF at BC planes during field smooth |
| `calibrate_vf` | `--calibrate-vf` | recipe | Bisect SDF isolevel to match voxel VF |

### `bc_build`

Build analytic NITO BC patches from `bcspecs` / `bc.npy` rows. No tunable options.

### `bc_transfer`

Label mesh triangles with patch IDs from voxel footprints.

| Option | CLI flag | Default | Description |
|--------|----------|---------|-------------|
| `bc_transfer` | `--bc-transfer` | `centroid` | `centroid` or `cell_native` (stricter) |
| `bc_transfer_band_cells` | `--bc-transfer-band-cells` | `1.0` | Plane proximity band in cells |

### `bc_enforce`

Snap BC triangles onto NITO analytic planes; flatten interiors; audit footprint.

| Option | CLI flag | Default | Description |
|--------|----------|---------|-------------|
| `bc_claim_coplanar` | `--no-bc-claim-coplanar` | `True` | Claim unlabeled tris on BC planes in footprint |
| `bc_strict_footprint` | `--bc-strict-footprint` | `False` | Drop labeled tris outside voxel footprint |
| `bc_plane_tol` | `--bc-plane-tol` | `1e-5` | Coplanarity tolerance (world units) |
| `on_bc_fail` | `--on-bc-fail` | `raise` | `raise` or `warn` on plane residual breach |
| `bc_oracle` | `--bc-oracle` | `False` | Cuberille gap metrics (QA) |

Runs twice in smoothed recipes (before and after smooth).

### `bc_assembly`

Replace MC BC caps with analytic planar caps (shared-seam backend only).

| Option | CLI flag | Default | Description |
|--------|----------|---------|-------------|
| `bc_assembly` | `--bc-assembly` | recipe | `monolithic` (no-op) or `shared_seam` |

Steps for `shared_seam`: strip labeled BC tris → plane-cut rim segments → chain to loops → Delaunay cap per patch → merge vertices with free skin.

### `bc_planar_cap` (optional, audit-only)

Contour voxel footprints into smooth planar caps. Writes cap mesh to `state.bc_mesh` and audit metrics; does **not** modify the skin mesh.

| Option | CLI flag | Default | Description |
|--------|----------|---------|-------------|
| `bc_planar_cap` | `--bc-planar-cap` | `False` | Enable contour audit stage |
| `bc_planar_cap_method` | `--bc-planar-cap-method` | `upsample_blur` | **`upsample_blur`** (preferred), `native_contour` |
| `bc_planar_cap_upsample` | `--bc-planar-cap-upsample` | `4` | Footprint mask upsample factor |
| `bc_planar_cap_blur_sigma` | `--bc-planar-cap-blur-sigma` | `1.0` | Blur σ in cells (`upsample_blur`) |
| `bc_planar_cap_outward_buffer` | `--bc-planar-cap-outward-buffer` | `0.25` | Outward buffer in cells |

### `bc_regrow`

Geodesic BC patch regrow (`pyvista_taubin_regrow` only).

| Option | CLI flag | Default | Description |
|--------|----------|---------|-------------|
| `bc_regrow` | `--bc-regrow` | `none` | `none` or `geodesic` |

### `subdivide`

Loop subdivision before smooth.

| Option | CLI flag | Default | Description |
|--------|----------|---------|-------------|
| `subdivide_levels` | `--subdivide-levels` | `0` | Subdivision iterations |
| `subdivide_free_only` | `--subdivide-free-only` | `False` | Subdivide only unlabeled triangles |

### `smooth`

Surface smoothing with BC vertex freeze.

| Option | CLI flag | Default | Description |
|--------|----------|---------|-------------|
| `smoother` | `--smoother` | recipe | `laplacian_bc`, `taubin_bc`, `hc_bc`, `tangential_bc`, `none` |
| `laplacian_iters` | `--laplacian-iters` | `8` | Laplacian iterations |
| `laplacian_relaxation` | `--laplacian-relaxation` | `0.5` | Laplacian ω |
| `taubin_iters` | `--taubin-iters` | `10` | Taubin / tangential iterations |
| `taubin_lambda` | `--taubin-lambda` | `0.5` | Taubin λ |
| `taubin_k_pb` | `--taubin-k-pb` | `0.1` | Taubin pass-band frequency; μ derived from λ and k_PB |
| `taubin_mu` | `--taubin-mu` | derived | Explicit signed Taubin μ (<0); overrides k_PB. `--taubin-nu` deprecated |
| `hc_iters` | `--hc-iters` | `10` | HC iterations |
| `hc_lambda` | `--hc-lambda` | `0.5` | HC λ |
| `hc_alpha` | `--hc-alpha` | `0.1` | HC α |
| `constrain_bc_planes` | `--no-constrain-bc-planes` | `True` | Re-project BC verts to planes each iter |
| `laplacian_freeze_rings` | `--laplacian-freeze-rings` | `0` | BC transition ring freeze width |
| `smooth_free_only` | `--smooth-free-only` | `False` | Smooth only unlabeled triangles |

`tangential_bc` projects Laplacian steps onto the local tangent plane (pair with `sdf_field_smooth` extract).

### `snap`

Post-smooth SDF vertex reprojection (`sdf_lewiner_snap`).

| Option | CLI flag | Default | Description |
|--------|----------|---------|-------------|
| `snap_to_sdf` | `--snap-to-sdf` | recipe | Enable SDF snap |
| `snap_to_sdf_steps` | `--snap-to-sdf-steps` | `3` | Snap iterations |

### `validate`

Final volume fraction, BC audits, free-boundary smoothness, load surface checks.

| Option | CLI flag | Default | Description |
|--------|----------|---------|-------------|
| `vf_tol_cells` | `--vf-tol-cells` | `2.0` | Soft VF gate (cells) |
| `check_loads` | `--no-load-check` | `True` | Verify NITO load anchors on surface |
| `load_surface_tol_cells` | `--load-surface-tol-cells` | `1.0` | Load distance tolerance |
| `on_load_fail` | `--on-load-fail` | `raise` | `raise` or `warn` |

---

## Global CLI flags

| Flag | Description |
|------|-------------|
| `--index N` | NITO topology index (required) |
| `--data-dir PATH` | NITO `Data/3D` root |
| `--density-cutoff` | Binarization threshold (default `0.5`) |
| `--no-bc` | Skip BC patches entirely |
| `-o PATH` / `--run-dir PATH` | Output VTP or experiment directory |
| `-q` | Quiet (summary only) |

---

## Validation matrices

| Script | Dataset ID | Purpose |
|--------|------------|---------|
| `./batch/run_validation.sh` | `compatible-pipelines-v1` | Production recipes: extract, smooth, SDF variants |
| `./batch/run_validation_planar_cap.sh` | `planar-cap-v1` | Planar cap contour methods (audit-only) |
| `./batch/run_validation_critique_methods.sh` | `critique-methods-v1` | Monolithic vs shared-seam vs field-smooth vs BC-planes SDF |

```bash
# View metrics for one index
python scripts/surface/view_validation.py --dataset critique-methods-v1 --index 119

# Open a mesh interactively
python scripts/surface/view_validation.py --dataset compatible-pipelines-v1 --index 119 --open pyvista_taubin
```

See [`batch/VALIDATION_DATASET.md`](../../../batch/VALIDATION_DATASET.md) for output layout and CSV columns.

---

## Output

Tagged VTK PolyData (`.vtp`):

- `cell_data`: `patch_id`, `facet_marker`, `fix_x/y/z`
- `FieldData`: patch metadata for FEA importers

---

## Known limitations

1. **MC stairsteps on free boundary** — PyVista binary extract produces axis-aligned facets; smoothing trades against VF and dihedral quality.
2. **VF under mesh smooth** — Laplacian is not volume-neutral (`vf_delta ≈ −0.4%` on 119); prefer `pyvista_taubin`, `sdf_lewiner_snap`, or field-smooth backends when VF matters.
3. **BC labeling is heuristic** — centroid/cell-native transfer + coplanar claim; optional `--bc-strict-footprint`.
4. **Shared-seam not production-ready** — prototype merges caps by vertex proximity; watertightness and footprint coverage regress on 119 until edge-split seam welding lands.
5. **Field-smooth VF** — `sdf_field_smooth_taubin` needs σ / calibration tuning; not yet within the 2-cell soft gate.
6. **No full feature-aware extractor** — `sdf_bc_planes` is an incremental step, not dual contouring / OpenVDB.

---

## Future work

| Priority | Item |
|----------|------|
| P0 | **Shared-seam edge-split merge** — weld cap/free interface edges, not just vertex proximity |
| P1 | **Field-smooth calibration** — default σ sweep; pair `tangential_bc` with VF gate on 119+ |
| P1 | **Feature-aware extraction** — extend `sdf_bc_planes` toward dual contouring / OpenVDB |
| P2 | **Expand validation indices** — beyond 119 for regression |
| P2 | **Remove deprecated recipe aliases** — after downstream scripts updated |

---

## Related docs

- [`PIPELINES.md`](../PIPELINES.md) — short recipe table
- [`VALIDATION.md`](VALIDATION.md) — acceptance gates and phase notes
- [`batch/VALIDATION_DATASET.md`](../../../batch/VALIDATION_DATASET.md) — dataset builder usage
