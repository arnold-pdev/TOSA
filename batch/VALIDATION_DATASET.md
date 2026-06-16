# Validation dataset

Pre-defined matrix of **compatible** voxel2surf recipes on shared probe indices.
Use this to compare BC enforcement, volume, and **free-boundary stairstepping** offline.

## Build the dataset

Requires `nito/Data/3D` (or `--test` for Test split).

```bash
# One-shot: run full matrix, log everything, print metrics table
bash batch/run_validation.sh
```

Log: `output/surfaces/validation/compatible-pipelines-v1/build.log`

Or step-by-step:

```bash
# Preview jobs (8 runs = index 119 × 8 recipes)
python scripts/surface/build_validation_dataset.py --dry-run

# Full build (~few minutes depending on hardware)
python scripts/surface/build_validation_dataset.py \
  --dataset-id compatible-pipelines-v1 \
  --on-load-fail warn --on-bc-fail warn
```

Output layout:

```text
output/surfaces/validation/compatible-pipelines-v1/
  manifest.json
  VIEW.md                 # per-mesh visualize.py commands
  metrics/summary.csv     # core QA metrics + timing / planar-cap columns
  vtp/<index>/<recipe>.vtp
  logs/<index>_<recipe>.log
```

## Recipes in the matrix

| CLI | Role |
|-----|------|
| `extract_only` | Stairstep baseline (no smooth) |
| `v3_default` | Laplacian + BC pin |
| `v3_taubin` | Primary candidate |
| `v3_regrow` | BC regrow + Taubin |
| `v3_hc` | HC smooth |
| `sdf_vf_match` | SDF + VF calibrate |
| `sdf_masked_taubin` | BC-masked SDF |

Edit `batch/validation_matrix.json` to add recipes or `batch/validation_indices.txt` for more indices.

## Stairstepping / smoothness

`metrics/summary.csv` includes **core QA columns** plus optional **planar cap audit** columns when present:

| Column | Meaning |
|--------|---------|
| `vf_delta` | Mesh VF − voxel VF |
| `bc_plane_max_residual` | BC plane snap residual |
| `bc_footprint_coverage` | Labeled BC tris vs voxel footprint |
| `bc_labeled_triangles` | Total BC-labeled triangles |
| `bc_min_patch_tris` | Smallest per-patch labeled count |
| `free_dihedral_mean_deg` / `free_dihedral_p95_deg` | Free-boundary crease angles |
| `axis_aligned_edge_frac` | Axis-aligned free-free edge fraction |
| `vertices` / `faces` | Final mesh size |
| `construction_sec` / `timing_*_sec` | Wall-clock / per-stage timing |
| `bc_planar_cap_method` | Planar cap contour method (audit-only runs) |
| `planar_cap_watertight` / `planar_cap_bodies` | Cap mesh topology (not skin mesh) |

Re-parse logs without re-running pipelines:

```bash
python scripts/surface/build_validation_dataset.py \
  --matrix batch/validation_planar_cap_matrix.json \
  --dataset-id planar-cap-v1 \
  --refresh-metrics
```

Compare each recipe to the extract baseline on the same index.

## View later

**Metric table** for one index:

```bash
python scripts/surface/view_validation.py \
  --dataset compatible-pipelines-v1 --index 119
```

**Interactive PyVista** (voxels + BC colors):

```bash
python scripts/surface/view_validation.py \
  --dataset compatible-pipelines-v1 --index 119 --open pyvista_taubin
```

Or open `output/surfaces/validation/compatible-pipelines-v1/VIEW.md` for all commands.

**Side-by-side workflow:** run `--open extract_baseline`, then `--open pyvista_taubin`, on index **119**.

## Gates (from matrix)

Hard: `bc_plane_max_residual < 1e-5`, `bc_footprint_coverage ≥ 0.9`, loads pass.

Soft: `|vf_delta| ≤ 2 cells`, smoothness better than `extract_baseline` baseline.
