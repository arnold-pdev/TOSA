# voxel2surf pipelines

Active entry point: `scripts/lib/converters/voxel2surf.py` (staged package under `voxel2surf/`).

## Recipes

| CLI `--recipe` | ID | Extract | Smooth |
|----------------|-----|---------|--------|
| `pyvista_laplacian` | pyvista-laplacian | PyVista binary | Laplacian + BC pin |
| `pyvista_taubin` | pyvista-taubin | PyVista binary | Taubin + BC pin |
| `pyvista_taubin_regrow` | pyvista-taubin-regrow | PyVista binary | Regrow + Taubin |
| `pyvista_hc` | pyvista-hc | PyVista binary | HC + BC pin |
| `sdf_lewiner_calibrated` | sdf-lewiner-calibrated-taubin | SDF + VF cal | Taubin |
| `sdf_masked_taubin` | sdf-masked-taubin | BC-masked SDF | Taubin |
| `sdf_lewiner_raw` | sdf-lewiner-raw-taubin | SDF Lewiner | Taubin |
| `sdf_lewiner_snap` | sdf-lewiner-snap-taubin | SDF + snap | Taubin |
| `extract_baseline` | extract-no-smooth | PyVista binary | none |
| `planar_cap_contour` | planar-cap-contour | PyVista binary | none (+ cap audit) |
| `pyvista_shared_seam_taubin` | pyvista-shared-seam-taubin | PyVista binary | Taubin + shared-seam assembly |
| `sdf_field_smooth_taubin` | sdf-field-smooth-taubin | SDF field smooth | tangential_bc |
| `sdf_bc_planes_taubin` | sdf-bc-planes-taubin | SDF BC-plane pin | Taubin |

List recipes: `python scripts/lib/converters/voxel2surf.py --list-recipes`

Deprecated aliases (`v3_default`, `v3_taubin`, `extract_only`, …) still resolve with a warning.

## Stage order

```
extract → bc_build → bc_transfer → bc_enforce → [bc_planar_cap] → [subdivide] → [smooth] → bc_enforce → validate
```

`bc_planar_cap` is optional (`--bc-planar-cap` or `planar_cap_contour` recipe). It contours footprint masks for QA; caps are not assembled onto the skin.

## Stage modules

| Module | Stages |
|--------|--------|
| `extract.py` | `extract` |
| `bc.py` | `bc_build`, `bc_transfer`, `bc_enforce`, `bc_planar_cap` |
| `bc_assembly.py` | `bc_assembly` (`monolithic`, `shared_seam`) |
| `refine.py` | `subdivide`, `smooth`, `snap`, `bc_regrow` |
| `validate.py` | `validate` |

## Experiment runs

```bash
python scripts/lib/converters/voxel2surf.py --index 119 \
  --recipe pyvista_taubin \
  --run-dir output/surfaces/runs/pyvista-taubin-119
```

Full stage/options reference and future work: [voxel2surf/README.md](voxel2surf/README.md).
