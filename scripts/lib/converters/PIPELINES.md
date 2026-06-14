# voxel2surf pipelines

Active entry point: `scripts/lib/converters/voxel2surf.py` (staged package under `voxel2surf/`).

## Recipes

| CLI `--recipe` | ID | Extract | Smooth |
|----------------|-----|---------|--------|
| `v3_default` | v3-pyvista-laplacian | PyVista binary threshold | Laplacian + BC pin |
| `v3_taubin` | v3-pyvista-taubin | PyVista binary threshold | Taubin λ\|μ + BC pin |
| `sdf_vf_match` | sdf-lewiner-taubin | SDF + Lewiner MC (+ optional VF calibrate) | Taubin + BC pin |
| `extract_only` | extract-only | PyVista binary threshold | none |

List recipes: `python scripts/lib/converters/voxel2surf.py --list-recipes`

## Stage order

```
extract → bc_build → bc_transfer → bc_enforce → [subdivide] → [smooth] → bc_enforce → validate
```

`bc_enforce` snaps all BC-labeled triangles onto NITO patch planes and records footprint coverage.

## Experiment runs

```bash
python scripts/lib/converters/voxel2surf.py --index 119 \
  --recipe v3_taubin \
  --run-dir output/surfaces/runs/2025-06-03_v3-taubin
```

Writes:

- `vtp/<index>.vtp`
- `logs/<index>.log`
- `metrics/summary.csv`
- `manifest.json`

Promote to batch output: copy `vtp/` → `public/vtp/` when probe metrics pass.

## Archives

Frozen references (not maintained):

- `archives/voxel2surf_v1_archive.py` — SDF + Lewiner MC + Taubin + geodesic regrow
- `archives/voxel2surf_v2_archive.py` — field upsample + VF calibrate + Laplacian

## Probe batch

```bash
python scripts/surface/benchmark.py --recipe v3_default --run-dir output/surfaces/runs/probe_v3
```

Uses `batch/probe_indices.txt` (10, 119). See [voxel2surf/VALIDATION.md](voxel2surf/VALIDATION.md) for the full test matrix and checklists.
