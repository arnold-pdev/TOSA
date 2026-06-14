# voxel2surf validation matrix

How to compare extractors, smoothers, and BC enforcement before promoting a recipe to `public/vtp/` or batch (see [PIPELINES.md](../PIPELINES.md)).

## Acceptance gates

Score every run in `metrics/summary.csv` and logs against these gates.

| Gate | Priority | Criterion |
|------|----------|-----------|
| BC plane residual | **Hard** | `bc_plane_max_residual < 1e-5` |
| BC footprint | **Hard** | `bc_footprint_coverage ≥ 0.90` (warn 0.80–0.90) |
| Load on surface | **Hard** (if loads) | all checks ≤ `load_surface_tol_cells` |
| Volume | **Soft** | `\|vf_mesh − vf_voxel\| ≤ vf_tol_cells` (default 2 cells, `--vf-tol-cells`) |
| Watertight | **Hard** | `watertight=True` in stage reports |
| Visual | **Manual** | no visible stairsteps on free boundary; BC faces flat |

### Gate checklist (per index)

- [ ] `bc_plane_max_residual < 1e-5`
- [ ] `bc_footprint_coverage ≥ 0.90`
- [ ] Load checks pass (or documented warn with `--on-load-fail warn`)
- [ ] `|vf_delta| ≤ vf_tol_cells` in cell-equivalent units
- [ ] Final stage `watertight=True`
- [ ] Visual review: free boundary smooth, BC planes flat

---

## Probe set

Expand `batch/probe_indices.txt` beyond the core regression pair.

| Tier | Purpose | Indices |
|------|---------|---------|
| **Core** | Always run | 10, 119 |
| **BC stress** | Multi-patch, corner load, weak transfer | 119 + TBD from footprint failures |
| **VF / thin members** | Volume and slender features | TBD via `inspect_dataset` |
| **2D sanity** | Fast iteration | one Test-split index (optional) |

### Probe set checklist

- [ ] Core indices (10, 119) in `batch/probe_indices.txt`
- [ ] At least 2 BC-stress indices identified and added
- [ ] At least 2 VF/thin-member indices identified and added
- [ ] Optional: `probe_manifest.csv` with columns `index,tag,notes`

---

## Phase 1 — implemented recipes (run now)

Use `scripts/surface/benchmark.py` or manual runs with `--run-dir`.

```bash
python scripts/surface/benchmark.py \
  --recipe <name> \
  --run-dir output/surfaces/runs/matrix_<name>
```

### A. Extract × smooth (primary)

| Run ID | `--recipe` | Extra flags | Question |
|--------|------------|-------------|----------|
| A1 | `extract_only` | — | Baseline stairstep + BC enforce only |
| A2 | `v3_default` | default (Lap 8) | Current production default |
| A3 | `v3_taubin` | default (10 iters) | VF + smoothness vs A2 |
| A4 | `sdf_vf_match` | `--calibrate-vf` | Field VF before mesh vs A3 |
| A5 | `v3_taubin` | `--taubin-iters 5` | Light smooth |
| A6 | `v3_taubin` | `--taubin-iters 15` | Heavier smooth (stairs vs VF) |

#### Phase 1A checklist

- [ ] A1 `extract_only` on core probes
- [ ] A2 `v3_default` on core probes
- [ ] A3 `v3_taubin` on core probes
- [ ] A4 `sdf_vf_match --calibrate-vf` on core probes
- [ ] A5 `v3_taubin --taubin-iters 5` on core probes
- [ ] A6 `v3_taubin --taubin-iters 15` on core probes
- [ ] Compare `metrics/summary.csv` across runs
- [ ] Visual compare on 10 and 119 (`scripts/surface/visualize.py`)

### B. BC enforcement (on best extract+smooth from A)

| Run ID | Flags | Question |
|--------|-------|----------|
| B1 | default | baseline |
| B2 | `--bc-strict-footprint` | drop off-footprint labels — help or hurt? |
| B3 | `--no-constrain-bc-planes` | value of per-iter plane pin during smooth |

#### Phase 1B checklist

- [ ] B1 default BC enforce
- [ ] B2 `--bc-strict-footprint`
- [ ] B3 `--no-constrain-bc-planes` (expect load/BC regressions)

### C. Subdivide (optional)

| Run ID | Flags | Question |
|--------|-------|----------|
| C1 | `--subdivide-levels 1` + best recipe | smoother skin at ~4× faces — worth cost? |

#### Phase 1C checklist

- [ ] C1 subdivide + chosen default recipe on core probes

### Phase 1 decision

- [ ] Pick default recipe by Pareto on core probes (VF, BC gates, visual)
- [ ] Document winner in `manifest.json` / team notes
- [ ] Promote winning `vtp/` → `public/vtp/` only after gates pass

**Current candidate:** `v3_taubin` (better VF on index 10 vs Laplacian; BC residual 0 on 119 in smoke runs).

---

## Phase 2 — port from archives / wire existing code

Not in the staged package yet; add recipes or stages before testing.

| Method | Effort | Recipe to add | Compare vs |
|--------|--------|---------------|------------|
| SDF Lewiner, no VF cal | Low | `sdf_raw_taubin` | A3 |
| SDF + σ smooth | Low | `sdf_vf_match --field-smooth-sigma 0.35` | A4 |
| v2 upsample 2× | Medium | `sdf_upsample2_vf` | A4 |
| HC-Laplacian | Low | `hc_bc` smoother | A3 |
| Laplacian then Taubin | Low | two-pass recipe | A3 |
| v1 geodesic regrow | Medium | `bc_regrow` stage | BC footprint on 119 |

#### Phase 2 checklist

- [ ] `sdf_raw_taubin` recipe + matrix run
- [ ] SDF Gaussian σ sweep (0, 0.35)
- [ ] Upsample-2 extractor wired from `lib/meshing/field.py`
- [ ] `hc_bc` smoother implemented
- [ ] Optional `bc_regrow` stage from `patch_regrow.py`
- [ ] Archives kept as regression references only (not batch default)

---

## Phase 3 — not implemented; build if Phase 1–2 fail gates

| Method | Addresses | Priority |
|--------|-----------|----------|
| SDF snap after smooth | VF without global rescale | **P1** |
| SDF Lewiner + Laplacian recipe | missing combo | **P1** |
| Cuberille BC oracle (QA) | exact BC planes / footprint reference | **P1** |
| Improved BC transfer / surface nets | exact cell↔label map | **P2** |
| Hybrid cuberille BC + iso free skin | exact BC + smooth exterior | **P3** |
| BC-aware vf_rescale | volume if snap insufficient | **P3** (risky) |
| Pressing | interior flats on design surface | **Defer** |
| Dual contouring / manifold DC | sharp organic features | **Defer** |
| fTetWild downstream | FEM mesh robustness | **Parallel** (not surface QA) |

#### Phase 3 checklist

- [ ] SDF reprojection pass after smooth
- [ ] `sdf_laplacian` recipe
- [ ] Cuberille BC overlay vs mesh BC audit tool
- [ ] Surface-nets or cell-native transfer prototype
- [ ] fTetWild tet mesh smoke on promoted surfaces

---

## What is lacking today

### Infrastructure

- [ ] Run comparator (`compare_runs.py` — diff two `metrics/summary.csv`)
- [ ] Expanded probe set (8+ tagged indices)
- [ ] `sdf_vf_match` auto-enables `--calibrate-vf`
- [ ] Automated visualize screenshots in benchmark
- [ ] Per-patch triangle counts in metrics CSV (e.g. patch with 0 tris)

### Algorithms vs goals

| Goal | Status |
|------|--------|
| BC on prescribed plane | **Good** — `bc_enforce` + pin during smooth |
| BC region inside contour | **Partial** — approximate transfer; `--bc-strict-footprint` optional |
| VF within ~1–2 cells | **Partial** — Taubin better; SDF cal not default |
| No visible stairsteps | **Partial** — smooth helps; binary extract limits quality |
| Shape-derivative smoothness | **Not measured** — no normal/curvature metric in `validate` |
| Exact BC label map | **Lacking** — edge-based MC/PyVista transfer |

### Extractors

- [x] `pyvista_binary`
- [x] `sdf_lewiner` (+ optional VF cal, σ)
- [ ] surface nets
- [ ] cuberille (QA / hybrid)
- [ ] upsampled field (`field.upsample_field`)

### Smoothers

- [x] `laplacian_bc`
- [x] `taubin_bc`
- [x] `none`
- [ ] `hc_bc`
- [ ] SDF reprojection pass
- [ ] v1 regrow as optional stage

---

## Minimal matrix (48 jobs)

Six runs × eight probe indices — enough to choose a default.

1. `extract_only`
2. `v3_default`
3. `v3_taubin`
4. `sdf_vf_match --calibrate-vf`
5. `v3_taubin --taubin-iters 15`
6. `v3_taubin --bc-strict-footprint`

### Minimal matrix checklist

- [ ] All 6 runs completed on expanded probe set
- [ ] All hard gates pass on chosen default
- [ ] Visual sign-off on 10 and 119
- [ ] Default recipe recorded in PIPELINES.md / batch sbatch

---

## Suggested execution order

1. **Week 1** — Phase 1 matrix on expanded probe set → pick `v3_taubin` vs `v3_default` vs `sdf_vf_match+calibrate`
2. **Week 2** — P1 infra: auto-calibrate SDF recipe, richer metrics CSV, `compare_runs.py`
3. **Week 3** — If footprint coverage fails: surface-nets or improved transfer; cuberille BC oracle for QA

---

## Quick commands

```bash
# List recipes
python scripts/lib/converters/voxel2surf.py --list-recipes

# Single experiment run
python scripts/lib/converters/voxel2surf.py --index 119 \
  --recipe v3_taubin \
  --run-dir output/surfaces/runs/matrix_v3_taubin

# Probe batch
python scripts/surface/benchmark.py \
  --recipe v3_taubin \
  --run-dir output/surfaces/runs/probe_v3_taubin \
  --on-load-fail warn

# Visual QA
python scripts/surface/visualize.py --index 119 \
  --scalar boundary --overlay-voxels --data-dir nito/Data/3D
```
