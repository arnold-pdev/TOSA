# NITO-3D Data & TOSA Setup Guide

This document summarizes how the TOSA sprint project relates to the **NITO-3D** dataset and what to do next. **TOSA** is our project name; the upstream work is **NITO-3D** (Neural Implicit Topology Optimization for 3D).

## Source project

| Item | Link / note |
|------|-------------|
| **Consolidated repo** | [ahnobari/NITO_Public](https://github.com/ahnobari/NITO_Public) |
| **Legacy 3D repo** | [Lyleregenwetter/NITO-3D](https://github.com/Lyleregenwetter/NITO-3D) — redirects to NITO_Public; do not rely on it |
| **Paper** | [NITO-3D PDF](https://decode.mit.edu/assets/papers/nobari_2024_nito3d.pdf) |
| **Manual download** | [Data & checkpoints (Drive)](https://drive.google.com/drive/folders/1_wKPq8HXjaoRa4oCy_tvLOopIcapk7wO?usp=sharing), [3D data folder](https://drive.google.com/drive/folders/1uK_X3-FcCWY9LiiXkVQDI69q0t6Vosgm) |

Clone **NITO_Public** for `download.py` and reference code. Data is **not** in git; it comes from Google Drive.

## Dataset scale (122K public outputs)

From the NITO-3D paper (Table 1) and measured Drive file sizes:

| Metric | Value |
|--------|--------|
| Topologies (headline) | **~122,000** |
| Total voxel elements | **4.3 billion** |
| Voxels per topology (min / max) | **32,000 / 48,000** (e.g. 40×40×20, 120×40×10) |
| Held-out test set | **2,000** samples |
| Unique BC templates | **210** configurations, **7** domain shapes/resolutions |

**On-disk size (train + test, all five `.npy` files each):**

| Bundle | Location | Size |
|--------|----------|------|
| Train | `Data/*.npy` | **~4.64 GB** (`topologies.npy` alone ~4.02 GB) |
| Test | `Data/Test/*.npy` | **~85 MB** |
| **Total** | | **~4.72 GB** |

After download, confirm sample count:

```python
import numpy as np
print(len(np.load("shapes.npy")))  # expect ~122K for train
```

**RAM note:** Loading the full train set with `np.load(..., allow_pickle=True)` can use noticeably more than 5 GB RAM due to pickle/object-array overhead. Prefer the **test split** for early exploration.

## Files and layout

Scripts in this folder expect these paths **relative to `database/`** (working directory must be `database/`):

```
database/
  shapes.npy
  boundary_conditions.npy
  loads.npy
  topologies.npy
  vfs.npy              # used by nates_reading.py only
  authors_reading.ipynb
  nates_reading.py
```

For NITO_Public’s layout, the same files live under `Data/` (train) and `Data/Test/` (test). Copy or symlink into `database/` or point download output here.

### Per-sample schema (index `i`)

All arrays are aligned by integer index `i`:

| Array | Content |
|-------|---------|
| `shapes[i]` | Grid size `(nelx, nely, nelz)` |
| `topologies[i]` | Flattened density field → `reshape(shapes[i], order='C')` |
| `boundary_conditions[i]` | Cols `0:3` normalized position; `3:` constraint flags |
| `loads[i]` | Cols `0:3` position; `3:` force direction |
| `vfs[i]` | Target volume fraction |

**Coordinate scaling:** BC/load positions in the files are normalized. Scale by `np.max(shapes[i])` when plotting (see comments in `nates_reading.py` / notebook).

## Download

From a clone of NITO_Public (needs `gdown`):

```bash
conda activate tosa   # or your env
python download.py --data --data_dir /path/to/TOSA/database
```

This downloads **train + test** (10 files). There is no official per-sample download API.

**Train-only or test-only:** edit `download.py` to run only `data_ids` or `test_data_ids`, or download individual files manually from Drive using the IDs in `download.py`.

**Checkpoints** (optional, for running their models): `python download.py --checkpoints` — not needed for visualization-only TOSA work.

## Python environment

Recommended for both notebooks/scripts (Python **3.10** matches the notebook kernel that was used):

```text
numpy
matplotlib
scikit-image    # authors_reading.ipynb (marching cubes)
pyvista         # nates_reading.py (interactive 3D)
jupyter
ipykernel
gdown           # for download.py
```

**Skip unless needed:** `tensorflow` (imported in the notebook but unused), full NITO_Public `environment.yml` (PyTorch/CUDA stack) unless training NITO models.

Example conda env:

```bash
conda create -n tosa python=3.10 -y
conda activate tosa
conda install -c conda-forge numpy matplotlib scikit-image pyvista jupyter ipykernel gdown -y
python -m ipykernel install --user --name tosa --display-name "tosa"
```

## Two visualization paths in this repo

| Script | Stack | Notes |
|--------|--------|------|
| `authors_reading.ipynb` | Matplotlib + scikit-image marching cubes | Matches authors’ plotting style; inline figures |
| `nates_reading.py` | PyVista | Interactive windows; loops `i in range(0, 20)` — close each window to continue |

## Pull-only-what-you-need workflow

NITO supports **index-based** use after files exist, not streaming single samples from the cloud.

| Level | What’s possible |
|-------|------------------|
| **Remote** | Whole files only (5 train + 5 test) |
| **Local index** | `i`, `ID_IDX`, `--start_index` / `--end_index` in NITO `train.py` / `evaluate.py` |
| **Memory** | Official scripts still `np.load()` entire `.npy` files before slicing indices |

**Practical TOSA approach:**

1. **Sprint start:** download only `Data/Test/*.npy` (~85 MB, ~2K samples).
2. **Work by index:** e.g. `ID_IDX = 10` in the notebook, or `topologies[i]` in the script.
3. **Full train later:** download train bundle (~4.6 GB) when sensitivity/training needs it.
4. **Optional cache layer:** extract chosen indices to `database/cache/{i}/` so TOSA code does not load 122K samples every run (not provided by NITO; add locally if needed).

`shapes.npy` and `vfs.npy` are small — useful to download first for scouting indices before pulling `topologies.npy`.

## NITO reference: training index range

If using NITO_Public training/eval later:

```bash
python train.py --data ./Data --start_index 0 --end_index 1000
```

This limits which indices are processed but **does not** avoid loading full `.npy` files into memory.

## Suggested next steps

1. **Install Python 3.10+** (Windows Store `python` stub is not sufficient).
2. **Create `tosa` conda env** with packages above.
3. **Clone NITO_Public** beside TOSA (optional but useful for `download.py` and format reference).
4. **Download test data** into `database/` (~85 MB) and run `authors_reading.ipynb` or `nates_reading.py`.
5. **Verify data:** `len(shapes)`, spot-check `shapes[i]` and `topologies[i].reshape(shapes[i])`.
6. **Document chosen indices** for sensitivity experiments (fixed set for reproducibility).
7. **Download full train bundle** when you need scale beyond the 2K test set.
8. **(Optional)** Add `inspect_data.py` or a cache helper for index subsets — not in repo yet.

## Open questions / gaps

- Paper mentions both **106,425** and **122K** topology counts in different sections; treat **122K** as the public release scale and verify with `len(shapes)` after download.
- No per-sample IDs or metadata file in the release — derive filters from `shapes[i]`, `vfs[i]`, and BC/load patterns if needed.
- **Hongrui’s clarification:** multiply normalized BC/load positions by `np.max(dims)` when plotting.

## Related files

- `authors_reading.ipynb` — matplotlib / marching-cubes viewer
- `nates_reading.py` — PyVista viewer
- Upstream: `NITO_Public/download.py`, `NITO/utils.py` (`NITO_Dataset.load(idx)`)
