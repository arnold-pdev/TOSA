#!/usr/bin/env python3
"""
Run pre-trained NITO inference on dataset samples and save density predictions.

Outputs continuous ρ ∈ (0, 1) (sigmoid of the field network), same layout as
topologies.npy elements. Feed into sensitivity analysis via main.py --rho-file.

Requires PyTorch (included in environment.yml). Checkpoints from:

    ./scripts/fetch/checkpoints.sh              # 64x64 (default)
    ./scripts/fetch/checkpoints.sh --256x256    # 256×256 test indices

Examples:

    python scripts/voxel/predict.py --index 0 --test
    python scripts/voxel/predict.py --index 0 --test --checkpoint nito/Checkpoints/64x64/checkpoint_epoch_50.pth
    python scripts/voxel/sensitivity/compliance.py --index 0 --test \\
        --rho-file output/nito_predictions/0/rho_pred.npy \\
        --output-dir output/atoms_results/nito_pred/0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from lib.paths import REPO_ROOT, ensure_nito_on_path

ensure_nito_on_path()

NITO_ROOT = REPO_ROOT / "nito"

from lib.nito_io import load_nito_arrays, load_sample, resolve_data_dir  # noqa: E402

logger = logging.getLogger(__name__)

CHECKPOINT_PRESETS: dict[str, Path] = {
    "64x64": NITO_ROOT / "Checkpoints" / "64x64" / "checkpoint_epoch_50.pth",
    "256x256": NITO_ROOT / "Checkpoints" / "256x256" / "checkpoint_epoch_50.pth",
    "64x64_256x256": NITO_ROOT
    / "Checkpoints"
    / "64x64_256x256"
    / "checkpoint_epoch_50.pth",
    "All": NITO_ROOT / "Checkpoints" / "All" / "checkpoint_epoch_50.pth",
}

# Must match nito/run_train_experiments.sh (not evaluate.py argparse defaults).
NITO_ARCH = {
    "BC_n_layers": 4,
    "BC_hidden_size": 256,
    "BC_emb_size": 80,
    "C_n_layers": 4,
    "C_hidden_size": 256,
    "C_mapping_size": 256,
    "Field_n_layers": 8,
    "Field_hidden_size": 1024,
    "Fourier_size": 512,
    "omega": 1.0,
    "freq_scale": 10.0,
}


def suggest_checkpoint_preset(shape: np.ndarray) -> str:
    """Heuristic matching NITO Drive checkpoint bundles."""
    shape = np.asarray(shape, dtype=int).ravel()
    m = int(shape.max())
    if m <= 64:
        return "64x64"
    if m <= 256:
        return "256x256"
    return "All"


def resolve_checkpoint(path: Path | None, preset: str | None, shape: np.ndarray) -> Path:
    if path is not None:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    key = preset or suggest_checkpoint_preset(shape)
    if key not in CHECKPOINT_PRESETS:
        raise ValueError(f"Unknown preset {key!r}; choose from {list(CHECKPOINT_PRESETS)}")
    ckpt = CHECKPOINT_PRESETS[key]
    if not ckpt.is_file():
        flag = "--" + key.replace("_", "-")
        raise FileNotFoundError(
            f"Checkpoint missing: {ckpt}\n"
            f"Run: ./scripts/fetch/checkpoints.sh {flag}"
        )
    return ckpt


def build_model_and_dataset(
    data: dict[str, np.ndarray],
    *,
    shape_normalize: bool = True,
    field_hidden_size: int = NITO_ARCH["Field_hidden_size"],
    bc_emb_size: int = NITO_ARCH["BC_emb_size"],
):
    import torch
    from NITO.model import NITO
    from NITO.utils import NITO_Dataset

    topologies = data["topologies"]
    shapes = data["shapes"]
    loads = data["loads"]
    vfs = data["vfs"]
    bcs = data["boundary_conditions"]
    dim = int(np.asarray(shapes[0]).size)

    if shape_normalize:
        dataset = NITO_Dataset(
            topologies,
            [bcs, loads],
            [vfs, shapes / shapes.max(1, keepdims=True)],
            shapes,
            n_samples=10,
            consistent_batch=False,
        )
    else:
        dataset = NITO_Dataset(
            topologies,
            [bcs, loads],
            [vfs, shapes],
            shapes,
            n_samples=10,
            consistent_batch=False,
        )

    bc_shape = [dim * 2, dim * 2]
    c_shape = [1, dim]
    model = NITO(
        BCs=bc_shape,
        BC_n_layers=[NITO_ARCH["BC_n_layers"]] * len(bc_shape),
        BC_hidden_size=[NITO_ARCH["BC_hidden_size"]] * len(bc_shape),
        BC_emb_size=[bc_emb_size] * len(bc_shape),
        Cs=c_shape,
        C_n_layers=[NITO_ARCH["C_n_layers"]] * len(c_shape),
        C_hidden_size=[NITO_ARCH["C_hidden_size"]] * len(c_shape),
        C_mapping_size=[NITO_ARCH["C_mapping_size"]] * len(c_shape),
        Field_n_layers=NITO_ARCH["Field_n_layers"],
        Field_hidden_size=field_hidden_size,
        Fourier_size=NITO_ARCH["Fourier_size"],
        omega=NITO_ARCH["omega"],
        freq_scale=NITO_ARCH["freq_scale"],
        input_channels=dim,
        output_channels=1,
    )
    return torch, model, dataset, dim


def predict_index(
    model,
    dataset,
    torch,
    index: int,
    *,
    device: str,
) -> np.ndarray:
    """Return flattened ρ_pred of length shape.prod() for one sample."""
    model.eval()
    with torch.no_grad():
        inputs, _ = dataset.batch_load([index], device=device, mode="test_no_pad")
        pred = torch.sigmoid(model(inputs)).reshape(-1)
    return pred.detach().cpu().numpy()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NITO pre-trained inference → rho_pred.npy")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--test", action="store_true")
    p.add_argument("--index", type=int, default=0)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to checkpoint_epoch_50.pth",
    )
    p.add_argument(
        "--checkpoint-preset",
        type=str,
        default=None,
        choices=tuple(CHECKPOINT_PRESETS),
        help="Auto path under nito/Checkpoints/ (default: from shape)",
    )
    p.add_argument(
        "--no-shape-normalize",
        action="store_true",
        help="Disable shape normalization (off by default; checkpoints use --shape_normalize)",
    )
    p.add_argument(
        "--field-hidden-size",
        type=int,
        default=NITO_ARCH["Field_hidden_size"],
        help="Field MLP width (default 1024 per published checkpoints)",
    )
    p.add_argument(
        "--bc-emb-size",
        type=int,
        default=NITO_ARCH["BC_emb_size"],
        help="BC encoder embedding size (default 80 per published checkpoints)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: output/nito_predictions/<index>/",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Also save rho_pred_binary.npy at this threshold",
    )
    p.add_argument("--gpu", type=int, default=0, help="CUDA device index (ignored on CPU)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    try:
        import torch
    except ImportError as e:
        raise SystemExit(
            "PyTorch is required for NITO inference. Install with:\n"
            "  micromamba env create -f environment.yml && micromamba activate tosa\n"
            "or use the upstream NITO conda env."
        ) from e

    data_dir = resolve_data_dir(args.data_dir, test=args.test)
    data = load_nito_arrays(data_dir)
    n = len(data["shapes"])
    if not (0 <= args.index < n):
        raise IndexError(f"index {args.index} out of range [0, {n})")

    shape, _, bc, load, vf = load_sample(data, args.index)
    ndim = int(np.asarray(shape).size)
    if ndim != 2:
        raise ValueError(
            f"Sample {args.index} is {ndim}D (shape={shape}). "
            "Use a 2D test index for the 2D NITO checkpoints."
        )

    ckpt = resolve_checkpoint(args.checkpoint, args.checkpoint_preset, shape)
    logger.info("Checkpoint: %s", ckpt)
    logger.info("Sample %d: shape=%s, vf=%.4f", args.index, shape.tolist(), vf)

    torch_mod, model, dataset, _ = build_model_and_dataset(
        data,
        shape_normalize=not args.no_shape_normalize,
        field_hidden_size=args.field_hidden_size,
        bc_emb_size=args.bc_emb_size,
    )
    device = f"cuda:{args.gpu}" if torch_mod.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    checkpoint = torch_mod.load(ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    rho_pred = predict_index(model, dataset, torch_mod, args.index, device=device)
    expected = int(np.prod(shape))
    if rho_pred.size != expected:
        raise RuntimeError(f"Expected {expected} elements, got {rho_pred.size}")

    out_dir = args.output_dir or REPO_ROOT / "output" / "nito_predictions" / str(args.index)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "rho_pred.npy", rho_pred.astype(np.float64))
    np.save(out_dir / "rho_pred_binary.npy", (rho_pred >= args.threshold).astype(np.float64))
    np.save(out_dir / "shape.npy", np.asarray(shape, dtype=int))
    meta = {
        "index": args.index,
        "checkpoint": str(ckpt),
        "shape": shape.tolist(),
        "vf_target": vf,
        "vf_pred": float(rho_pred.mean()),
        "vf_pred_binary": float((rho_pred >= args.threshold).mean()),
        "shape_normalize": not args.no_shape_normalize,
        "field_hidden_size": args.field_hidden_size,
        "bc_emb_size": args.bc_emb_size,
    }
    np.save(out_dir / "meta.npy", meta, allow_pickle=True)
    logger.info(
        "Wrote %s (vf_pred=%.4f, vf_binary=%.4f)",
        out_dir,
        meta["vf_pred"],
        meta["vf_pred_binary"],
    )
    print(f"rho_pred: {out_dir / 'rho_pred.npy'}")
    print(f"Next: python scripts/voxel/sensitivity/compliance.py --index {args.index} "
          f"{'--test ' if args.test else ''}--rho-file {out_dir / 'rho_pred.npy'} "
          f"--output-dir output/atoms_results/nito_pred/{args.index}")


if __name__ == "__main__":
    main()
