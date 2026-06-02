#!/usr/bin/env python3
"""
Report which NITO samples are 2D vs 3D (from shapes.npy).

Examples:

    python scripts/inspect_nito_dataset.py --test
    python scripts/inspect_nito_dataset.py --test --list-2d --limit 20
    python scripts/inspect_nito_dataset.py --test --save output/dataset_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from nito_io import (
    REPO_ROOT,
    indices_by_ndim,
    load_nito_arrays,
    resolve_data_dir,
    summarize_dimensions,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect 2D vs 3D NITO samples.")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--test", action="store_true", help="Use nito/Data/Test/")
    p.add_argument(
        "--list-2d",
        action="store_true",
        help="Print indices of 2D samples (after summary)",
    )
    p.add_argument(
        "--list-3d",
        action="store_true",
        help="Print indices of 3D samples (after summary)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Max indices to print per --list-* (0 = all)",
    )
    p.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Write full summary JSON (default: output/dataset_summary.json with --save alone)",
    )
    return p.parse_args()


def _print_indices(label: str, indices: list[int], limit: int) -> None:
    if not indices:
        print(f"  {label}: (none)")
        return
    if limit <= 0 or len(indices) <= limit:
        shown = indices
        suffix = ""
    else:
        shown = indices[:limit]
        suffix = f" ... +{len(indices) - limit} more"
    print(f"  {label} ({len(indices)} total): {shown}{suffix}")


def main() -> None:
    args = parse_args()
    data_dir = resolve_data_dir(args.data_dir, test=args.test)
    split = "test" if args.test else "train"
    shapes = load_nito_arrays(data_dir)["shapes"]
    summary = summarize_dimensions(shapes)
    by_dim = indices_by_ndim(shapes)

    print(f"Data: {data_dir}  ({split} split)")
    print(f"Samples: {summary['n_samples']}")
    print("By dimension:")
    for d in sorted(summary["counts_by_ndim"]):
        print(f"  {d}D: {summary['counts_by_ndim'][d]}")

    for key, label in (("dim_2d", "2D"), ("dim_3d", "3D")):
        stats = summary[key]
        if stats["count"] == 0:
            continue
        print(f"\n{label} element counts: {stats['n_elements_min']} .. {stats['n_elements_max']}")
        print(
            f"  smallest: index {stats['smallest_index']}  shape {stats['smallest_shape']}"
        )
        print(
            f"  largest:  index {stats['largest_index']}  shape {stats['largest_shape']}"
        )

    for d, label in ((2, "2D"), (3, "3D")):
        templates = summary["unique_shapes"].get(d, [])
        if not templates:
            continue
        print(f"\n{label} unique grid templates ({len(templates)}):")
        for t in templates[:10]:
            print(
                f"  {t['shape']}: {t['count']} samples  (e.g. index {t['example_index']})"
            )
        if len(templates) > 10:
            print(f"  ... +{len(templates) - 10} more templates")

    if args.list_2d:
        print()
        _print_indices("2D indices", summary["indices_2d"], args.limit)
    if args.list_3d:
        print()
        _print_indices("3D indices", summary["indices_3d"], args.limit)

    if args.save is not None:
        out = args.save
        if out == Path("-"):
            print(json.dumps(summary, indent=2))
        else:
            out = Path(out)
            if out.is_dir() or str(out).endswith("/"):
                out = out / f"dataset_summary_{split}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
