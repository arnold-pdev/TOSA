#!/usr/bin/env python3
"""
Surface shape-derivative sensitivity analysis (.vtp distribution).

Planned workflow: shape derivatives of objective functions on surface meshes,
scalar fields saved on VTK PolyData (.vtp), PyVista visualization.

For voxel density-based compliance (∂C/∂ρ on grids), use:

    ./scripts/voxel/sensitivity/compliance.sh --index 0 --test
    python scripts/voxel/sensitivity/compliance.py --index 0 --test
"""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "Surface shape-derivative sensitivity is not yet implemented.\n\n"
        "Voxel density-based compliance SA (dC/drho on grids):\n"
        "  ./scripts/voxel/sensitivity/compliance.sh --index 0 --test"
    )


if __name__ == "__main__":
    main()
