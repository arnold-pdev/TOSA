#!/usr/bin/env bash
# Voxel density-based compliance sensitivity (dC/drho on structured grids).
#
# Examples:
#   ./scripts/voxel/sensitivity/compliance.sh --index 0 --test
#   ./scripts/voxel/sensitivity/compliance.sh --index 3 --test \
#       --rho-file output/nito_predictions/3/rho_pred.npy
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
exec python "$ROOT/scripts/voxel/sensitivity/compliance.py" "$@"
