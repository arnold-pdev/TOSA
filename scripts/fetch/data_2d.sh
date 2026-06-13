#!/usr/bin/env bash
# Fetch NITO 2D .npy data into nito/Data/.
#
# Examples:
#   ./scripts/fetch/data_2d.sh --test-only
#   ./scripts/fetch/data_2d.sh --full
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec python "$ROOT/scripts/fetch/data_2d.py" "$@"
