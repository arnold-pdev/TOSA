#!/usr/bin/env bash
# Fetch NITO 3D train data into nito/Data/3D/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec python "$ROOT/scripts/fetch/data_3d.py" "$@"
