#!/usr/bin/env bash
# Fetch selected NITO-3D STL meshes into public/stl/.
#
# Examples:
#   ./scripts/fetch/stl.sh --indices 0 3 42
#   ./scripts/fetch/stl.sh --range 0 49
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec python "$ROOT/scripts/fetch/stl.py" "$@"
