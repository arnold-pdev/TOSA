#!/usr/bin/env bash
# Remove cached STL files from public/stl/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec python "$ROOT/scripts/fetch/clean_stl.py" "$@"
