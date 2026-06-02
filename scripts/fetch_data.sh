#!/usr/bin/env bash
# Fetch NITO data into nito/Data/ (TOSA script; does not patch the submodule).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec python "$ROOT/scripts/fetch_nito_data.py" "$@"
