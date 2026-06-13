#!/usr/bin/env bash
# Fetch pre-trained NITO checkpoints into nito/Checkpoints/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec python "$ROOT/scripts/fetch/checkpoints.py" "$@"
