#!/usr/bin/env bash
# Compare critique-proposal methods vs monolithic baseline on index 119.
# Logs: output/surfaces/validation/critique-methods-v1/build.log

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${REPO_ROOT}/scripts"
PYTHON="${PYTHON:-python}"

DATASET_ID="critique-methods-v1"
MATRIX="${REPO_ROOT}/batch/validation_critique_methods_matrix.json"
OUT_ROOT="${REPO_ROOT}/output/surfaces/validation/${DATASET_ID}"
MAIN_LOG="${OUT_ROOT}/build.log"

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/metrics" "${OUT_ROOT}/vtp"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "${MAIN_LOG}"
}

: > "${MAIN_LOG}"
log "=== critique-methods validation start ==="

${PYTHON} scripts/surface/build_validation_dataset.py \
  --matrix "${MATRIX}" \
  --dataset-id "${DATASET_ID}" \
  --on-load-fail warn \
  --on-bc-fail warn \
  -q \
  2>&1 | tee -a "${MAIN_LOG}"

${PYTHON} scripts/surface/view_validation.py \
  --dataset "${DATASET_ID}" \
  --index 119 \
  2>&1 | tee -a "${MAIN_LOG}"

log "=== critique-methods validation complete ==="
log "VIEW.md: ${OUT_ROOT}/VIEW.md"
