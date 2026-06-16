#!/usr/bin/env bash
# Planar-cap method validation on index 119 (PyVista pipelines only).
# Logs: output/surfaces/validation/planar-cap-v1/build.log
#
# Usage (from repo root):
#   bash batch/run_validation_planar_cap.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${REPO_ROOT}/scripts"
PYTHON="${PYTHON:-python}"

DATASET_ID="planar-cap-v1"
MATRIX="${REPO_ROOT}/batch/validation_planar_cap_matrix.json"
OUT_ROOT="${REPO_ROOT}/output/surfaces/validation/${DATASET_ID}"
MAIN_LOG="${OUT_ROOT}/build.log"
SUMMARY="${OUT_ROOT}/metrics/summary.csv"

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/metrics" "${OUT_ROOT}/vtp"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "${MAIN_LOG}"
}

: > "${MAIN_LOG}"
log "=== planar-cap validation start ==="
log "dataset=${DATASET_ID} matrix=${MATRIX}"

log "--- dry-run ---"
${PYTHON} scripts/surface/build_validation_dataset.py \
  --matrix "${MATRIX}" \
  --dataset-id "${DATASET_ID}" \
  --dry-run 2>&1 | tee -a "${MAIN_LOG}"

log "--- build matrix ---"
${PYTHON} scripts/surface/build_validation_dataset.py \
  --matrix "${MATRIX}" \
  --dataset-id "${DATASET_ID}" \
  --on-load-fail warn \
  --on-bc-fail warn \
  -q \
  2>&1 | tee -a "${MAIN_LOG}"

log "--- metrics table ---"
${PYTHON} scripts/surface/view_validation.py \
  --dataset "${DATASET_ID}" \
  --index 119 \
  2>&1 | tee -a "${MAIN_LOG}"

if [[ -f "${SUMMARY}" ]]; then
  log "--- summary.csv ---"
  tee -a "${MAIN_LOG}" < "${SUMMARY}"
fi

log "=== planar-cap validation complete ==="
log "VIEW.md: ${OUT_ROOT}/VIEW.md"
