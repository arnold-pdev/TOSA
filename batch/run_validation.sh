#!/usr/bin/env bash
# Run the compatible-pipelines validation matrix on index 119.
# Logs: output/surfaces/validation/compatible-pipelines-v1/build.log
#
# Usage (from repo root):
#   bash batch/run_validation.sh
#   bash batch/run_validation.sh 2>&1 | tee -a path/to/extra.log

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${REPO_ROOT}/scripts"
PYTHON="${PYTHON:-python}"

DATASET_ID="compatible-pipelines-v1"
OUT_ROOT="${REPO_ROOT}/output/surfaces/validation/${DATASET_ID}"
MAIN_LOG="${OUT_ROOT}/build.log"
SUMMARY="${OUT_ROOT}/metrics/summary.csv"

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/metrics" "${OUT_ROOT}/vtp"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "${MAIN_LOG}"
}

on_error() {
  log "ERROR: validation interrupted at line $1 (exit $?)"
}
trap 'on_error $LINENO' ERR

: > "${MAIN_LOG}"
log "=== validation run start ==="
log "repo=${REPO_ROOT}"
log "python=$(${PYTHON} --version 2>&1)"
log "dataset=${DATASET_ID} index=119"

log "--- dry-run plan ---"
${PYTHON} scripts/surface/build_validation_dataset.py --dry-run 2>&1 | tee -a "${MAIN_LOG}"

log "--- full matrix build ---"
${PYTHON} scripts/surface/build_validation_dataset.py \
  --dataset-id "${DATASET_ID}" \
  --on-load-fail warn \
  --on-bc-fail warn \
  2>&1 | tee -a "${MAIN_LOG}"

log "--- metrics table (index 119) ---"
${PYTHON} scripts/surface/view_validation.py \
  --dataset "${DATASET_ID}" \
  --index 119 \
  2>&1 | tee -a "${MAIN_LOG}"

log "--- per-recipe smoke: list VTP outputs ---"
for recipe in extract_baseline pyvista_laplacian pyvista_taubin pyvista_taubin_regrow pyvista_hc sdf_lewiner_calibrated sdf_masked_taubin; do
  vtp="${OUT_ROOT}/vtp/119/${recipe}.vtp"
  if [[ -f "${vtp}" ]]; then
    log "OK  ${vtp} ($(wc -c < "${vtp}") bytes)"
  else
    log "MISSING  ${vtp}"
  fi
done

if [[ -f "${SUMMARY}" ]]; then
  log "--- summary.csv ---"
  tee -a "${MAIN_LOG}" < "${SUMMARY}"
else
  log "WARN: missing ${SUMMARY}"
fi

log "=== validation run complete ==="
log "VIEW.md: ${OUT_ROOT}/VIEW.md"
log "Open mesh: python scripts/surface/view_validation.py --dataset ${DATASET_ID} --open pyvista_taubin"
