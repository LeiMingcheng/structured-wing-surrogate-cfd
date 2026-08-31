#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${REPO_ROOT}/tmp"
SMOKE_ROOT="${SMOKE_ROOT:-$(mktemp -d "${REPO_ROOT}/tmp/ddp-smoke.XXXXXX")}"
DATASET_DIR="${SMOKE_ROOT}/dataset"
RUN_DIR="${SMOKE_ROOT}/run"

"${CONDA_PREFIX}/bin/python" "${REPO_ROOT}/scripts/create_ddp_smoke_dataset.py" \
  --output-dir "${DATASET_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
  "${CONDA_PREFIX}/bin/torchrun" --standalone --nproc-per-node=2 \
  -m surrogate3d.cli.train \
  --config "${REPO_ROOT}/configs/ddp-smoke.yaml" \
  --index "${DATASET_DIR}/index.csv" \
  --output-dir "${RUN_DIR}"

test -f "${RUN_DIR}/last.pt"
test -f "${RUN_DIR}/summary.json"
printf 'DDP smoke passed: %s\n' "${SMOKE_ROOT}"
