#!/usr/bin/env bash
set -euo pipefail

: "${DATASET_INDEX:?Set DATASET_INDEX to the structured dataset index.csv}"
: "${TRAIN_OUTPUT_DIR:?Set TRAIN_OUTPUT_DIR to the run output directory}"

CONFIG_PATH="${CONFIG_PATH:-configs/global-fno-gpu.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

"${CONDA_PREFIX}/bin/torchrun" --standalone --nproc-per-node="${NPROC_PER_NODE}" \
  -m surrogate3d.cli.train \
  --config "${CONFIG_PATH}" \
  --index "${DATASET_INDEX}" \
  --output-dir "${TRAIN_OUTPUT_DIR}"
