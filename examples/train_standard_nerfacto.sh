#!/usr/bin/env bash
# Standard Nerfacto on an independently prepared BEG-Connect reconstruction.
# This is the BEG-C training stage, not a separate BEG-C neural model.
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: bash examples/train_standard_nerfacto.sh PROCESSED_DATA OUTPUT_DIR EXPERIMENT_NAME" >&2
  exit 2
fi

processed_data="$1"
output_dir="$2"
experiment_name="$3"
if [[ ! -f "${processed_data}/transforms.json" ]]; then
  echo "Missing transforms.json in processed data: ${processed_data}" >&2
  exit 2
fi
if ! command -v ns-train >/dev/null 2>&1; then
  echo "ns-train not found; install a compatible Nerfstudio environment" >&2
  exit 2
fi

ns-train nerfacto \
  --experiment-name "${experiment_name}" \
  --output-dir "${output_dir}" \
  --max-num-iterations 15000 \
  --steps-per-eval-batch 500 \
  --pipeline.model.camera-optimizer.mode off \
  --pipeline.model.num-nerf-samples-per-ray 32 \
  --pipeline.model.eval-num-rays-per-chunk 2048 \
  --pipeline.datamanager.train-num-rays-per-batch 4096 \
  nerfstudio-data \
  --data "${processed_data}" \
  --downscale-factor 2 \
  --eval-mode interval \
  --eval-interval 8
