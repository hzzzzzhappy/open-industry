#!/bin/bash
set -euo pipefail

# ====== Config ======
SEEDS=(1)
thr=0
lam=0.1

CUDA_ID=0
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

# Datasets to run
DATASETS=(real)  #open-industry real shapenet

# Dataset-specific known defects
declare -A KNOWN_DEFECTS
KNOWN_DEFECTS[mc3dad]="Bump Deformation"
KNOWN_DEFECTS[real]="bulge"
KNOWN_DEFECTS[shapenet]="bulge"

# ====== Run ======
for dataset in "${DATASETS[@]}"; do
  known_defects="${KNOWN_DEFECTS[$dataset]:-}"

  if [[ -z "${known_defects}" ]]; then
    echo "ERROR: known_defects not set for dataset=${dataset}"
    exit 1
  fi

  for SEED in "${SEEDS[@]}"; do
    EXP_NAME="${dataset}_thr${thr}_lam${lam}_seed${SEED}"

    # Create independent directory for each seed
    SEED_DIR="${LOG_DIR}/${SEED}"
    mkdir -p "${SEED_DIR}"

    LOG_FILE="${SEED_DIR}/${EXP_NAME}.txt"

    echo "============================================================"
    echo "Run: open3dad | dataset=${dataset} | seed=${SEED}"
    echo "thr=${thr}, lam=${lam}, known_defects=${known_defects}"
    echo "Log: ${LOG_FILE}"
    echo "============================================================"

    # Write experiment configuration
    cat << EOF > "${LOG_FILE}"
========== Experiment Config ==========
dataset: ${dataset}
abnormal_ratio_threshold: ${thr}
lam: ${lam}
seed: ${SEED}
known_defects: ${known_defects}
cuda: ${CUDA_ID}
=======================================

EOF

    # Run experiment and append logs
    CUDA_VISIBLE_DEVICES="${CUDA_ID}" \
    python main.py \
      --dataset "${dataset}" \
      --num_group 4096 \
      --group_size 128 \
      --max_nn 40 \
      --use_LFSA True \
      --use_MSND True \
      --expname "${EXP_NAME}" \
      --known_defects ${known_defects} \
      --pollution_per_defect 5 \
      --abnormal_ratio_threshold "${thr}" \
      --lam "${lam}" \
      --seed "${SEED}" \
      2>&1 | tee -a "${LOG_FILE}"
  done
done
