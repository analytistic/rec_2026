#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

python3 -m zipfile -e "${SCRIPT_DIR}/src.zip" "${SCRIPT_DIR}"

# ---- Step 1 (optional): Pre-split data by timestamp ----
# Uncomment to run preprocessing. Comment out after first run to skip.
# python3 -u -m src.preprocess_split_by_timestamp --valid_ratio 0.1

# ---- Step 2 (optional): Pre-compute K-means centroids for f61/f87 ----
# Comment out after first run to skip.
python3 -u -m src.kmeans_precompute \
    --data_dir "${TRAIN_DATA_PATH}" \
    --K 128 \
    --output_dir "${USER_CACHE_PATH}/centroids"

# ---- Step 3: Train ----
CENTROIDS_DIR="${USER_CACHE_PATH}/centroids"
if [ -d "$CENTROIDS_DIR" ]; then
    CENTROIDS_ARG="--centroids_dir ${CENTROIDS_DIR}"
else
    CENTROIDS_ARG=""
fi
python3 -u -m src.train \
    --config "${SCRIPT_DIR}/config.yaml" \
    --data_dir "${TRAIN_DATA_PATH}" \
    --ckpt_dir "${TRAIN_CKPT_PATH}" \
    --log_dir "${TRAIN_LOG_PATH}" \
    --ns_groups_json "${SCRIPT_DIR}/v1.json" \
    ${CENTROIDS_ARG} \
    "$@"