#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

python3 -m zipfile -e "${SCRIPT_DIR}/src.zip" "${SCRIPT_DIR}"

# python3 -u -m src.analyze_time_dist \
    --data_dir $TRAIN_DATA_PATH \
    --valid_dir $USER_CACHE_PATH/valid \
    --threshold_json $USER_CACHE_PATH/threshold.json \


# ---- Step 1 (optional): Pre-split data by timestamp into USER_CACHE_PATH ----
# Uncomment to run preprocessing. Comment out after first run to skip.
# python3 -u -m src.preprocess_split_by_timestamp --valid_ratio 0.1

# ---- Step 2: Train ----
# Valid reads from pre-split cache (strict time separation).
python3 -u -m src.train \
    --config "${SCRIPT_DIR}/config.yaml" \
    --data_dir "${TRAIN_DATA_PATH}" \
    --ckpt_dir "${TRAIN_CKPT_PATH}" \
    --log_dir "${TRAIN_LOG_PATH}" \
    --ns_groups_json "${SCRIPT_DIR}/v1.json" \
    #--valid_data_dir "${USER_CACHE_PATH}/valid" \
    "$@"