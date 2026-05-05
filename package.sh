#!/bin/bash
# Package src/ + analyze_labels.py into src.zip for online training.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"
zip -r src.zip src/ analyze_labels.py -x "src/__pycache__/*" "*.pyc" "src/.DS_Store"
echo "Created ${SCRIPT_DIR}/src.zip"
