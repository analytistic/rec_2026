#!/usr/bin/env python3
"""Scan training parquet files for label distribution (positive ratio).

Usage:
    python analyze_labels.py [data_dir] [max_rows]

Output: total/positive/negative counts, positive ratio, recommended focal_alpha.
"""

import sys
import os
import json
import glob
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def analyze(data_dir: str, max_rows: int = 0) -> None:
    # Find parquet files.
    pattern = os.path.join(data_dir, '*.parquet')
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No .parquet files found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(files)} parquet files in {data_dir}")

    # Get column index for label_type.
    pf = pq.ParquetFile(files[0])
    names = pf.schema_arrow.names
    if 'label_type' not in names:
        print(f"'label_type' column not found. Available: {names}")
        sys.exit(1)
    label_idx = names.index('label_type')
    print(f"'label_type' column index: {label_idx}")
    print(f"Total row groups across files: {sum(pq.ParquetFile(f).metadata.num_row_groups for f in files)}")

    total = 0
    pos = 0

    for fpath in files:
        pf = pq.ParquetFile(fpath)
        for rg_idx in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(rg_idx)
            arr = table.column(label_idx).to_numpy(zero_copy_only=False).astype(np.int64)
            labels = (arr == 2).astype(np.int64)
            total += len(labels)
            pos += int(labels.sum())

            if max_rows > 0 and total >= max_rows:
                break
        if max_rows > 0 and total >= max_rows:
            break

    neg = total - pos
    ratio = pos / total if total > 0 else 0.0

    print(f"\n{'='*50}")
    print(f"Total samples: {total:,}")
    print(f"Positive:      {pos:,}  ({ratio*100:.2f}%)")
    print(f"Negative:      {neg:,}  ({(1-ratio)*100:.2f}%)")
    print(f"Pos/Neg ratio: 1:{neg/max(pos,1):.1f}")
    print(f"\nRecommended Focal Loss params (alpha = neg_ratio for class balance):")
    print(f"  --loss_type focal")
    print(f"  --focal_alpha {1-ratio:.4f}")
    print(f"  --focal_gamma 2.0")


if __name__ == '__main__':
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data'
    max_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    analyze(data_dir, max_rows)
