"""Pre-split parquet data by timestamp into train/valid at row level.

Usage (on Taiji platform):
    python preprocess_split_by_timestamp.py

Reads ``TRAIN_DATA_PATH`` (input) and ``USER_CACHE_PATH`` (output + marker).
Writes ``{USER_CACHE_PATH}/train/`` and ``{USER_CACHE_PATH}/valid/`` with
strict time separation: all train rows have timestamp <= threshold < all valid rows.

After running once, subsequent training jobs see the marker and skip repartitioning.
"""

import os
import sys
import json
import shutil
import logging
import argparse
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _collect_all_timestamps(data_dir: str, sample_rate: float = 1.0) -> np.ndarray:
    """Read timestamp column from every row group and return all timestamps."""
    timestamps = []
    pq_files = sorted(f for f in os.listdir(data_dir) if f.endswith('.parquet'))
    if not pq_files:
        raise FileNotFoundError(f"No .parquet files found in {data_dir}")

    for fname in pq_files:
        fpath = os.path.join(data_dir, fname)
        pf = pq.ParquetFile(fpath)
        for i in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(i, columns=['timestamp'])
            ts = table.column('timestamp').to_numpy().astype(np.int64)
            if sample_rate < 1.0:
                keep = np.random.rand(len(ts)) < sample_rate
                ts = ts[keep]
            timestamps.append(ts)

    return np.concatenate(timestamps)


def _write_filtered_parquet(
    src_dir: str, dst_dir: str, col_names: list,
    ts_threshold: int, take_high: bool, schema_path: str,
) -> int:
    """Read every parquet file, keep rows with timestamp above/below threshold,
    write to a new parquet file in dst_dir.

    Returns the total number of rows written.
    """
    os.makedirs(dst_dir, exist_ok=True)

    pq_files = sorted(f for f in os.listdir(src_dir) if f.endswith('.parquet'))
    total_rows = 0
    for fname in pq_files:
        fpath = os.path.join(src_dir, fname)
        pf = pq.ParquetFile(fpath)
        slices = []
        for i in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(i)
            ts = table.column('timestamp').to_numpy().astype(np.int64)
            if take_high:
                mask = ts > ts_threshold
            else:
                mask = ts <= ts_threshold
            if mask.sum() == 0:
                continue
            sliced = table.filter(pa.array(mask))
            slices.append(sliced)

        if not slices:
            continue

        combined = pa.concat_tables(slices)
        out_path = os.path.join(dst_dir, fname)
        pq.write_table(combined, out_path)
        total_rows += combined.num_rows
        logging.info(f"  {fname}: {combined.num_rows} rows -> {out_path}")

    # copy schema.json
    if os.path.exists(schema_path):
        shutil.copy2(schema_path, os.path.join(dst_dir, 'schema.json'))

    return total_rows


def split_by_timestamp(
    data_dir: str,
    cache_dir: str,
    schema_path: str = '',
    valid_ratio: float = 0.1,
    force: bool = False,
    sample_rate: float = 1.0,
) -> str:
    """Split parquet data at row level by timestamp.

    Args:
        data_dir: Directory containing input .parquet files.
        cache_dir: Output root; ``train/`` and ``valid/`` subdirs written here.
        schema_path: Path to schema.json (auto-detected if empty).
        valid_ratio: Fraction of (latest) rows to reserve for validation.
        force: Re-run even if marker file exists.
        sample_rate: Fraction of timestamps to sample when computing threshold
            (set < 1.0 for large datasets to speed up the scan).

    Returns:
        The absolute path to the cache directory (``cache_dir``).
    """
    marker = os.path.join(cache_dir, '_TIMESTAMP_SPLIT_DONE')
    if not force and os.path.exists(marker):
        logging.info(f"Timestamp split already done (marker={marker}), skipping.")
        return cache_dir

    if not schema_path:
        schema_path = os.path.join(data_dir, 'schema.json')
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"schema.json not found at {schema_path}")

    # --- Step 1: collect timestamps and compute threshold ---
    logging.info("Collecting timestamps from all row groups ...")
    t0 = time.time()
    all_ts = _collect_all_timestamps(data_dir, sample_rate=sample_rate)
    n_total = len(all_ts)
    logging.info(f"  sampled {n_total} timestamps in {time.time() - t0:.1f}s, "
                 f"range=[{all_ts.min()}, {all_ts.max()}]")

    # Sort and find threshold
    all_ts.sort()
    split_idx = int(n_total * (1 - valid_ratio))
    threshold = int(all_ts[split_idx])
    n_train_est = split_idx
    n_valid_est = n_total - split_idx
    logging.info(f"  threshold={threshold} "
                 f"(train ~{n_train_est} rows, valid ~{n_valid_est} rows)")

    # --- Step 2: read + filter + write ---
    train_dir = os.path.join(cache_dir, 'train')
    valid_dir = os.path.join(cache_dir, 'valid')

    # Clear any partial output from a previous failed run
    for d in [train_dir, valid_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)

    # Read and filter
    col_names = pq.read_schema(os.path.join(
        data_dir, sorted(f for f in os.listdir(data_dir) if f.endswith('.parquet'))[0])
    ).names

    logging.info("Writing training split (ts <= threshold) ...")
    t0 = time.time()
    n_train = _write_filtered_parquet(data_dir, train_dir, col_names, threshold,
                                       take_high=False, schema_path=schema_path)
    logging.info(f"  done in {time.time() - t0:.1f}s, {n_train} rows")

    logging.info("Writing validation split (ts > threshold) ...")
    t0 = time.time()
    n_valid = _write_filtered_parquet(data_dir, valid_dir, col_names, threshold,
                                       take_high=True, schema_path=schema_path)
    logging.info(f"  done in {time.time() - t0:.1f}s, {n_valid} rows")

    # --- Step 3: write marker ---
    info = dict(
        threshold=int(threshold),
        num_train_rows=n_train,
        num_valid_rows=n_valid,
        valid_ratio=valid_ratio,
        timestamp_min=int(all_ts.min()),
        timestamp_max=int(all_ts.max()),
    )
    with open(marker, 'w') as f:
        json.dump(info, f, indent=2)
    logging.info(f"Marker written to {marker}: {json.dumps(info)}")
    logging.info(f"Done. Train: {n_train} rows, Valid: {n_valid} rows")

    return cache_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-split parquet data by timestamp into train/valid.")
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Input data dir (default: $TRAIN_DATA_PATH)')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='Cache dir for output (default: $USER_CACHE_PATH)')
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                        help='Fraction of latest rows for validation (default: 0.1)')
    parser.add_argument('--force', action='store_true', default=False,
                        help='Force re-split even if marker exists')
    parser.add_argument('--sample_rate', type=float, default=1.0,
                        help='Sample rate for timestamp scan (default: 1.0; set <1 for speed)')
    args = parser.parse_args()

    data_dir = args.data_dir or os.environ.get('TRAIN_DATA_PATH', 'data')
    cache_dir = args.cache_dir or os.environ.get('USER_CACHE_PATH', '')
    if not cache_dir:
        raise RuntimeError(
            "Must set --cache_dir or $USER_CACHE_PATH (Taiji platform sets this automatically)")

    schema_path = os.path.join(data_dir, 'schema.json')
    if not os.path.exists(schema_path):
        # search one level up
        alt = os.path.join(os.path.dirname(data_dir.rstrip('/')), 'schema.json')
        if os.path.exists(alt):
            schema_path = alt

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        stream=sys.stderr,
    )

    split_by_timestamp(
        data_dir=data_dir,
        cache_dir=cache_dir,
        schema_path=schema_path,
        valid_ratio=args.valid_ratio,
        force=args.force,
        sample_rate=args.sample_rate,
    )


if __name__ == '__main__':
    main()
