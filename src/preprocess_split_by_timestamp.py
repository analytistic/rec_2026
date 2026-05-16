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


def _log_dir_stats(data_dir: str, label: str) -> None:
    """Log number of parquet files, row groups, total rows, size, and
    timestamp range (from column chunk statistics, no data scan)."""
    pq_files = sorted(f for f in os.listdir(data_dir) if f.endswith('.parquet'))
    if not pq_files:
        logging.info(f"  [{label}] 0 parquet files, empty dir")
        return
    total_rgs = 0
    total_rows = 0
    total_bytes = 0
    ts_min = np.iinfo(np.int64).max
    ts_max = np.iinfo(np.int64).min
    for fname in pq_files:
        fpath = os.path.join(data_dir, fname)
        pf = pq.ParquetFile(fpath)
        nrg = pf.metadata.num_row_groups
        total_rgs += nrg
        # find the timestamp column index
        ts_col_idx = pf.schema_arrow.get_field_index('timestamp')
        for i in range(nrg):
            rg = pf.metadata.row_group(i)
            total_rows += rg.num_rows
            total_bytes += rg.total_byte_size
            col = rg.column(ts_col_idx)
            if col.is_stats_set and col.statistics.has_min_max:
                ts_min = min(ts_min, int(col.statistics.min))
                ts_max = max(ts_max, int(col.statistics.max))
    logging.info(
        f"  [{label}] {len(pq_files)} files, {total_rgs} row groups, "
        f"{total_rows} rows, {total_bytes / 1e9:.2f} GB, "
        f"ts=[{ts_min}, {ts_max}]")


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
    row_group_size: int = 65536,
) -> int:
    """Read every parquet file, keep rows with timestamp above/below threshold,
    write to a single output file in dst_dir with properly sized row groups.

    Accumulates filtered rows across all source files and flushes every
    ``row_group_size`` rows so output row groups are consistently sized
    (no tiny boundary RGs → no tiny validation batches).

    Returns the total number of rows written.
    """
    os.makedirs(dst_dir, exist_ok=True)

    out_path = os.path.join(dst_dir, 'valid.parquet')
    pq_files = sorted(f for f in os.listdir(src_dir) if f.endswith('.parquet'))
    total_rows = 0

    writer: Optional[pq.ParquetWriter] = None
    buf: Optional[pa.Table] = None
    schema: Optional[pa.Schema] = None
    try:
        for fname in pq_files:
            fpath = os.path.join(src_dir, fname)
            pf = pq.ParquetFile(fpath)
            for i in range(pf.metadata.num_row_groups):
                table = pf.read_row_group(i)
                ts = table.column('timestamp').to_numpy().astype(np.int64)
                mask = (ts > ts_threshold) if take_high else (ts <= ts_threshold)
                if mask.sum() == 0:
                    continue
                sliced = table.filter(pa.array(mask))
                if schema is None:
                    schema = sliced.schema
                buf = sliced if buf is None else pa.concat_tables([buf, sliced])

                # flush when buffer reaches row_group_size
                if buf.num_rows >= row_group_size:
                    if writer is None:
                        writer = pq.ParquetWriter(out_path, schema)
                    writer.write_table(buf)
                    total_rows += buf.num_rows
                    buf = None

        # flush remainder
        if buf is not None:
            if writer is None:
                writer = pq.ParquetWriter(out_path, schema)
            writer.write_table(buf)
            total_rows += buf.num_rows
    finally:
        if writer is not None:
            writer.close()

    if total_rows > 0:
        logging.info(f"  valid.parquet: {total_rows} rows -> {out_path}")

    # copy schema.json
    if os.path.exists(schema_path):
        shutil.copy2(schema_path, os.path.join(dst_dir, 'schema.json'))

    return total_rows


def split_by_timestamp(
    data_dir: str,
    cache_dir: str,
    schema_path: str = '',
    valid_ratio: float = 0.1,
    sample_rate: float = 1.0,
    row_group_size: int = 65536,
) -> str:
    """Split parquet data at row level by timestamp.

    Always runs (clears any previous output in ``cache_dir/valid``).
    Control whether to call this from your shell script.

    Args:
        data_dir: Directory containing input .parquet files.
        cache_dir: Output root; ``valid/`` subdir written here.
        schema_path: Path to schema.json (auto-detected if empty).
        valid_ratio: Fraction of (latest) rows to reserve for validation.
        sample_rate: Fraction of timestamps to sample when computing threshold
            (set < 1.0 for large datasets to speed up the scan).
        row_group_size: Output row group size; consecutive filtered rows are
            merged and flushed every ``row_group_size`` rows.  Larger values
            reduce the number of tiny batches during validation.

    Returns:
        The absolute path to the cache directory (``cache_dir``).
    """
    if not schema_path:
        schema_path = os.path.join(data_dir, 'schema.json')
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"schema.json not found at {schema_path}")

    # --- Step 0: log input dir ---
    logging.info(f"Input data dir: {data_dir}")
    _log_dir_stats(data_dir, "input")

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
    valid_dir = os.path.join(cache_dir, 'valid')

    # Clear previous output (valid + stale train/ from earlier attempts)
    for subdir in ('valid', 'train'):
        p = os.path.join(cache_dir, subdir)
        if os.path.exists(p):
            shutil.rmtree(p)

    # Read and filter
    col_names = pq.read_schema(os.path.join(
        data_dir, sorted(f for f in os.listdir(data_dir) if f.endswith('.parquet'))[0])
    ).names

    # Only write the validation split (~10%) to cache to stay within quota;
    # training reads from the original TRAIN_DATA_PATH directly.
    logging.info("Writing validation split (ts > threshold) ...")
    t0 = time.time()

    # Match output RG size to the original data's RG row count so each RG
    # yields several full batches and multi-worker loading distributes RGs
    # evenly across workers.
    all_pq = sorted(f for f in os.listdir(data_dir) if f.endswith('.parquet'))
    orig_max_rg = max(
        pq.ParquetFile(os.path.join(data_dir, f)).metadata.row_group(i).num_rows
        for f in all_pq
        for i in range(pq.ParquetFile(os.path.join(data_dir, f)).metadata.num_row_groups)
    )
    logging.info(f"  orig_max_rg={orig_max_rg}, using as output row_group_size")

    n_valid = _write_filtered_parquet(data_dir, valid_dir, col_names, threshold,
                                       take_high=True, schema_path=schema_path,
                                       row_group_size=orig_max_rg)
    logging.info(f"  done in {time.time() - t0:.1f}s, {n_valid} rows")

    logging.info(f"Done. Valid only: {n_valid} rows (train reads from original data)")
    _log_dir_stats(valid_dir, "output/valid")

    # Write threshold so training can filter rows > threshold from original data
    with open(os.path.join(cache_dir, 'threshold.json'), 'w') as f:
        json.dump(dict(threshold=int(threshold), num_valid_rows=n_valid), f)
    logging.info(f"Threshold written to {os.path.join(cache_dir, 'threshold.json')}: {threshold}")

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
    parser.add_argument('--sample_rate', type=float, default=1.0,
                        help='Sample rate for timestamp scan (default: 1.0; set <1 for speed)')
    parser.add_argument('--row_group_size', type=int, default=65536,
                        help='Output row group size (default: 65536). '
                             'Merges filtered rows within each file and flushes '
                             'every N rows to avoid tiny boundary row groups.')
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
        sample_rate=args.sample_rate,
        row_group_size=args.row_group_size,
    )


if __name__ == '__main__':
    main()
