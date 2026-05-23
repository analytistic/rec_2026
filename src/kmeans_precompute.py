"""Pre-compute K-means centroids for f61/f87 dense embeddings.

Usage:
    python -m src.kmeans_precompute --data_dir <train_data> --K 128 --output_dir <out>
"""
import argparse, logging, os, numpy as np
from pathlib import Path
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _kmeans(X, K, max_iter=50, seed=42):
    rng = np.random.RandomState(seed)
    N, D = X.shape
    centers = X[rng.choice(N, K, replace=False)].copy()
    for it in range(max_iter):
        labels = np.empty(N, dtype=np.int32)
        for start in range(0, N, 100000):
            end = min(start + N, N)
            Xc = X[start:end]
            dists = np.zeros((end - start, K), dtype=np.float32)
            for k in range(K):
                diff = Xc - centers[k:k + 1]
                dists[:, k] = np.sum(diff * diff, axis=1)
            labels[start:end] = np.argmin(dists, axis=1).astype(np.int32)
        new_centers = np.zeros_like(centers)
        for k in range(K):
            mask = labels == k
            new_centers[k] = X[mask].mean(axis=0) if mask.sum() > 0 else centers[k]
        shift = np.mean((new_centers - centers) ** 2)
        centers = new_centers
        logging.info(f"  Iter {it + 1}/{max_iter}: shift={shift:.8f}")
        if shift < 1e-10:
            break
    return centers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--K', type=int, default=128)
    parser.add_argument('--output_dir', type=str, default='centroids')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    EXPECTED = {'user_dense_feats_61': 256, 'user_dense_feats_87': 320}
    files = sorted(Path(args.data_dir).glob("*.parquet"))
    f61_list, f87_list = [], []
    total_rows = 0
    for pf in files:
        t = pq.read_table(str(pf))
        for name, out in [('user_dense_feats_61', f61_list), ('user_dense_feats_87', f87_list)]:
            col = t.column(name).to_numpy()
            for arr in col:
                total_rows += 1
                if isinstance(arr, np.ndarray) and arr.ndim == 1 and len(arr) == EXPECTED[name]:
                    out.append(arr.astype(np.float32))
    f61, f87 = np.stack(f61_list), np.stack(f87_list)
    logging.info(f"f61: {len(f61_list)}/{total_rows} valid, f87: {len(f87_list)}/{total_rows} valid")
    logging.info(f"f61: {f61.shape}, f87: {f87.shape}")

    np.save(os.path.join(args.output_dir, 'f61_centroids.npy'), _kmeans(f61, args.K))
    np.save(os.path.join(args.output_dir, 'f87_centroids.npy'), _kmeans(f87, args.K))
    logging.info(f"Saved centroids to {args.output_dir}/")


if __name__ == "__main__":
    main()
