"""Pre-compute K-means centroids for f61/f87 dense embeddings (GPU).

Usage:
    python -m src.kmeans_precompute --data_dir <train_data> --K 128 --output_dir <out>
"""
import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _kmeans_gpu(X_np, K, max_iter=10, seed=42):
    """K-means on GPU via PyTorch. X_np: (N, D) float32 numpy array."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"K-means on {device}, N={X_np.shape[0]}, D={X_np.shape[1]}, K={K}")

    X = torch.from_numpy(X_np).to(device)  # (N, D)
    rng = torch.Generator(device=device).manual_seed(seed)
    idx = torch.randperm(X.shape[0], generator=rng, device=device)[:K]
    centers = X[idx].clone()

    for it in range(max_iter):
        # Distance: (N, K)
        dists = torch.cdist(X, centers, p=2)  # (N, K)
        labels = dists.argmin(dim=1)  # (N,)

        # Update centers
        new_centers = torch.zeros_like(centers)
        for k in range(K):
            mask = labels == k
            if mask.sum() > 0:
                new_centers[k] = X[mask].mean(dim=0)

        shift = (new_centers - centers).pow(2).mean().item()
        centers = new_centers
        logging.info(f"  Iter {it + 1}/{max_iter}: shift={shift:.8f}")
        if shift < 1e-10:
            break

    return centers.cpu()


def _extract(data_dir: str, col_name: str, expected_dim: int) -> np.ndarray:
    """Extract valid vectors from all parquet files."""
    files = sorted(Path(data_dir).glob("*.parquet"))
    all_vecs = []
    total = 0
    for pf in files:
        t = pq.read_table(str(pf))
        col = t.column(col_name).to_numpy()
        for arr in col:
            total += 1
            if isinstance(arr, np.ndarray) and arr.ndim == 1 and len(arr) == expected_dim:
                all_vecs.append(arr.astype(np.float32))
    arr = np.stack(all_vecs)
    logging.info(f"{col_name}: {len(all_vecs)}/{total} valid, shape={arr.shape}")
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--K', type=int, default=128)
    parser.add_argument('--output_dir', type=str, default='centroids')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    f61 = _extract(args.data_dir, 'user_dense_feats_61', 256)
    f87 = _extract(args.data_dir, 'user_dense_feats_87', 320)

    torch.save(_kmeans_gpu(f61, args.K), os.path.join(args.output_dir, 'f61_centroids.pt'))
    torch.save(_kmeans_gpu(f87, args.K), os.path.join(args.output_dir, 'f87_centroids.pt'))
    logging.info(f"Saved centroids to {args.output_dir}/")


if __name__ == "__main__":
    main()
