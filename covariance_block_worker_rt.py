import os
import sys
import time
import numpy as np
from scipy.interpolate import interp1d

from config_custom_10bins import (
    NBINS, WORK_ROOT, CL_NPZ_PATH,
    pair_meta_path, pair_counts_path, pair_merged_dir,
    xi_table_path, cov_block_dir, ensure_dirs
)

def angular_distance(vec1, vec2):
    """
    角度距離を計算する関数。スカラーベクトルと配列に対応。
    
    Args:
        vec1: shape (3,) または (N, 3)
        vec2: shape (3,) または (M, 3)
    
    Returns:
        スカラーまたは (N, M) 形の距離行列
        例: angular_distance((3,), (3,)) -> スカラー
            angular_distance((N,3), (M,3)) -> (N,M) 行列
    """
    vec1 = np.atleast_1d(np.asarray(vec1))
    vec2 = np.atleast_1d(np.asarray(vec2))
    
    # 1次元の場合は形状を (1, 3) に変更
    if vec1.ndim == 1:
        vec1 = vec1[np.newaxis, :]
    if vec2.ndim == 1:
        vec2 = vec2[np.newaxis, :]
    
    # (N, 3) と (M, 3) -> (N, 1, 3) と (1, M, 3) でブロードキャスト
    # 内積: (N, 1, 3) ・ (1, M, 3) -> (N, M)
    c = np.sum(vec1[:, np.newaxis, :] * vec2[np.newaxis, :, :], axis=2)
    c = np.clip(c, -1.0, 1.0)
    result = np.arccos(c)
    
    # スカラー入力だった場合はスカラーを返す
    if result.shape == (1, 1):
        return result.item()
    return result


def main():
    if len(sys.argv) != 4:
        raise ValueError("Usage: python covariance_block_worker.py <block_id> <a_start> <a_end>")

    block_id = int(sys.argv[1])
    a_start = int(sys.argv[2])
    a_end = int(sys.argv[3])

    ensure_dirs()
    os.makedirs(cov_block_dir(), exist_ok=True)

    print("=== covariance_block_worker.py started ===", flush=True)
    print(f"block_id = {block_id}", flush=True)
    print(f"a_start  = {a_start}", flush=True)
    print(f"a_end    = {a_end}", flush=True)

    t0_all = time.time()

    with np.load(pair_meta_path()) as d:
        theta = d["theta"]
        theta_edges = d["theta_edges"]
        theta_edges_rad = d["theta_edges_rad"]
        weights = d["weights"].astype(np.float64)
        theta_vecs = d["theta_vecs"].astype(np.float64)

    pair_counts = np.load(pair_counts_path())

    with np.load(xi_table_path()) as d:
        theta_grid = d["theta_grid"]
        xi_grid = d["xi_grid"]

    a_end = min(a_end, NBINS)
    nrows = a_end - a_start
    cov_block = np.full((nrows, NBINS), np.nan, dtype=np.float64)

    
    
    for a in range(a_start, a_end):
        t_row = time.time()
        pair_file_a = os.path.join(pair_merged_dir(), f"pairs_bin_{a:02d}.npy")
        pairs_a = np.load(pair_file_a, mmap_mode="r")

        print(
            f"[row-start] a={a}/{NBINS-1}, pairs_a={pairs_a.shape[0]}, pair_count_expected={pair_counts[a]}",
            flush=True
        )

        for b in range(a, NBINS):
            pair_file_b = os.path.join(pair_merged_dir(), f"pairs_bin_{b:02d}.npy")
            pairs_b = np.load(pair_file_b, mmap_mode="r")

            print(
                f"  [cell-start] (a,b)=({a},{b}), pairs_a={pairs_a.shape[0]}, pairs_b={pairs_b.shape[0]}",
                flush=True
            )


            cov_block[a - a_start, b] = compute_covariance_vectorized(pairs_a, pairs_b, weights, theta_vecs, theta_grid, xi_grid)
            
            print(
                f"  [cell-done] (a,b)=({a},{b}), cov={total:.6e}, "
                f"elapsed_row={time.time() - t_row:.1f} sec",
                flush=True
            )

        print(
            f"[row-done] a={a}/{NBINS-1}, row_elapsed={time.time() - t_row:.1f} sec, "
            f"total_elapsed={time.time() - t0_all:.1f} sec",
            flush=True
        )

    outpath = os.path.join(cov_block_dir(), f"cov_block_{block_id:04d}.npz")
    np.savez(
        outpath,
        block_id=block_id,
        a_start=a_start,
        a_end=a_end,
        cov_block=cov_block,
        theta=theta,
        theta_edges=theta_edges,
        theta_edges_rad=theta_edges_rad,
        pair_counts=pair_counts,
    )

    print(f"saved {outpath}", flush=True)
    print(f"total elapsed = {time.time() - t0_all:.1f} sec", flush=True)
    print("=== covariance_block_worker.py finished ===", flush=True)

if __name__ == "__main__":
    main()


def xi_between(p, q, theta_vecs, theta_grid, xi_grid):
        """
        xi値を計算する関数。スカラーと配列に対応。
        
        Args:
            p, q: numpy配列
        Returns:
            xi値の配列 shape=(p.shape[0], q.shape[0]) 
        """
        
        ang = angular_distance(theta_vecs[p], theta_vecs[q])
        xi = interp1d(theta_grid, xi_grid, bounds_error=False, fill_value="extrapolate")(ang)    
            
        return xi

def compute_covariance_vectorized(pairs_a, pairs_b, weights, theta_vecs, theta_grid, xi_grid):
    """
    """
    # ベクトル化された方法で共分散を計算する関数。
    pairs_a_i = pairs_a[:, 0]
    pairs_a_j = pairs_a[:, 1]
    pairs_b_k = pairs_b[:, 0]
    pairs_b_l = pairs_b[:, 1]
    xixi_matrix = xi_between(pairs_a_i, pairs_b_k, theta_vecs, theta_grid, xi_grid) * xi_between(pairs_a_j, pairs_b_l, theta_vecs, theta_grid, xi_grid) \
                + xi_between(pairs_a_i, pairs_b_l, theta_vecs, theta_grid, xi_grid) * xi_between(pairs_a_j, pairs_b_k, theta_vecs, theta_grid, xi_grid)  # (N1, N2)
    
    xixi_matrix /= 2.0  # 対称性を考慮して半分にする
    # weight matrixを作成
    wij = weights(pairs_a[:, 0]) * weights(pairs_a[:, 1])  # (N1,)
    wkl = weights(pairs_b[:, 0]) * weights(pairs_b[:, 1])  # (N2,)
    weights_matrix = wij[:, np.newaxis] * wkl[np.newaxis, :]  # (N1, N2) 

    # 共分散の合計を計算
    cov_sum = np.sum(weights_matrix * xixi_matrix)  
        
    return cov_sum

