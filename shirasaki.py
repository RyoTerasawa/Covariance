from functools import lru_cache
import numpy as np

# ===== グリッド座標とインデックスの相互変換 =====
def coords_to_index(x, y, grid_width):
    """
    2次元グリッド座標(x, y)を1次元インデックスに変換
    
    座標を行優先（row-major）順で平坦化：
    index = y * grid_width + x
    
    Args:
        x: 列インデックス (0 <= x < grid_width)
        y: 行インデックス (0 <= y < grid_height)
        grid_width: グリッドの幅
    
    Returns:
        index: 1次元インデックス
    """
    return int(y * grid_width + x)


def index_to_coords(idx, grid_width):
    """
    1次元インデックスを2次元グリッド座標(x, y)に変換
    
    Args:
        idx: 1次元インデックス
        grid_width: グリッドの幅
    
    Returns:
        (x, y): グリッド座標 (列, 行)
    """
    y = idx // grid_width
    x = idx % grid_width
    return (x, y)


def coords_array_to_index_array(coords, grid_width):
    """
    複数の2次元座標をまとめて1次元インデックスに変換（ベクトル化版）
    
    Args:
        coords: (N, 2) 配列、各行は (x, y)
        grid_width: グリッドの幅
    
    Returns:
        indices: (N,) 配列、各要素は1次元インデックス
    """
    indices = coords[:, 1] * grid_width + coords[:, 0]
    return indices.astype(int)


def index_array_to_coords_array(indices, grid_width):
    """
    複数の1次元インデックスをまとめて2次元座標に変換（ベクトル化版）
    
    Args:
        indices: (N,) 配列、1次元インデックス
        grid_width: グリッドの幅
    
    Returns:
        coords: (N, 2) 配列、各行は (x, y)
    """
    y = indices // grid_width
    x = indices % grid_width
    return np.column_stack([x, y])

@lru_cache(maxsize=None)
def xi_between_cached(i, k):
    """キャッシング付きの距離関数"""
    return xi_between(i, k)

def compute_pair_distances_vectorized(pairs_a, pairs_b, weights, xi_between):
    """
    ペア間の距離を行列演算で高速計算（ループなし）
    
    Args:
        pairs_a: [(i, j), ...] のペアリスト
        pairs_b: [(k, l), ...] のペアリスト
        weights: グリッド上のウェイト配列
        xi_between: 距離関数
    """
    pairs_a = np.array(pairs_a)
    pairs_b = np.array(pairs_b)
    
    # 全インデックスを集めて距離行列をあらかじめ計算
    all_indices = np.unique(np.concatenate([pairs_a.flatten(), pairs_b.flatten()]))
    n_indices = len(all_indices)
    idx_map = {idx: i for i, idx in enumerate(all_indices)}
    
    # 距離行列を事前計算
    dist_matrix = np.zeros((n_indices, n_indices))
    for i, idx_i in enumerate(all_indices):
        for j, idx_j in enumerate(all_indices):
            dist_matrix[i, j] = xi_between(idx_i, idx_j)
    
    # ペアのインデックスを行列インデックスに変換
    pairs_a_idx = np.array([[idx_map[i], idx_map[j]] for i, j in pairs_a])
    pairs_b_idx = np.array([[idx_map[k], idx_map[l]] for k, l in pairs_b])
    
    # ウェイトの積を事前計算
    wij = weights[pairs_a[:, 0]] * weights[pairs_a[:, 1]]  # shape: (len(pairs_a),)
    wkl = weights[pairs_b[:, 0]] * weights[pairs_b[:, 1]]  # shape: (len(pairs_b),)
    
    # 距離を一括取得
    i_indices = pairs_a_idx[:, 0]  # shape: (len(pairs_a),)
    j_indices = pairs_a_idx[:, 1]  # shape: (len(pairs_a),)
    k_indices = pairs_b_idx[:, 0]  # shape: (len(pairs_b),)
    l_indices = pairs_b_idx[:, 1]  # shape: (len(pairs_b),)
    
    # ブロードキャスト計算
    # wij: (n_a, 1), wkl: (1, n_b)
    wij_broadcast = wij[:, np.newaxis]  # shape: (len(pairs_a), 1)
    wkl_broadcast = wkl[np.newaxis, :]  # shape: (1, len(pairs_b))
    weight_product = wij_broadcast * wkl_broadcast  # shape: (len(pairs_a), len(pairs_b))
    
    # 距離の計算をブロードキャスト
    d_ik = dist_matrix[i_indices[:, np.newaxis], k_indices[np.newaxis, :]]
    d_jl = dist_matrix[j_indices[:, np.newaxis], l_indices[np.newaxis, :]]
    d_il = dist_matrix[i_indices[:, np.newaxis], l_indices[np.newaxis, :]]
    d_jk = dist_matrix[j_indices[:, np.newaxis], k_indices[np.newaxis, :]]
    
    # 最終計算
    total = np.sum(weight_product * (d_ik * d_jl + d_il * d_jk))
    
    return total


def compute_pair_distances_same_pairs(pairs, weights, xi_between):
    """
    pairs_a = pairs_b の場合の最適化版
    
    計算量: O(n²) (距離行列) + O(m²) (ペア計算)
    元の式より大幅に高速
    
    Args:
        pairs: [(i, j), ...] のペアリスト
        weights: グリッド上のウェイト配列
        xi_between: 距離関数
    """
    pairs = np.array(pairs)
    
    # 全インデックスを集めて距離行列をあらかじめ計算
    all_indices = np.unique(pairs.flatten())
    n_indices = len(all_indices)
    idx_map = {idx: i for i, idx in enumerate(all_indices)}
    
    # 距離行列を事前計算 O(n^2)
    dist_matrix = np.zeros((n_indices, n_indices))
    for i, idx_i in enumerate(all_indices):
        for j, idx_j in enumerate(all_indices):
            dist_matrix[i, j] = xi_between(idx_i, idx_j)
    
    # ペアのインデックスを行列インデックスに変換
    pairs_idx = np.array([[idx_map[i], idx_map[j]] for i, j in pairs])
    
    # ウェイトの積を事前計算
    w = weights[pairs[:, 0]] * weights[pairs[:, 1]]  # shape: (len(pairs),)
    
    # インデックスを抽出
    i_indices = pairs_idx[:, 0]  # shape: (len(pairs),)
    j_indices = pairs_idx[:, 1]  # shape: (len(pairs),)
    
    # ブロードキャスト計算：w_product[a, b] = w[a] * w[b]
    w_broadcast = w[:, np.newaxis]  # shape: (len(pairs), 1)
    w_product = w_broadcast * w[np.newaxis, :]  # shape: (len(pairs), len(pairs))
    
    # 距離の計算をブロードキャスト
    # d_ik[a, b] = dist_matrix[i_indices[a], i_indices[b]]
    d_ik = dist_matrix[i_indices[:, np.newaxis], i_indices[np.newaxis, :]]
    d_jl = dist_matrix[j_indices[:, np.newaxis], j_indices[np.newaxis, :]]
    d_il = dist_matrix[i_indices[:, np.newaxis], j_indices[np.newaxis, :]]
    d_jk = dist_matrix[j_indices[:, np.newaxis], i_indices[np.newaxis, :]]
    
    # 最終計算
    total = np.sum(w_product * (d_ik * d_jl + d_il * d_jk))
    
    return total


def place_needles_on_paper(N1, L1, N2, L2, grid_width, grid_height, seed=None):
    """
    グリッド上にランダムに針を配置
    
    Args:
        N1: 長さL1の針の本数
        L1: L1の針の長さ（グリッドセル単位）
        N2: 長さL2の針の本数
        L2: L2の針の長さ（グリッドセル単位）
        grid_width: グリッドの幅（セル数）
        grid_height: グリッドの高さ（セル数）
        seed: 乱数シード（再現性のため）
    
    Returns:
        dict: 以下のキーを含む
            - 'needles_L1': (N1, 2) 配列、各行は [開始端点インデックス, 終了端点インデックス]
            - 'needles_L2': (N2, 2) 配列、各行は [開始端点インデックス, 終了端点インデックス]
            - 'centers_L1': (N1, 2) 配列、各針のグリッド座標 (x, y)
            - 'centers_L2': (N2, 2) 配列、各針のグリッド座標 (x, y)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 針の中心位置（グリッド座標）と向き角度（0-2π）をランダムに生成
    # N1本のL1の針
    centers_L1 = np.random.uniform(0, [grid_width, grid_height], (N1, 2))
    angles_L1 = np.random.uniform(0, 2 * np.pi, N1)
    
    # N2本のL2の針
    centers_L2 = np.random.uniform(0, [grid_width, grid_height], (N2, 2))
    angles_L2 = np.random.uniform(0, 2 * np.pi, N2)
    
    # 端点の計算（グリッド座標）
    # 針の端点 = center ± (L/2) * (cos(angle), sin(angle))
    offsets_L1 = (L1 / 2) * np.column_stack([np.cos(angles_L1), np.sin(angles_L1)])
    offsets_L2 = (L2 / 2) * np.column_stack([np.cos(angles_L2), np.sin(angles_L2)])
    
    endpoints_L1_start = centers_L1 - offsets_L1  # (N1, 2)
    endpoints_L1_end = centers_L1 + offsets_L1    # (N1, 2)
    
    endpoints_L2_start = centers_L2 - offsets_L2  # (N2, 2)
    endpoints_L2_end = centers_L2 + offsets_L2    # (N2, 2)
    
    # グリッド座標をクリップして有効な範囲内に収める
    endpoints_L1_start = np.clip(endpoints_L1_start, 0, [grid_width - 0.01, grid_height - 0.01])
    endpoints_L1_end = np.clip(endpoints_L1_end, 0, [grid_width - 0.01, grid_height - 0.01])
    endpoints_L2_start = np.clip(endpoints_L2_start, 0, [grid_width - 0.01, grid_height - 0.01])
    endpoints_L2_end = np.clip(endpoints_L2_end, 0, [grid_width - 0.01, grid_height - 0.01])
    
    # グリッド座標を整数インデックスに変換
    indices_L1_start = coords_array_to_index_array(endpoints_L1_start.astype(int), grid_width)
    indices_L1_end = coords_array_to_index_array(endpoints_L1_end.astype(int), grid_width)
    indices_L2_start = coords_array_to_index_array(endpoints_L2_start.astype(int), grid_width)
    indices_L2_end = coords_array_to_index_array(endpoints_L2_end.astype(int), grid_width)
    
    needles_L1 = np.column_stack([indices_L1_start, indices_L1_end])  # (N1, 2)
    needles_L2 = np.column_stack([indices_L2_start, indices_L2_end])  # (N2, 2)
    
    return {
        'needles_L1': needles_L1,       # (N1, 2) - インデックス表現
        'needles_L2': needles_L2,       # (N2, 2) - インデックス表現
        'centers_L1': centers_L1,       # (N1, 2) - グリッド座標
        'centers_L2': centers_L2,       # (N2, 2) - グリッド座標
        'angles_L1': angles_L1,
        'angles_L2': angles_L2,
        'grid_width': grid_width,
        'grid_height': grid_height,
        'L1': L1,
        'L2': L2,
        'N1': N1,
        'N2': N2
    }


def compute_endpoint_distances_vectorized(needle_data):
    """
    すべてのL1の針の端点からすべてのL2の針の端点までの距離を計算
    
    Args:
        needle_data: place_needles_on_paper()の出力辞書
    
    Returns:
        dist_matrix: (N1*2, N2*2) の距離行列
                     dist_matrix[i, j] = i行目の端点からj列目の端点への距離
        stats: 距離の統計情報（最小値、最大値、平均値）
    """
    needles_L1 = needle_data['needles_L1']  # (N1, 2) - インデックス表現
    needles_L2 = needle_data['needles_L2']  # (N2, 2) - インデックス表現
    grid_width = needle_data['grid_width']
    
    N1 = needle_data['N1']
    N2 = needle_data['N2']
    
    # インデックスをグリッド座標に変換
    coords_L1_start = index_array_to_coords_array(needles_L1[:, 0], grid_width)  # (N1, 2)
    coords_L1_end = index_array_to_coords_array(needles_L1[:, 1], grid_width)    # (N1, 2)
    coords_L2_start = index_array_to_coords_array(needles_L2[:, 0], grid_width)  # (N2, 2)
    coords_L2_end = index_array_to_coords_array(needles_L2[:, 1], grid_width)    # (N2, 2)
    
    # すべての端点を集約
    endpoints_L1 = np.vstack([coords_L1_start, coords_L1_end])  # (N1*2, 2)
    endpoints_L2 = np.vstack([coords_L2_start, coords_L2_end])  # (N2*2, 2)
    
    # ブロードキャスト演算で全距離を計算
    diff = endpoints_L1[:, np.newaxis, :] - endpoints_L2[np.newaxis, :, :]  # (N1*2, N2*2, 2)
    
    # ユークリッド距離を計算
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))  # shape: (N1*2, N2*2)
    
    # 統計情報
    stats = {
        'min_distance': np.min(dist_matrix),
        'max_distance': np.max(dist_matrix),
        'mean_distance': np.mean(dist_matrix),
        'std_distance': np.std(dist_matrix),
        'median_distance': np.median(dist_matrix)
    }
    
    return dist_matrix, stats


def compute_endpoint_distances_by_needle(needle_data):
    """
    各L1の針ごとにL2の全端点との距離を計算（構造化版）
    
    Returns:
        distances: (N1, 2, N2*2) 配列
                   distances[i, j, k] = i番目のL1の針のj番目の端点からk番目のL2の端点への距離
    """
    needles_L1 = needle_data['needles_L1']  # (N1, 2) - インデックス表現
    needles_L2 = needle_data['needles_L2']  # (N2, 2) - インデックス表現
    grid_width = needle_data['grid_width']
    
    N1 = needle_data['N1']
    N2 = needle_data['N2']
    
    # インデックスをグリッド座標に変換
    coords_L1_start = index_array_to_coords_array(needles_L1[:, 0], grid_width)  # (N1, 2)
    coords_L1_end = index_array_to_coords_array(needles_L1[:, 1], grid_width)    # (N1, 2)
    coords_L2_start = index_array_to_coords_array(needles_L2[:, 0], grid_width)  # (N2, 2)
    coords_L2_end = index_array_to_coords_array(needles_L2[:, 1], grid_width)    # (N2, 2)
    
    # すべての端点を集約
    endpoints_L1 = np.vstack([coords_L1_start, coords_L1_end])  # (N1*2, 2)
    endpoints_L2 = np.vstack([coords_L2_start, coords_L2_end])  # (N2*2, 2)
    
    # ブロードキャスト計算で全距離を計算
    diff = endpoints_L1[:, np.newaxis, :] - endpoints_L2[np.newaxis, :, :]  # (N1*2, N2*2, 2)
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))  # (N1*2, N2*2)
    
    # (N1*2, N2*2) を (N1, 2, N2*2) に再形成
    distances = dist_matrix.reshape(N1, 2, N2 * 2)  # (N1, 2, N2*2)
    
    return distances

