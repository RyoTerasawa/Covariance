import numpy as np
from scipy.special import legendre_p_all


def get_bin_averaged_Pl(ell, x_edges):
    """
    式 (5.6) に基づき、各ビンの P_bar_l を計算
    legendre_p_all を使用して効率化
    戻り値形状: (n_bins, len(ell))
    """
    x_edges = np.asarray(x_edges)
    ell = np.asarray(ell)
    ell_max = np.max(ell) + 1
    
    # legendre_p_all で P_0 から P_{ell_max+1} を計算
    # 形状: (ell_max+2, n_edges)
    p_all = legendre_p_all(ell_max + 1, x_edges)[0]
    
    # P_{ell+1} と P_{ell-1} をインデックスで取得
    # ell が複数値の場合に対応
    p_lp1 = p_all[ell + 1]  # (len(ell), n_edges)
    p_lm1 = p_all[ell - 1]  # (len(ell), n_edges)
    
    # 差分: [P_{l+1}(x) - P_{l-1}(x)]
    vals = p_lp1 - p_lm1  # (len(ell), n_edges)
    
    # ビンごとの差分をとる（x_edges は逆順なので注意）
    delta_vals = vals[:, :-1] - vals[:, 1:]  # (len(ell), n_bins)
    delta_x = x_edges[:-1] - x_edges[1:]     # (n_bins,)
    
    # 式 (5.6): 分母の (2l+1) と delta_x で割る
    p_bar = delta_vals / ((2 * ell[:, None] + 1) * delta_x)
    return p_bar.T  # (n_bins, len(ell)) に転置

def get_bin_averaged_P2l(ell, x_edges):
    """
    式 (5.8) に基づき、各ビンの G_bar_l,2 を計算
    legendre_p_all を使用して効率化
    戻り値形状: (n_bins, len(ell))
    """
    x_edges = np.asarray(x_edges)
    ell = np.asarray(ell)
    ell_max = np.max(ell) + 1
    
    # legendre_p_all で P_0 から P_{ell_max+1} を計算
    # 形状: (ell_max+2, n_edges)
    p_all = legendre_p_all(ell_max + 1, x_edges)[0] 

    # ルジャンドル多項式と導関数をインデックスで取得
    p_l = p_all[ell]      # (len(ell), n_edges)
    p_lm1 = p_all[ell - 1]
    p_lp1 = p_all[ell + 1]

    # F 関数の各項を計算
    term1 = (ell + 2 / (2 * ell + 1))[:, None] * p_lm1
    term2 = (2 - ell)[:, None] * x_edges * p_l
    term3 = - (2 / (2 * ell + 1))[:, None] * p_lp1
    
    vals = term1 + term2 + term3 
    
    # ビンごとの差分を計算
    delta_vals = vals[:, :-1] - vals[:, 1:]
    delta_x = x_edges[:-1] - x_edges[1:]
    
    return (delta_vals / delta_x).T  # (n_bins, len(ell)) に転置


def get_bin_averaged_G2l(ell, x_edges, sign='+'):
    """
    式 (5.8) に基づき、各ビンの G_bar_l,2 を計算
    legendre_p_all を使用して効率化
    戻り値形状: (n_bins, len(ell))
    """
    x_edges = np.asarray(x_edges)
    ell = np.asarray(ell)
    ell_max = np.max(ell) + 1
    
    # legendre_p_all で P_0 から P_{ell_max+1} を計算
    # 形状: (ell_max+2, n_edges)
    p_all, dp_all = legendre_p_all(ell_max + 1, x_edges, diff_n=1) 

    # ルジャンドル多項式と導関数をインデックスで取得
    p_l = p_all[ell]      # (len(ell), n_edges)
    p_lm1 = p_all[ell - 1]
    p_lp1 = p_all[ell + 1]

    dp_l = dp_all[ell]   # (len(ell), n_edges)
    dp_lm1 = dp_all[ell - 1]
    
    # F 関数の各項を計算
    term1 = -(ell * (ell - 1) / 2.0)[:, None] * (ell[:, None] + 2 / (2 * ell[:, None] + 1)) * p_lm1
    term2 = -(ell * (ell - 1) * (2 - ell) / 2.0)[:, None] * x_edges * p_l
    term3 = (ell * (ell - 1) / (2 * ell + 1))[:, None] * p_lp1
    term4 = (4 - ell[:, None]) * dp_l
    term5 = (ell[:, None] + 2) * (x_edges * dp_lm1 - p_lm1)
    
    s = 1.0 if sign == '+' else -1.0
    term6_7 = s * 2.0 * (((ell - 1) * (x_edges * dp_l - p_l)) - (ell[:, None] + 2) * dp_lm1)
    
    vals = term1 + term2 + term3 + term4 + term5 + term6_7
    
    # ビンごとの差分を計算
    delta_vals = vals[:, :-1] - vals[:, 1:]
    delta_x = x_edges[:-1] - x_edges[1:]
    
    return (delta_vals / delta_x).T  # (n_bins, len(ell)) に転置

# --- メインの書き換えられた関数 ---

def get_legfactors_00_binav(ell_array, theta_edges):
    """
    式 (5.12) を全ビン・全ellに対して一括計算
    引数:
        ell_array: 多重極子の配列 (例: np.arange(ell_max))
        theta_edges: ビン境界の配列 [theta_0, theta_1, ..., theta_N] (ラジアン)
    戻り値:
        P_wl 行列: (n_bins, len(ell_array))
    """
    x_edges = np.cos(theta_edges)
    Pl_bar = get_bin_averaged_Pl(ell_array, x_edges) # (n_bins, n_ell)
    
    factor = (2 * ell_array + 1) / (4 * np.pi)
    return factor * Pl_bar

def get_legfactors_02_binav(ell_array, theta_edges, mode='plus'):
    """
    式 (5.14) を全ビン・全ellに対して一括計算
    """
    x_edges = np.cos(theta_edges)
    sign = '+' if mode == 'plus' else '-'
    
    # ell < 2 の処理（呼び出し前に実行）
    mask = ell_array < 2
    safe_ell = np.where(mask, 1, ell_array) # ゼロ除算回避
    
    P2l_bar = get_bin_averaged_P2l(safe_ell, x_edges) # (n_bins, n_ell)
    
    factor = (2 * safe_ell + 1) / (4 * np.pi * (safe_ell * (safe_ell + 1)))
    result = factor * P2l_bar
    result[:, mask] = 0.0 # l < 2 は 0
    
    return result



def get_legfactors_22_binav(ell_array, theta_edges, mode='plus'):
    """
    式 (5.14) を全ビン・全ellに対して一括計算
    """
    x_edges = np.cos(theta_edges)
    
    # ell < 2 の処理（呼び出し前に実行）
    mask = ell_array < 2
    safe_ell = np.where(mask, 1, ell_array) # ゼロ除算回避
    
    G2l_bar = get_bin_averaged_G2l(safe_ell, x_edges, sign=sign) # (n_bins, n_ell)
    
    factor = (2 * safe_ell + 1) / (2 * np.pi * (safe_ell**2 * (safe_ell + 1)**2))
    result = factor * G2l_bar
    result[:, mask] = 0.0 # l < 2 は 0
    
    return result




