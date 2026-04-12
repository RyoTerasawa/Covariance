import numpy as np
from scipy.special import eval_legendre, lpn

def calculate_real_space_covariance(cov_cl, weight_p_xi, weight_p_theta):
    """
    論文の式 (5.11) に基づき、実空間の角二点相関関数の共分散を計算します。
    
    引数:
        cov_cl (np.ndarray): 
            角パワースペクトルの共分散行列 Cov(C_ij^Xi(l), C_km^Theta(l'))。
            形状は (len(ell), len(ell))。
        weight_p_xi (np.ndarray): 
            観測量 Xi に対するビン平均ウェイト関数 P_{Xi, l} (式 5.12-5.14)。
            形状は (n_theta_bins, len(ell))。
        weight_p_theta (np.ndarray): 
            観測量 Theta に対するビン平均ウェイト関数 P_{Theta, l'} (式 5.12-5.14)。
            形状は (n_theta_prime_bins, len(ell))。
            
    戻り値:
        np.ndarray: 
            実空間での共分散行列 Cov(Xi(theta), Theta(theta'))。
            形状は (n_theta_bins, n_theta_prime_bins)。
    """
    
    # 式 (5.11): Cov = sum_l P_{Xi, l} * sum_l' P_{Theta, l'} * Cov(Cl, Cl')
    # 行列積を用いることで、二重和を効率的に計算できます。
    # 計算式: P_xi * Cov_Cl * P_theta^T
    cov_real_space = weight_p_xi @ cov_cl @ weight_p_theta.T
    
    return cov_real_space




def legendre_derivative_old(l, x):
    """
    Calculate the derivative of the Legendre polynomial P_l(x) with respect to x.
    """
    
    l = np.asarray(l)
    x = np.asarray(x)

    eps = 1e-12
    denom = x**2 - 1.0 + eps

    # Use recurrence relation for the derivative of Legendre polynomials
    # dP_l(x)/dx = l/(x^2 - 1) * (x P_l(x) - P_{l-1}(x))
    deriv = l * (x * eval_legendre(l, x) - eval_legendre(l - 1, x)) / denom

    mask_x = np.abs(1.0 - x**2) < eps
    if np.any(mask_x):
        deriv = np.where(mask_x, l * (l + 1) / 2.0, deriv)

    deriv = np.where(l == 0, 0.0, deriv)

    return deriv


def legendre_derivative(l, x):
    """
    Calculate the derivative of the Legendre polynomial P_l(x) with respect to x.
    """
    
    l = np.asarray(l)
    x = np.asarray(x)

    deriv = lpn(l[-1] + 1, x)[1][1:]

    return deriv

def get_bin_averaged_p_bar(ell, x_edges):
    """
    式 (5.6) に基づき、各ビンの P_bar_l を計算
    戻り値形状: (n_bins, len(ell))
    """
    # 各境界での [P_{l+1}(x) - P_{l-1}(x)] を計算
    # 形状: (n_edges, len(ell))
    vals = (eval_legendre(ell + 1, x_edges[:, None]) - 
            eval_legendre(ell - 1, x_edges[:, None]))
    
    # 隣り合う境界の差分をとる
    delta_vals = vals[:-1, :] - vals[1:, :] # (n_bins, len(ell))
    delta_x = x_edges[:-1] - x_edges[1:]    # (n_bins,)
    
    # 式 (5.6): 分母の (2l+1) と delta_x で割る
    p_bar = delta_vals / ((2 * ell + 1) * delta_x[:, None])
    return p_bar

def get_bin_averaged_g_bar(ell, x_edges, sign='+'):
    """
    式 (5.8) に基づき、各ビンの G_bar_l,2 を計算
    戻り値形状: (n_bins, len(ell))
    """
    def F_func(l, x, mode='+'):
        p_l = eval_legendre(l, x)
        p_lm1 = eval_legendre(l - 1, x)
        p_lp1 = eval_legendre(l + 1, x)
        dp_l = legendre_derivative(l, x)
        dp_lm1 = legendre_derivative(l - 1, x)
        
        term1 = - (l * (l - 1) / 2.0) * (l + 2 / (2 * l + 1)) * p_lm1
        term2 = - (l * (l - 1) * (2 - l) / 2.0) * x * p_l
        term3 = (l * (l - 1) / (2 * l + 1)) * p_lp1
        term4 = (4 - l) * dp_l
        term5 = (l + 2) * (x * dp_lm1 - p_lm1)
        s = 1.0 if mode == '+' else -1.0
        term6_7 = s * 2.0 * ((l - 1) * (x * dp_l - p_l) - (l + 2) * dp_lm1)
        return term1 + term2 + term3 + term4 + term5 + term6_7

    # 各境界での F(x) を評価
    vals = F_func(ell, x_edges[:, None], mode=sign) # (n_edges, len(ell))
    
    # ビンごとの差分を計算
    delta_vals = vals[:-1, :] - vals[1:, :]
    delta_x = x_edges[:-1] - x_edges[1:]
    
    return delta_vals / delta_x[:, None]

# --- メインの書き換えられた関数 ---

def calculate_pw_l_bins(ell_array, theta_edges):
    """
    式 (5.12) を全ビン・全ellに対して一括計算
    引数:
        ell_array: 多重極子の配列 (例: np.arange(ell_max))
        theta_edges: ビン境界の配列 [theta_0, theta_1, ..., theta_N] (ラジアン)
    戻り値:
        P_wl 行列: (n_bins, len(ell_array))
    """
    x_edges = np.cos(theta_edges)
    p_bar = get_bin_averaged_p_bar(ell_array, x_edges) # (n_bins, n_ell)
    
    factor = (2 * ell_array + 1) / (4 * np.pi)
    return factor * p_bar

def calculate_pxi_pm_l_bins(ell_array, theta_edges, mode='plus'):
    """
    式 (5.14) を全ビン・全ellに対して一括計算
    """
    x_edges = np.cos(theta_edges)
    sign = '+' if mode == 'plus' else '-'
    g_bar = get_bin_averaged_g_bar(ell_array, x_edges, sign=sign) # (n_bins, n_ell)
    
    # ell < 2 の処理
    mask = ell_array < 2
    safe_ell = np.where(mask, 1, ell_array) # ゼロ除算回避
    
    factor = (2 * safe_ell + 1) / (2 * np.pi * (safe_ell**2 * (safe_ell + 1)**2))
    result = factor * g_bar
    result[:, mask] = 0.0 # l < 2 は 0
    
    return result




