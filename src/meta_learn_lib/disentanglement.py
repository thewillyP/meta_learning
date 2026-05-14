import numpy as np
from sklearn.metrics import mutual_info_score

from meta_learn_lib.config import (
    DisentanglementMetric,
    MIGMetric,
    ModularityMetric,
    TCMetric,
)


def bin_array(x: np.ndarray, n_bins: int) -> np.ndarray:
    out = np.empty_like(x, dtype=np.int64)
    for j in range(x.shape[1]):
        col = x[:, j]
        lo, hi = float(col.min()), float(col.max())
        if hi <= lo:
            out[:, j] = 0
            continue
        edges = np.linspace(lo, hi, n_bins + 1)
        edges[-1] = np.nextafter(edges[-1], np.inf)
        out[:, j] = np.clip(np.digitize(col, edges) - 1, 0, n_bins - 1)
    return out


def entropy_of_counts(counts: np.ndarray) -> float:
    p = counts.astype(np.float64) / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def mi_matrix(latent_binned: np.ndarray, factors_binned: np.ndarray) -> np.ndarray:
    D = latent_binned.shape[1]
    K = factors_binned.shape[1]
    M = np.zeros((D, K), dtype=np.float64)
    for j in range(D):
        for k in range(K):
            M[j, k] = mutual_info_score(latent_binned[:, j], factors_binned[:, k])
    return M


def factor_entropies(factors_binned: np.ndarray, n_bins: int) -> np.ndarray:
    K = factors_binned.shape[1]
    out = np.zeros(K, dtype=np.float64)
    for k in range(K):
        counts = np.bincount(factors_binned[:, k], minlength=n_bins)
        out[k] = entropy_of_counts(counts)
    return out


def metric_name(metric: DisentanglementMetric) -> str:
    match metric:
        case MIGMetric():
            return "mig"
        case ModularityMetric():
            return "modularity"
        case TCMetric():
            return "tc"


def compute_metric(
    metric: DisentanglementMetric,
    latent_mu: np.ndarray,
    factors: np.ndarray,
) -> float:
    match metric:
        case MIGMetric(n_bins):
            lb = bin_array(latent_mu, n_bins)
            fb = bin_array(factors, n_bins)
            M = mi_matrix(lb, fb)
            H = factor_entropies(fb, n_bins)
            sorted_M = np.sort(M, axis=0)[::-1]
            gaps = sorted_M[0] - sorted_M[1]
            safe_H = np.where(H > 0, H, 1.0)
            return float((gaps / safe_H).mean())
        case ModularityMetric(n_bins):
            lb = bin_array(latent_mu, n_bins)
            fb = bin_array(factors, n_bins)
            M = mi_matrix(lb, fb)
            D, K = M.shape
            if K < 2:
                return np.nan
            theta = M.max(axis=1)
            argmax_idx = M.argmax(axis=1)
            t = np.zeros_like(M)
            t[np.arange(D), argmax_idx] = theta
            diff_sq = ((M - t) ** 2).sum(axis=1)
            safe_theta_sq = np.where(theta > 0, theta**2, 1.0)
            delta = diff_sq / (safe_theta_sq * (K - 1))
            return float((1.0 - delta).mean())
        case TCMetric():
            if latent_mu.shape[1] < 2:
                return np.nan
            cov = np.cov(latent_mu.T)
            diag = np.diag(cov)
            sign, logdet = np.linalg.slogdet(cov)
            if sign <= 0 or np.any(diag <= 0):
                return np.nan
            return float(0.5 * (np.log(diag).sum() - logdet))
