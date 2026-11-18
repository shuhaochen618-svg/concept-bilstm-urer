"""Evaluation metrics for continuous sentiment regression, calibration, and explainability."""

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.stats import ks_2samp, kstest, pearsonr, spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score


def concordance_correlation_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Lin's Concordance Correlation Coefficient (CCC)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) < 2:
        return 0.0
    mu_t = np.mean(y_true)
    mu_p = np.mean(y_pred)
    var_t = np.var(y_true)
    var_p = np.var(y_pred)
    cov = np.mean((y_true - mu_t) * (y_pred - mu_p))
    denom = var_t + var_p + (mu_t - mu_p) ** 2
    if denom <= 1e-12:
        return 0.0
    return float((2.0 * cov) / denom)


def safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return 0.0
    return float(pearsonr(y_true, y_pred)[0])


def tail_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    low_thresh: Optional[float] = None,
    high_thresh: Optional[float] = None,
    q: float = 0.10,
) -> float:
    """Compute RMSE on the extreme upper and lower tail samples."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if low_thresh is None:
        low_thresh = float(np.quantile(y_true, q))
    if high_thresh is None:
        high_thresh = float(np.quantile(y_true, 1.0 - q))
    mask = (y_true <= low_thresh) | (y_true >= high_thresh)
    if mask.sum() == 0:
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def regression_metrics_va(
    v_true: np.ndarray,
    a_true: np.ndarray,
    v_pred: np.ndarray,
    a_pred: np.ndarray,
    tail_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, float]:
    """Compute official DimABSA2026 Subtask 1 metrics plus anti-collapse diagnostics."""
    v_true = np.asarray(v_true, dtype=np.float64)
    a_true = np.asarray(a_true, dtype=np.float64)
    v_pred = np.asarray(v_pred, dtype=np.float64)
    a_pred = np.asarray(a_pred, dtype=np.float64)

    mse_v = float(np.mean((v_true - v_pred) ** 2))
    mse_a = float(np.mean((a_true - a_pred) ** 2))
    rmse_v = float(np.sqrt(mse_v))
    rmse_a = float(np.sqrt(mse_a))
    # Official DimABSA2026 metrics_subtask_1_2_3.py definition: sqrt(MSE_V + MSE_A)
    rmse_va = float(np.sqrt(mse_v + mse_a))
    rmse_va_norm = float(rmse_va / np.sqrt(128.0))

    mae_v = float(np.mean(np.abs(v_true - v_pred)))
    mae_a = float(np.mean(np.abs(a_true - a_pred)))
    pcc_v = safe_pearson(v_true, v_pred)
    pcc_a = safe_pearson(a_true, a_pred)
    ccc_v = concordance_correlation_coefficient(v_true, v_pred)
    ccc_a = concordance_correlation_coefficient(a_true, a_pred)

    v_low, v_high = tail_bounds["V"] if tail_bounds and "V" in tail_bounds else (None, None)
    a_low, a_high = tail_bounds["A"] if tail_bounds and "A" in tail_bounds else (None, None)
    tail_v = tail_rmse(v_true, v_pred, v_low, v_high)
    tail_a = tail_rmse(a_true, a_pred, a_low, a_high)

    # Std ratio checks variance shrinkage (mean-regression trap diagnostic)
    std_ratio_v = float(np.std(v_pred) / max(np.std(v_true), 1e-8))
    std_ratio_a = float(np.std(a_pred) / max(np.std(a_true), 1e-8))

    return {
        "RMSE_V": rmse_v,
        "RMSE_A": rmse_a,
        "RMSE_VA": rmse_va,
        "RMSE_VA_norm": rmse_va_norm,
        "MAE_V": mae_v,
        "MAE_A": mae_a,
        "PCC_V": pcc_v,
        "PCC_A": pcc_a,
        "CCC_V": ccc_v,
        "CCC_A": ccc_a,
        "Tail20_RMSE_V": tail_v,
        "Tail20_RMSE_A": tail_a,
        "StdRatio_V": std_ratio_v,
        "StdRatio_A": std_ratio_a,
    }


def interval_calibration_metrics(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    uncertainty: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    alpha: float = 0.10,
) -> Dict[str, float]:
    """Compute PICP, MPIW, Winkler interval score, and uncertainty-error rank correlation."""
    y_true = np.asarray(y_true, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)

    covered = (y_true >= lower) & (y_true <= upper)
    picp = float(np.mean(covered))
    width = upper - lower
    mpiw = float(np.mean(width))

    below = np.maximum(0.0, lower - y_true)
    above = np.maximum(0.0, y_true - upper)
    winkler = float(np.mean(width + (2.0 / alpha) * (below + above)))

    out = {"PICP": picp, "MPIW": mpiw, "Winkler": winkler}
    if uncertainty is not None and y_pred is not None:
        err = np.abs(y_true - y_pred)
        if np.std(uncertainty) > 1e-12 and np.std(err) > 1e-12:
            rho, pval = spearmanr(uncertainty, err)
            out["UncErr_Spearman"] = float(rho)
            out["UncErr_p"] = float(pval)
        else:
            out["UncErr_Spearman"] = 0.0
            out["UncErr_p"] = 1.0
    return out


def ood_detection_auroc(in_uncertainty: np.ndarray, ood_uncertainty: np.ndarray) -> float:
    """Compute AUROC for distinguishing OOD/rhetorical shift samples via epistemic uncertainty."""
    y_true = np.concatenate([np.zeros(len(in_uncertainty)), np.ones(len(ood_uncertainty))])
    scores = np.concatenate([in_uncertainty, ood_uncertainty])
    if len(np.unique(y_true)) < 2 or np.std(scores) < 1e-12:
        return 0.5
    return float(roc_auc_score(y_true, scores))


def residual_ks_tests(y_true: np.ndarray, y_pred: np.ndarray, other_residuals: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Run K-S test on standardized residuals (and optional two-sample shift test)."""
    res = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    std = np.std(res)
    z_res = (res - np.mean(res)) / (std if std > 1e-12 else 1.0)
    stat_norm, p_norm = kstest(z_res, "norm")
    out = {
        "res_mean": float(np.mean(res)),
        "res_std": float(std),
        "ks_norm_stat": float(stat_norm),
        "ks_norm_p": float(p_norm),
    }
    if other_residuals is not None:
        stat_2, p_2 = ks_2samp(res, other_residuals)
        out["ks_2samp_stat"] = float(stat_2)
        out["ks_2samp_p"] = float(p_2)
    return out


def paired_wilcoxon(errors_a: np.ndarray, errors_b: np.ndarray) -> Tuple[float, float]:
    """Run paired Wilcoxon signed-rank test on per-sample absolute or squared errors."""
    diff = np.asarray(errors_a) - np.asarray(errors_b)
    if np.all(np.abs(diff) < 1e-12):
        return 0.0, 1.0
    stat, p = wilcoxon(errors_a, errors_b)
    return float(stat), float(p)
