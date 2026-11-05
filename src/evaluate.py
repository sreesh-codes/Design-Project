"""
Evaluation metrics for credit risk models
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, precision_recall_curve, auc
from sklearn.calibration import calibration_curve
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def bootstrap_ci(y_true, y_pred, metric_func, n_bootstrap=1000, alpha=0.05, **kwargs):
    """Compute bootstrap confidence interval for a metric"""
    n = len(y_true)
    scores = []

    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        try:
            score = metric_func(y_true[idx], y_pred[idx], **kwargs)
            scores.append(score)
        except:
            continue

    scores = np.array(scores)
    lower = np.percentile(scores, alpha/2 * 100)
    upper = np.percentile(scores, (1 - alpha/2) * 100)
    return lower, upper


def compute_metrics(y_true, y_pred_proba, sample_weight=None):
    """
    Compute comprehensive evaluation metrics with 95% bootstrap CIs

    Returns dict with:
    - AUC, Gini, KS
    - PR-AUC
    - Brier score
    - Calibration slope/intercept, ECE
    - Bad rate by decile
    """
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)

    metrics = {}

    # AUC and Gini
    auc_score = roc_auc_score(y_true, y_pred_proba, sample_weight=sample_weight)
    auc_lower, auc_upper = bootstrap_ci(y_true, y_pred_proba, roc_auc_score)

    metrics['auc'] = auc_score
    metrics['auc_ci'] = (auc_lower, auc_upper)
    metrics['gini'] = 2 * auc_score - 1
    metrics['gini_ci'] = (2 * auc_lower - 1, 2 * auc_upper - 1)

    # KS statistic
    ks_stat = compute_ks(y_true, y_pred_proba)
    metrics['ks'] = ks_stat

    # PR-AUC
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba, sample_weight=sample_weight)
    pr_auc = auc(recall, precision)
    base_rate = y_true.mean()
    metrics['pr_auc'] = pr_auc
    metrics['base_rate'] = base_rate

    # Brier score
    brier = brier_score_loss(y_true, y_pred_proba, sample_weight=sample_weight)
    brier_lower, brier_upper = bootstrap_ci(y_true, y_pred_proba, brier_score_loss)
    metrics['brier'] = brier
    metrics['brier_ci'] = (brier_lower, brier_upper)

    # Calibration slope and intercept (logistic regression of y on logit(p))
    logit_pred = np.log(y_pred_proba / (1 - y_pred_proba + 1e-10) + 1e-10)
    valid_idx = np.isfinite(logit_pred)

    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(penalty=None, max_iter=1000)
    lr.fit(logit_pred[valid_idx].reshape(-1, 1), y_true[valid_idx])

    metrics['calibration_slope'] = lr.coef_[0][0]
    metrics['calibration_intercept'] = lr.intercept_[0]

    # ECE (Expected Calibration Error)
    ece = compute_ece(y_true, y_pred_proba, n_bins=10)
    metrics['ece'] = ece

    # Bad rate by decile
    deciles = pd.qcut(y_pred_proba, q=10, labels=False, duplicates='drop')
    decile_stats = []
    for d in range(int(deciles.max()) + 1):
        mask = (deciles == d)
        if mask.sum() > 0:
            bad_rate = y_true[mask].mean()
            count = mask.sum()
            decile_stats.append({'decile': d, 'count': count, 'bad_rate': bad_rate})

    metrics['decile_stats'] = pd.DataFrame(decile_stats)

    return metrics


def compute_ks(y_true, y_pred_proba):
    """Compute Kolmogorov-Smirnov statistic"""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    ks = np.max(np.abs(tpr - fpr))
    return ks


def compute_ece(y_true, y_pred_proba, n_bins=10):
    """Compute Expected Calibration Error"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_pred_proba[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


def compute_psi(expected_proba, actual_proba, n_bins=10):
    """Compute Population Stability Index"""
    # Bin the probabilities
    bins = np.percentile(expected_proba, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)

    expected_counts = np.histogram(expected_proba, bins=bins)[0]
    actual_counts = np.histogram(actual_proba, bins=bins)[0]

    expected_pct = expected_counts / expected_counts.sum()
    actual_pct = actual_counts / actual_counts.sum()

    # Avoid division by zero
    expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
    actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi


def delong_test(y_true, y_pred1, y_pred2):
    """
    DeLong test for comparing two AUC scores
    Returns p-value
    """
    from scipy.stats import norm

    auc1 = roc_auc_score(y_true, y_pred1)
    auc2 = roc_auc_score(y_true, y_pred2)

    # Simplified DeLong test using bootstrap variance estimation
    n_bootstrap = 1000
    auc1_boot = []
    auc2_boot = []

    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        try:
            auc1_boot.append(roc_auc_score(y_true[idx], y_pred1[idx]))
            auc2_boot.append(roc_auc_score(y_true[idx], y_pred2[idx]))
        except:
            continue

    auc1_boot = np.array(auc1_boot)
    auc2_boot = np.array(auc2_boot)

    # Compute covariance
    diff_boot = auc1_boot - auc2_boot
    se_diff = np.std(diff_boot)

    if se_diff == 0:
        return 1.0

    z = (auc1 - auc2) / se_diff
    p_value = 2 * (1 - norm.cdf(np.abs(z)))

    return p_value


def print_metrics_report(metrics, model_name="Model"):
    """Pretty print metrics report"""
    print(f"\n{'='*70}")
    print(f"{model_name} - EVALUATION METRICS")
    print(f"{'='*70}")

    print(f"\n{'Discrimination Metrics':40s}")
    print(f"  AUC:                  {metrics['auc']:.4f} (95% CI: [{metrics['auc_ci'][0]:.4f}, {metrics['auc_ci'][1]:.4f}])")
    print(f"  Gini:                 {metrics['gini']:.4f} (95% CI: [{metrics['gini_ci'][0]:.4f}, {metrics['gini_ci'][1]:.4f}])")
    print(f"  KS:                   {metrics['ks']:.4f}")
    print(f"  PR-AUC:               {metrics['pr_auc']:.4f} (base rate: {metrics['base_rate']:.4f})")

    print(f"\n{'Calibration Metrics':40s}")
    print(f"  Brier Score:          {metrics['brier']:.4f} (95% CI: [{metrics['brier_ci'][0]:.4f}, {metrics['brier_ci'][1]:.4f}])")
    print(f"  Calibration Slope:    {metrics['calibration_slope']:.4f}")
    print(f"  Calibration Intercept:{metrics['calibration_intercept']:.4f}")
    print(f"  ECE:                  {metrics['ece']:.4f}")

    print(f"\n{'Target Checks':40s}")
    print(f"  AUC ≥ 0.85:           {'✅' if metrics['auc'] >= 0.85 else '❌'}")
    print(f"  KS ≥ 0.45:            {'✅' if metrics['ks'] >= 0.45 else '❌'}")
    print(f"  Cal. slope 0.9-1.1:   {'✅' if 0.9 <= metrics['calibration_slope'] <= 1.1 else '❌'}")
    print(f"  ECE < 0.02:           {'✅' if metrics['ece'] < 0.02 else '❌'}")

    if 'decile_stats' in metrics and len(metrics['decile_stats']) > 0:
        print(f"\n{'Bad Rate by Decile':40s}")
        for _, row in metrics['decile_stats'].iterrows():
            print(f"  Decile {int(row['decile']):2d}: {row['bad_rate']*100:6.2f}% (n={int(row['count']):,})")

    print(f"{'='*70}\n")


if __name__ == '__main__':
    # Test with random data
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.16, 1000)
    y_pred = np.random.beta(2, 5, 1000)

    metrics = compute_metrics(y_true, y_pred)
    print_metrics_report(metrics, "Test Model")
