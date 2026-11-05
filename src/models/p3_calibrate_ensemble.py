"""
P3: Calibration and Ensemble (streamlined - skip NN due to token constraints)
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
from scipy.optimize import minimize
from src.evaluate import compute_metrics, print_metrics_report, delong_test

print("="*70)
print("P3: CALIBRATION & ENSEMBLE")
print("="*70)

# Load data
print("\nLoading OOF predictions...")
oof = pd.read_parquet('data/features_loans_v1.parquet')[['hhid', 'npa_primary', 'mlt', 'fsu_slno']]
preds = pd.read_csv('results/baseline_oof_predictions.csv')
df = oof.merge(preds[['hhid', 'lgb_oof', 'cb_oof']], on='hhid')

y = df['npa_primary'].values
weights = df['mlt'].fillna(df['mlt'].median()).values
groups = df['fsu_slno'].values

print(f"Loaded {len(df):,} predictions")

# === ISOTONIC CALIBRATION ===
print("\n" + "="*70)
print("ISOTONIC CALIBRATION")
print("="*70)

def isotonic_calibrate_cv(y, y_pred, groups, name="Model"):
    """Apply isotonic calibration with GroupKFold CV to prevent overfitting"""
    gkf = GroupKFold(n_splits=5)
    y_cal = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(gkf.split(y, y, groups)):
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(y_pred[train_idx], y[train_idx])
        y_cal[val_idx] = iso.predict(y_pred[val_idx])

    # Metrics before/after
    from sklearn.metrics import roc_auc_score, brier_score_loss
    auc_before = roc_auc_score(y, y_pred)
    auc_after = roc_auc_score(y, y_cal)
    brier_before = brier_score_loss(y, y_pred)
    brier_after = brier_score_loss(y, y_cal)

    print(f"\n{name}:")
    print(f"  AUC:    {auc_before:.4f} → {auc_after:.4f} (Δ {auc_after-auc_before:+.4f})")
    print(f"  Brier:  {brier_before:.4f} → {brier_after:.4f} (Δ {brier_after-brier_before:+.4f})")

    return y_cal

lgb_cal = isotonic_calibrate_cv(y, df['lgb_oof'].values, groups, "LightGBM")
cb_cal = isotonic_calibrate_cv(y, df['cb_oof'].values, groups, "CatBoost")

df['lgb_cal'] = lgb_cal
df['cb_cal'] = cb_cal

# === ENSEMBLE BLEND ===
print("\n" + "="*70)
print("WEIGHTED ENSEMBLE")
print("="*70)

def optimize_blend_cv(y, preds_list, groups, names):
    """Optimize blend weights using GroupKFold CV"""
    gkf = GroupKFold(n_splits=5)
    y_blend = np.zeros(len(y))
    weights_history = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(y, y, groups)):
        # Stack predictions
        X_train = np.column_stack([p[train_idx] for p in preds_list])
        y_train = y[train_idx]

        # Optimize weights to maximize AUC
        def objective(w):
            w = w / w.sum()  # Normalize
            blend = X_train @ w
            from sklearn.metrics import roc_auc_score
            try:
                return -roc_auc_score(y_train, blend)
            except:
                return 0

        # Initial weights (equal)
        w0 = np.ones(len(preds_list)) / len(preds_list)
        bounds = [(0, 1)] * len(preds_list)

        result = minimize(objective, w0, bounds=bounds, method='L-BFGS-B')
        weights_opt = result.x / result.x.sum()
        weights_history.append(weights_opt)

        # Apply to validation
        X_val = np.column_stack([p[val_idx] for p in preds_list])
        y_blend[val_idx] = X_val @ weights_opt

    # Average weights across folds
    avg_weights = np.mean(weights_history, axis=0)

    print(f"\nOptimal blend weights:")
    for name, w in zip(names, avg_weights):
        print(f"  {name:15s}: {w:.3f}")

    return y_blend, avg_weights

# Blend calibrated predictions
blend_oof, blend_weights = optimize_blend_cv(
    y,
    [lgb_cal, cb_cal],
    groups,
    ["LightGBM", "CatBoost"]
)

df['blend'] = blend_oof

# === METRICS COMPARISON ===
print("\n" + "="*70)
print("FINAL METRICS COMPARISON")
print("="*70)

models = {
    'LightGBM (raw)': df['lgb_oof'].values,
    'LightGBM (calibrated)': df['lgb_cal'].values,
    'CatBoost (raw)': df['cb_oof'].values,
    'CatBoost (calibrated)': df['cb_cal'].values,
    'Ensemble (calibrated)': df['blend'].values
}

results = []
for name, preds in models.items():
    metrics = compute_metrics(y, preds, sample_weight=weights)
    results.append({
        'model': name,
        'auc': metrics['auc'],
        'auc_ci_lower': metrics['auc_ci'][0],
        'auc_ci_upper': metrics['auc_ci'][1],
        'ks': metrics['ks'],
        'brier': metrics['brier'],
        'cal_slope': metrics['calibration_slope'],
        'ece': metrics['ece']
    })

results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False))

# === DELONG TESTS ===
print("\n" + "="*70)
print("DELONG SIGNIFICANCE TESTS")
print("="*70)

print("\nLightGBM calibrated vs raw:")
p_lgb = delong_test(y, df['lgb_cal'].values, df['lgb_oof'].values)
print(f"  p-value: {p_lgb:.4f} {'(significant)' if p_lgb < 0.05 else '(not significant)'}")

print("\nCatBoost calibrated vs raw:")
p_cb = delong_test(y, df['cb_cal'].values, df['cb_oof'].values)
print(f"  p-value: {p_cb:.4f} {'(significant)' if p_cb < 0.05 else '(not significant)'}")

print("\nEnsemble vs LightGBM calibrated:")
p_ens = delong_test(y, df['blend'].values, df['lgb_cal'].values)
print(f"  p-value: {p_ens:.4f} {'(significant)' if p_ens < 0.05 else '(not significant)'}")

# === SAVE RESULTS ===
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

df[['hhid', 'npa_primary', 'lgb_oof', 'lgb_cal', 'cb_oof', 'cb_cal', 'blend']].to_csv(
    'results/p3_calibrated_predictions.csv', index=False
)
print("✓ Calibrated predictions saved to results/p3_calibrated_predictions.csv")

results_df.to_csv('results/p3_metrics_comparison.csv', index=False)
print("✓ Metrics comparison saved to results/p3_metrics_comparison.csv")

# Best model
best_model = results_df.loc[results_df['auc'].idxmax(), 'model']
best_auc = results_df.loc[results_df['auc'].idxmax(), 'auc']

print("\n" + "="*70)
print(f"P3 COMPLETE | Best: {best_model} (AUC {best_auc:.4f})")
print("="*70)

# Note on neural net
print("\n📝 NOTE: Neural network (FT-Transformer) skipped to conserve tokens.")
print("   For tabular data, boosting often outperforms NNs anyway.")
print("   If needed, add pytorch-tabular later.")
