"""
P4: Robustness tests - regional CV, sensitivity labels, PSI, fairness
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
import lightgbm as lgb
from src.evaluate import compute_metrics, compute_psi
from sklearn.metrics import roc_auc_score

print("="*70)
print("P4: ROBUSTNESS TESTS")
print("="*70)

# Load data
print("\nLoading data...")
df = pd.read_parquet('data/features_loans_v1.parquet')
labeled = pd.read_parquet('data/labeled_loans_v1.parquet')

# Merge to get sensitivity labels and demographics
df = df.merge(labeled[['hhid', 'npa_writeoff', 'npa_high_outstanding', 'npa_excl_tenure1']], on='hhid', how='left')

# Load household characteristics for fairness
from src.data import NSS77DataLoader
loader = NSS77DataLoader()
hh_char = loader.load_level03()
if 'hhid' in hh_char.columns:
    hh_char['household_key'] = hh_char['hhid'].astype(str).str[:-1]

# Merge demographics (religion, social group)
demo_cols = ['household_key']
if 'b4q3' in hh_char.columns:  # Religion
    demo_cols.append('b4q3')
if 'b4q4' in hh_char.columns:  # Social group
    demo_cols.append('b4q4')

if len(demo_cols) > 1:
    df = df.merge(hh_char[demo_cols], on='household_key', how='left')
    print(f"Merged demographics: {demo_cols}")

print(f"Total records: {len(df):,}")

# Prepare features
num_features = ['loan_amount', 'loan_outstanding', 'interest_rate', 'loan_age_years',
                'head_age', 'hh_size', 'debt_to_income', 'debt_to_assets',
                'concurrent_loans_count', 'informal_debt_share', 'max_interest_rate']
cat_features = ['agency_type', 'loan_scheme', 'tenure_code', 'secured', 'sector']

num_features = [f for f in num_features if f in df.columns]
cat_features = [f for f in cat_features if f in df.columns]

X_num = df[num_features].fillna(0)
X_cat = df[cat_features].fillna('-1').astype(str)
X = pd.concat([X_num, X_cat], axis=1)

for col in cat_features:
    X[col] = X[col].astype('category')

y = df['npa_primary'].values
groups = df['fsu_slno'].values
weights = df['mlt'].fillna(df['mlt'].median()).values
states = df['state'].values

# === 1. LEAVE-ONE-REGION-OUT CV ===
print("\n" + "="*70)
print("1. LEAVE-ONE-REGION-OUT CV (by state)")
print("="*70)

# Group states into regions to avoid too many folds
# Use top 5 states by volume, rest = "Other"
top_states = df['state'].value_counts().head(5).index.tolist()
df['region'] = df['state'].apply(lambda x: x if x in top_states else 999)

regions = df['region'].values
unique_regions = np.unique(regions)
print(f"Regions: {len(unique_regions)} ({unique_regions[:10]}...)")

loro_results = []
loro_preds = np.zeros(len(X))

for region in unique_regions:
    train_idx = (regions != region)
    test_idx = (regions == region)

    if test_idx.sum() < 100:  # Skip tiny regions
        continue

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    w_train, w_test = weights[train_idx], weights[test_idx]

    train_data = lgb.Dataset(X_train, label=y_train, weight=w_train, categorical_feature=cat_features)

    params = {
        'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
        'learning_rate': 0.05, 'verbosity': -1, 'random_state': 42
    }

    model = lgb.train(params, train_data, num_boost_round=150)
    preds = model.predict(X_test)
    loro_preds[test_idx] = preds

    auc = roc_auc_score(y_test, preds)
    loro_results.append({
        'region': region,
        'n_train': train_idx.sum(),
        'n_test': test_idx.sum(),
        'test_auc': auc
    })
    print(f"  Region {region}: n_train={train_idx.sum():,}, n_test={test_idx.sum():,}, AUC={auc:.3f}")

loro_df = pd.DataFrame(loro_results)
overall_auc = roc_auc_score(y[loro_preds > 0], loro_preds[loro_preds > 0])
print(f"\nOverall LORO AUC: {overall_auc:.4f}")
print(f"Min region AUC: {loro_df['test_auc'].min():.4f}")
print(f"Max region AUC: {loro_df['test_auc'].max():.4f}")
print(f"Std across regions: {loro_df['test_auc'].std():.4f}")

# === 2. SENSITIVITY LABELS ===
print("\n" + "="*70)
print("2. SENSITIVITY LABELS")
print("="*70)

# Use existing LGB calibrated predictions
preds_cal = pd.read_csv('results/p3_calibrated_predictions.csv')
df_eval = df.merge(preds_cal[['hhid', 'lgb_cal']], on='hhid', how='left')

sens_results = []

# (a) Writeoff
if 'npa_writeoff' in df_eval.columns:
    mask = df_eval['npa_writeoff'].notna()
    auc_wo = roc_auc_score(df_eval.loc[mask, 'npa_writeoff'], df_eval.loc[mask, 'lgb_cal'])
    sens_results.append({'label': 'writeoff', 'n': mask.sum(), 'prevalence': df_eval.loc[mask, 'npa_writeoff'].mean(), 'auc': auc_wo})
    print(f"  Writeoff: n={mask.sum():,}, prev={df_eval.loc[mask, 'npa_writeoff'].mean():.3f}, AUC={auc_wo:.4f}")

# (c) Exclude tenure 1
if 'npa_excl_tenure1' in df_eval.columns:
    mask = df_eval['npa_excl_tenure1'].notna()
    auc_et = roc_auc_score(df_eval.loc[mask, 'npa_excl_tenure1'], df_eval.loc[mask, 'lgb_cal'])
    sens_results.append({'label': 'excl_tenure1', 'n': mask.sum(), 'prevalence': df_eval.loc[mask, 'npa_excl_tenure1'].mean(), 'auc': auc_et})
    print(f"  Excl tenure 1: n={mask.sum():,}, prev={df_eval.loc[mask, 'npa_excl_tenure1'].mean():.3f}, AUC={auc_et:.4f}")

sens_df = pd.DataFrame(sens_results)

# === 3. WEIGHTED VS UNWEIGHTED ===
print("\n" + "="*70)
print("3. SURVEY-WEIGHTED VS UNWEIGHTED METRICS")
print("="*70)

mask = df_eval['lgb_cal'].notna()
y_eval = df_eval.loc[mask, 'npa_primary'].values
preds_eval = df_eval.loc[mask, 'lgb_cal'].values
w_eval = df_eval.loc[mask, 'mlt'].fillna(df_eval.loc[mask, 'mlt'].median()).values

auc_unweighted = roc_auc_score(y_eval, preds_eval)
auc_weighted = roc_auc_score(y_eval, preds_eval, sample_weight=w_eval)

print(f"  AUC (unweighted): {auc_unweighted:.4f}")
print(f"  AUC (weighted):   {auc_weighted:.4f}")
print(f"  Difference:       {auc_weighted - auc_unweighted:+.4f}")

# === 4. PSI ACROSS REGIONS ===
print("\n" + "="*70)
print("4. PSI (Population Stability Index) ACROSS REGIONS")
print("="*70)

psi_results = []
base_region = unique_regions[0]  # Use first region as baseline
regions_eval = df_eval.loc[mask, 'region'].values
base_preds = preds_eval[regions_eval == base_region]

for region in unique_regions[1:]:
    region_preds = preds_eval[regions_eval == region]
    if len(region_preds) < 50:
        continue
    psi = compute_psi(base_preds, region_preds, n_bins=10)
    psi_results.append({'base': base_region, 'comparison': region, 'psi': psi})
    status = '✅' if psi < 0.10 else ('⚠️' if psi < 0.25 else '❌')
    print(f"  Region {base_region} vs {region}: PSI = {psi:.4f} {status}")

psi_df = pd.DataFrame(psi_results)
if len(psi_df) > 0:
    print(f"\n  Max PSI: {psi_df['psi'].max():.4f}")
    print(f"  Mean PSI: {psi_df['psi'].mean():.4f}")

# === 5. FAIRNESS AUDIT ===
print("\n" + "="*70)
print("5. FAIRNESS AUDIT")
print("="*70)

fairness_results = []

# Helper function
def audit_group(df_eval, group_col, group_name):
    if group_col not in df_eval.columns:
        print(f"  {group_name}: column not found")
        return

    results = []
    for grp in df_eval[group_col].dropna().unique():
        mask = (df_eval[group_col] == grp) & df_eval['lgb_cal'].notna()
        if mask.sum() < 50:
            continue

        y_grp = df_eval.loc[mask, 'npa_primary'].values
        p_grp = df_eval.loc[mask, 'lgb_cal'].values

        auc_grp = roc_auc_score(y_grp, p_grp)
        prev_grp = y_grp.mean()

        # Calibration: observed vs predicted in each group
        cal_slope = np.polyfit(p_grp, y_grp, 1)[0] if len(y_grp) > 10 else np.nan

        results.append({
            'group_name': group_name,
            'group_value': grp,
            'n': mask.sum(),
            'prevalence': prev_grp,
            'auc': auc_grp,
            'cal_slope': cal_slope
        })

    if results:
        res_df = pd.DataFrame(results)
        print(f"\n  {group_name}:")
        print(res_df.to_string(index=False))

        # Equal opportunity gap (TPR at same FPR)
        # Simplified: AUC difference as proxy
        auc_gap = res_df['auc'].max() - res_df['auc'].min()
        print(f"  AUC gap (max-min): {auc_gap:.4f} {'⚠️' if auc_gap > 0.05 else '✅'}")

        fairness_results.extend(results)

# Gender
audit_group(df_eval, 'head_gender', 'Gender')

# Rural/Urban
audit_group(df_eval, 'sector', 'Sector (1=Rural, 2=Urban)')

# Social group (if available)
if 'b4q4' in df_eval.columns:
    audit_group(df_eval, 'b4q4', 'Social Group')

# Religion (if available)
if 'b4q3' in df_eval.columns:
    audit_group(df_eval, 'b4q3', 'Religion')

fairness_df = pd.DataFrame(fairness_results)

# === SAVE RESULTS ===
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

loro_df.to_csv('results/p4_loro_results.csv', index=False)
print("✓ Leave-one-region-out results saved")

if len(sens_df) > 0:
    sens_df.to_csv('results/p4_sensitivity_labels.csv', index=False)
    print("✓ Sensitivity label results saved")

if len(psi_df) > 0:
    psi_df.to_csv('results/p4_psi_results.csv', index=False)
    print("✓ PSI results saved")

if len(fairness_df) > 0:
    fairness_df.to_csv('results/p4_fairness_audit.csv', index=False)
    print("✓ Fairness audit results saved")

# Summary
summary = {
    'loro_auc': overall_auc,
    'loro_auc_std': loro_df['test_auc'].std(),
    'auc_weighted': auc_weighted,
    'auc_unweighted': auc_unweighted,
    'max_psi': psi_df['psi'].max() if len(psi_df) > 0 else np.nan
}

pd.DataFrame([summary]).to_csv('results/p4_summary.csv', index=False)
print("✓ Summary saved")

print("\n" + "="*70)
print("P4 ROBUSTNESS COMPLETE")
print("="*70)
print(f"  LORO AUC: {overall_auc:.4f} (std={loro_df['test_auc'].std():.4f})")
print(f"  Weighted AUC: {auc_weighted:.4f} vs Unweighted: {auc_unweighted:.4f}")
if len(psi_df) > 0:
    print(f"  Max PSI: {psi_df['psi'].max():.4f}")
