"""
P2: Fast baseline training - LightGBM and CatBoost only
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import catboost as cb

print("="*70)
print("P2: FAST BASELINE TRAINING")
print("="*70)

# Load data
print("\nLoading data...")
df = pd.read_parquet('data/features_loans_v1.parquet')
print(f"Loaded {len(df):,} loans, NPA rate: {df['npa_primary'].mean()*100:.2f}%")

# Features
num_features = ['loan_amount', 'loan_outstanding', 'interest_rate', 'loan_age_years',
                'head_age', 'hh_size', 'debt_to_income', 'debt_to_assets',
                'concurrent_loans_count', 'informal_debt_share', 'max_interest_rate']
cat_features = ['agency_type', 'loan_scheme', 'tenure_code', 'secured', 'sector']

# Filter available
num_features = [f for f in num_features if f in df.columns]
cat_features = [f for f in cat_features if f in df.columns]

# Prepare X
X_num = df[num_features].fillna(0)
X_cat = df[cat_features].fillna('-1').astype(str)
X = pd.concat([X_num, X_cat], axis=1)

for col in cat_features:
    X[col] = X[col].astype('category')

y = df['npa_primary'].values
groups = df['fsu_slno'].values
weights = df['mlt'].fillna(df['mlt'].median()).values

print(f"Features: {len(num_features)} num + {len(cat_features)} cat = {len(X.columns)}")

# GroupKFold 5-fold CV
gkf = GroupKFold(n_splits=5)
lgb_oof = np.zeros(len(X))
cb_oof = np.zeros(len(X))

print("\n" + "="*70)
print("LIGHTGBM (5-FOLD GROUPKFOLD)")
print("="*70)

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    print(f"\nFold {fold+1}/5...")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    w_train, w_val = weights[train_idx], weights[val_idx]

    train_data = lgb.Dataset(X_train, label=y_train, weight=w_train, categorical_feature=cat_features)
    val_data = lgb.Dataset(X_val, label=y_val, weight=w_val, reference=train_data, categorical_feature=cat_features)

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbosity': -1,
        'random_state': 42
    }

    model = lgb.train(params, train_data, num_boost_round=300, valid_sets=[val_data],
                      callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])

    lgb_oof[val_idx] = model.predict(X_val)
    print(f"  Val AUC: {model.best_score['valid_0']['auc']:.4f}")

lgb_auc = roc_auc_score(y, lgb_oof)
print(f"\nLightGBM OOF AUC: {lgb_auc:.4f}")

print("\n" + "="*70)
print("CATBOOST (5-FOLD GROUPKFOLD)")
print("="*70)

cat_indices = list(range(len(num_features), len(X.columns)))

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    print(f"\nFold {fold+1}/5...")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    w_train, w_val = weights[train_idx], weights[val_idx]

    train_pool = cb.Pool(X_train, y_train, cat_features=cat_indices, weight=w_train)
    val_pool = cb.Pool(X_val, y_val, cat_features=cat_indices, weight=w_val)

    model = cb.CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        loss_function='Logloss',
        eval_metric='AUC',
        early_stopping_rounds=30,
        random_seed=42,
        verbose=0
    )

    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    cb_oof[val_idx] = model.predict_proba(X_val)[:, 1]
    print(f"  Val AUC: {model.get_best_score()['validation']['AUC']:.4f}")

cb_auc = roc_auc_score(y, cb_oof)
print(f"\nCatBoost OOF AUC: {cb_auc:.4f}")

# Save results
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

oof_df = pd.DataFrame({
    'hhid': df['hhid'].values,
    'npa_primary': y,
    'lgb_oof': lgb_oof,
    'cb_oof': cb_oof
})
oof_df.to_csv('results/baseline_oof_predictions.csv', index=False)
print("✓ OOF predictions saved")

# Metrics summary
summary = pd.DataFrame({
    'model': ['LightGBM', 'CatBoost'],
    'oof_auc': [lgb_auc, cb_auc]
})
summary.to_csv('results/baseline_metrics_summary.csv', index=False)
print("✓ Metrics summary saved")

print("\n" + "="*70)
print(f"P2 BASELINE COMPLETE | LightGBM: {lgb_auc:.4f} | CatBoost: {cb_auc:.4f}")
print("="*70)
