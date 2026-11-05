"""
P2: Train baseline models and run ablation study
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path
import json

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.baselines import (
    prepare_modeling_data, train_lightgbm, train_catboost,
    WoEScorecard, MultilevelLogistic
)
from src.evaluate import compute_metrics, print_metrics_report
from sklearn.metrics import roc_auc_score


def define_monotone_constraints():
    """Define economically-obvious monotone constraints"""
    return {
        # Increasing default risk
        'interest_rate': 1,
        'debt_to_income': 1,
        'debt_to_assets': 1,
        'informal_debt_share': 1,
        'num_moneylender_loans': 1,
        'loan_age_years': 1,  # Older loans more likely to default

        # Decreasing default risk
        'monthly_consumption': -1,  # Higher income proxy
        'hh_size': 0,  # Ambiguous
        'loan_amount': 0,  # Ambiguous (larger loans could be better screened or riskier)
    }


def run_ablation_study(df, label_col='npa_primary'):
    """
    Feature group ablation study

    Test feature groups:
    1. Loan only
    2. Loan + Borrower
    3. Loan + Borrower + Balance Sheet
    4. Loan + Borrower + Balance Sheet + Portfolio
    5. Full model
    """
    print("\n" + "="*70)
    print("FEATURE GROUP ABLATION STUDY")
    print("="*70)

    feature_groups = {
        'Loan': ['loan_amount', 'loan_outstanding', 'interest_rate', 'loan_age_years',
                 'agency_type', 'loan_scheme', 'tenure_code', 'secured', 'interest_type'],
        'Borrower': ['head_age', 'head_gender', 'head_education', 'hh_size'],
        'Balance Sheet': ['debt_to_income', 'debt_to_assets', 'monthly_consumption', 'total_debt_outstanding'],
        'Portfolio': ['concurrent_loans_count', 'informal_debt_share', 'num_moneylender_loans', 'max_interest_rate'],
        'Context': ['sector', 'state']
    }

    # Filter to available features
    for group in feature_groups:
        feature_groups[group] = [f for f in feature_groups[group] if f in df.columns]

    ablation_results = []

    # Cumulative ablation
    cumulative_features = []
    for i, (group_name, features) in enumerate(feature_groups.items()):
        cumulative_features.extend(features)

        print(f"\n--- Ablation {i+1}: {' + '.join(list(feature_groups.keys())[:i+1])} ---")
        print(f"  Features: {len(cumulative_features)}")

        # Prepare data with only these features
        num_features = [f for f in cumulative_features if f not in ['agency_type', 'loan_scheme', 'tenure_code',
                                                                      'secured', 'interest_type', 'head_gender',
                                                                      'head_education', 'sector', 'state']]
        cat_features = [f for f in cumulative_features if f in ['agency_type', 'loan_scheme', 'tenure_code',
                                                                  'secured', 'interest_type', 'head_gender',
                                                                  'head_education', 'sector', 'state']]

        # Quick LightGBM with fixed params (no Optuna for speed)
        X_num = df[num_features].fillna(0) if num_features else pd.DataFrame()
        X_cat = df[cat_features].fillna('-1').astype(str) if cat_features else pd.DataFrame()

        if len(num_features) == 0 and len(cat_features) == 0:
            print("  No features available, skipping.")
            continue

        X_combined = pd.concat([X_num, X_cat], axis=1)
        for col in cat_features:
            X_combined[col] = X_combined[col].astype('category')

        y = df[label_col].values
        groups = df['fsu_slno'].values
        weights = df['mlt'].fillna(df['mlt'].median()).values

        # Simple train-test split for speed (not full CV)
        from sklearn.model_selection import GroupKFold
        gkf = GroupKFold(n_splits=5)
        train_idx, val_idx = next(gkf.split(X_combined, y, groups))

        X_train, X_val = X_combined.iloc[train_idx], X_combined.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        w_train, w_val = weights[train_idx], weights[val_idx]

        import lightgbm as lgb
        train_data = lgb.Dataset(X_train, label=y_train, weight=w_train, categorical_feature=cat_features)
        val_data = lgb.Dataset(X_val, label=y_val, weight=w_val, reference=train_data, categorical_feature=cat_features)

        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'verbosity': -1,
            'random_state': 42
        }

        model = lgb.train(
            params, train_data, num_boost_round=200,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
        )

        val_auc = model.best_score['valid_0']['auc']
        print(f"  Validation AUC: {val_auc:.4f}")

        ablation_results.append({
            'feature_groups': ' + '.join(list(feature_groups.keys())[:i+1]),
            'num_features': len(cumulative_features),
            'val_auc': val_auc
        })

    ablation_df = pd.DataFrame(ablation_results)
    print("\n" + "="*70)
    print("ABLATION SUMMARY")
    print("="*70)
    print(ablation_df.to_string(index=False))

    # Check for leakage (suspicious jumps)
    if len(ablation_df) > 1:
        ablation_df['auc_change'] = ablation_df['val_auc'].diff()
        max_jump = ablation_df['auc_change'].max()
        print(f"\nMax AUC jump: {max_jump:.4f}")
        if max_jump > 0.10:
            print("⚠️  WARNING: Large AUC jump detected! Check for leakage.")
        else:
            print("✅ No suspicious AUC jumps detected.")

    return ablation_df


def main():
    """Main training script"""
    print("="*70)
    print("P2: BASELINE MODEL TRAINING")
    print("="*70)

    # Load features
    print("\nLoading feature dataset...")
    df = pd.read_parquet('data/features_loans_v1.parquet')
    print(f"Loaded {len(df):,} loans with {len(df.columns)} columns")
    print(f"NPA rate: {df['npa_primary'].mean()*100:.2f}%")

    # Prepare data
    print("\nPreparing modeling data...")
    X_num, X_cat, y, groups, weights, num_features, cat_features = prepare_modeling_data(df)

    # Define monotone constraints
    monotone = define_monotone_constraints()

    # Run ablation study first
    print("\n" + "="*70)
    print("RUNNING ABLATION STUDY")
    print("="*70)
    ablation_df = run_ablation_study(df)
    ablation_df.to_csv('results/ablation_study.csv', index=False)
    print("\n✓ Ablation results saved to results/ablation_study.csv")

    # Train baseline models
    results = {}

    # 1. LightGBM
    print("\n" + "="*70)
    print("TRAINING LIGHTGBM")
    print("="*70)
    lgb_models, lgb_oof, lgb_importance = train_lightgbm(
        X_num, X_cat, y, groups, weights, num_features, cat_features, monotone
    )
    lgb_metrics = compute_metrics(y, lgb_oof, sample_weight=weights)
    print_metrics_report(lgb_metrics, "LightGBM")
    results['lightgbm'] = {
        'oof_preds': lgb_oof,
        'metrics': lgb_metrics,
        'models': lgb_models,
        'importance': lgb_importance
    }

    # 2. CatBoost
    print("\n" + "="*70)
    print("TRAINING CATBOOST")
    print("="*70)
    cb_models, cb_oof, cb_importance = train_catboost(
        X_num, X_cat, y, groups, weights, num_features, cat_features, monotone
    )
    cb_metrics = compute_metrics(y, cb_oof, sample_weight=weights)
    print_metrics_report(cb_metrics, "CatBoost")
    results['catboost'] = {
        'oof_preds': cb_oof,
        'metrics': cb_metrics,
        'models': cb_models,
        'importance': cb_importance
    }

    # 3. WoE Scorecard (if available)
    try:
        from optbinning import OptimalBinning
        print("\n" + "="*70)
        print("TRAINING WoE SCORECARD")
        print("="*70)
        woe_model = WoEScorecard(monotone_features=monotone)
        woe_model.fit(X_num.values, y, num_features)
        woe_oof = woe_model.predict_proba(X_num.values)
        woe_metrics = compute_metrics(y, woe_oof, sample_weight=weights)
        print_metrics_report(woe_metrics, "WoE Scorecard")
        results['woe'] = {
            'oof_preds': woe_oof,
            'metrics': woe_metrics,
            'model': woe_model
        }
    except Exception as e:
        print(f"\nWoE Scorecard skipped: {e}")

    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)

    # Save OOF predictions
    oof_df = pd.DataFrame({
        'hhid': df['hhid'].values,
        'npa_primary': y,
        'lgb_oof': lgb_oof,
        'cb_oof': cb_oof
    })
    if 'woe' in results:
        oof_df['woe_oof'] = results['woe']['oof_preds']

    oof_df.to_csv('results/baseline_oof_predictions.csv', index=False)
    print("✓ OOF predictions saved to results/baseline_oof_predictions.csv")

    # Save feature importance
    lgb_importance.to_csv('results/lgb_feature_importance.csv', index=False)
    cb_importance.to_csv('results/cb_feature_importance.csv', index=False)
    print("✓ Feature importance saved")

    # Save metrics summary
    metrics_summary = []
    for model_name, res in results.items():
        m = res['metrics']
        metrics_summary.append({
            'model': model_name,
            'auc': m['auc'],
            'auc_ci_lower': m['auc_ci'][0],
            'auc_ci_upper': m['auc_ci'][1],
            'gini': m['gini'],
            'ks': m['ks'],
            'brier': m['brier'],
            'calibration_slope': m['calibration_slope'],
            'ece': m['ece']
        })

    metrics_df = pd.DataFrame(metrics_summary)
    metrics_df.to_csv('results/baseline_metrics_summary.csv', index=False)
    print("✓ Metrics summary saved to results/baseline_metrics_summary.csv")

    print("\n" + "="*70)
    print("P2 BASELINE TRAINING COMPLETE")
    print("="*70)

    return results


if __name__ == '__main__':
    results = main()
