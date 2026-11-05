"""
P2: Baseline models - WoE scorecard, multilevel logistic, LightGBM, CatBoost
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import catboost as cb

# Optional: optbinning for WoE
try:
    from optbinning import OptimalBinning, Scorecard
    HAS_OPTBINNING = True
except ImportError:
    HAS_OPTBINNING = False
    print("Warning: optbinning not installed. WoE scorecard will be skipped.")

# Optional: statsmodels for multilevel
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not installed. Multilevel logistic will be skipped.")


def prepare_modeling_data(df, label_col='npa_primary'):
    """
    Prepare data for modeling

    Returns:
    - X_num: numerical features
    - X_cat: categorical features
    - y: labels
    - groups: FSU groups for CV
    - feature_names
    """
    # Define feature groups
    numerical_features = [
        'loan_amount', 'loan_outstanding', 'interest_rate', 'loan_age_years', 'exposure_months',
        'head_age', 'hh_size',
        'total_debt_outstanding', 'monthly_consumption', 'debt_to_income', 'debt_to_assets',
        'concurrent_loans_count', 'num_informal_loans', 'num_moneylender_loans',
        'informal_debt_share', 'max_interest_rate'
    ]

    categorical_features = [
        'agency_type', 'loan_scheme', 'tenure_code', 'interest_type',
        'purpose_code', 'secured', 'sector', 'head_gender', 'head_education'
    ]

    # Filter to available features
    numerical_features = [f for f in numerical_features if f in df.columns]
    categorical_features = [f for f in categorical_features if f in df.columns]

    print(f"Numerical features: {len(numerical_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    # Prepare numerical
    X_num = df[numerical_features].copy()
    X_num = X_num.fillna(X_num.median())

    # Prepare categorical
    X_cat = df[categorical_features].copy()
    for col in categorical_features:
        X_cat[col] = X_cat[col].fillna(-1).astype(str)

    # Labels
    y = df[label_col].values

    # Groups (FSU for GroupKFold)
    groups = df['fsu_slno'].values

    # Weights (survey multiplier)
    weights = df['mlt'].fillna(df['mlt'].median()).values

    return X_num, X_cat, y, groups, weights, numerical_features, categorical_features


class WoEScorecard:
    """Weight of Evidence Scorecard using OptimalBinning"""

    def __init__(self, monotone_features=None):
        self.monotone_features = monotone_features or {}
        self.binners = {}
        self.model = None

    def fit(self, X, y, feature_names):
        """Fit WoE binning and logistic regression"""
        if not HAS_OPTBINNING:
            print("OptimalBinning not available. Skipping WoE scorecard.")
            return self

        X_woe = X.copy()

        # Bin each numerical feature
        for i, fname in enumerate(feature_names):
            monotonic_trend = self.monotone_features.get(fname, None)

            try:
                binner = OptimalBinning(
                    name=fname,
                    dtype='numerical',
                    monotonic_trend=monotonic_trend,
                    min_prebin_size=0.05,
                    min_bin_size=0.05,
                    max_n_prebins=20,
                    max_n_bins=10
                )

                binner.fit(X[:, i], y)
                X_woe[:, i] = binner.transform(X[:, i], metric='woe')
                self.binners[fname] = binner
            except Exception as e:
                print(f"  Warning: Could not bin {fname}: {e}")
                X_woe[:, i] = X[:, i]

        # Fit logistic regression on WoE-transformed features
        self.model = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42)
        self.model.fit(X_woe, y)

        print(f"  WoE Scorecard fitted with {len(self.binners)} binned features")
        return self

    def predict_proba(self, X):
        """Predict probabilities"""
        if self.model is None:
            raise ValueError("Model not fitted")

        X_woe = X.copy()
        for i, fname in enumerate(self.binners.keys()):
            if fname in self.binners:
                X_woe[:, i] = self.binners[fname].transform(X[:, i], metric='woe')

        return self.model.predict_proba(X_woe)[:, 1]


class MultilevelLogistic:
    """Multilevel logistic regression with random intercepts"""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def fit(self, X, y, groups, feature_names):
        """Fit multilevel logistic using statsmodels"""
        if not HAS_STATSMODELS:
            print("Statsmodels not available. Skipping multilevel logistic.")
            return self

        # Prepare data
        df = pd.DataFrame(X, columns=feature_names)
        df['y'] = y
        df['fsu'] = groups

        # Scale features
        df[feature_names] = self.scaler.fit_transform(df[feature_names])

        # Build formula (use a subset of features to avoid convergence issues)
        top_features = feature_names[:10]  # Use top 10 for speed
        formula = f"y ~ {' + '.join(top_features)}"

        try:
            # Mixed effects logit (requires MixedLM, approximate with GEE)
            # Note: True mixed effects logit is complex; using GEE as proxy
            from statsmodels.genmod.generalized_estimating_equations import GEE
            from statsmodels.genmod.families import Binomial

            self.model = GEE.from_formula(
                formula,
                groups=df['fsu'],
                data=df,
                family=Binomial()
            )
            self.model = self.model.fit()

            print(f"  Multilevel model fitted (GEE proxy)")
        except Exception as e:
            print(f"  Warning: Multilevel model failed: {e}")
            # Fallback to standard logistic
            self.model = LogisticRegression(max_iter=1000, random_state=42)
            self.model.fit(X, y)
            print(f"  Fallback: Standard logistic regression fitted")

        return self

    def predict_proba(self, X):
        """Predict probabilities"""
        if self.model is None:
            raise ValueError("Model not fitted")

        if hasattr(self.model, 'predict'):
            # GEE or sklearn model
            if hasattr(self.model, 'exog_names'):  # GEE
                df = pd.DataFrame(X, columns=self.scaler.feature_names_in_)
                df[self.scaler.feature_names_in_] = self.scaler.transform(df[self.scaler.feature_names_in_])
                return self.model.predict(df)
            else:  # sklearn
                return self.model.predict_proba(X)[:, 1]
        else:
            raise ValueError("Model cannot predict")


def train_lightgbm(X_num, X_cat, y, groups, weights, num_features, cat_features, monotone_constraints=None):
    """
    Train LightGBM with GroupKFold CV and Optuna tuning

    Returns:
    - models: list of fitted models (one per fold)
    - oof_preds: out-of-fold predictions
    - feature_importance: aggregated feature importance
    """
    print("\n=== Training LightGBM ===")

    # Combine numerical and categorical
    X = pd.concat([X_num, X_cat], axis=1)
    feature_names = list(num_features) + list(cat_features)

    # Encode categoricals
    for col in cat_features:
        X[col] = X[col].astype('category')

    # Monotone constraints (map feature name to +1, -1, or 0)
    if monotone_constraints is None:
        monotone_constraints = {}

    constraint_list = [monotone_constraints.get(f, 0) for f in feature_names]

    # GroupKFold CV
    gkf = GroupKFold(n_splits=5)
    models = []
    oof_preds = np.zeros(len(X))

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"\n  Fold {fold + 1}/5")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        w_train, w_val = weights[train_idx], weights[val_idx]

        train_data = lgb.Dataset(
            X_train, label=y_train, weight=w_train,
            categorical_feature=cat_features
        )
        val_data = lgb.Dataset(
            X_val, label=y_val, weight=w_val, reference=train_data,
            categorical_feature=cat_features
        )

        # Fixed hyperparameters (Optuna tuning would go here)
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'monotone_constraints': constraint_list,
            'verbosity': -1,
            'random_state': 42
        }

        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )

        oof_preds[val_idx] = model.predict(X_val)
        models.append(model)

        print(f"    Best iteration: {model.best_iteration}, Val AUC: {model.best_score['valid_0']['auc']:.4f}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': np.mean([m.feature_importance(importance_type='gain') for m in models], axis=0)
    }).sort_values('importance', ascending=False)

    print(f"\n  OOF AUC: {roc_auc_score(y, oof_preds):.4f}")

    return models, oof_preds, feature_importance


def train_catboost(X_num, X_cat, y, groups, weights, num_features, cat_features, monotone_constraints=None):
    """Train CatBoost with GroupKFold CV"""
    print("\n=== Training CatBoost ===")

    # Combine features
    X = pd.concat([X_num, X_cat], axis=1)

    # CatBoost expects categorical indices
    cat_indices = [i + len(num_features) for i in range(len(cat_features))]

    # Monotone constraints (CatBoost uses dict: feature_idx -> direction)
    if monotone_constraints is None:
        monotone_constraints = {}

    monotone_dict = {}
    for i, fname in enumerate(num_features):
        if fname in monotone_constraints:
            monotone_dict[i] = monotone_constraints[fname]

    # GroupKFold CV
    gkf = GroupKFold(n_splits=5)
    models = []
    oof_preds = np.zeros(len(X))

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"\n  Fold {fold + 1}/5")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        w_train, w_val = weights[train_idx], weights[val_idx]

        train_pool = cb.Pool(X_train, y_train, cat_features=cat_indices, weight=w_train)
        val_pool = cb.Pool(X_val, y_val, cat_features=cat_indices, weight=w_val)

        model = cb.CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3,
            loss_function='Logloss',
            eval_metric='AUC',
            monotone_constraints=monotone_dict if monotone_dict else None,
            early_stopping_rounds=50,
            random_seed=42,
            verbose=0
        )

        model.fit(train_pool, eval_set=val_pool, use_best_model=True)

        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        models.append(model)

        print(f"    Best iteration: {model.get_best_iteration()}, Val AUC: {model.get_best_score()['validation']['AUC']:.4f}")

    # Feature importance
    feature_names = list(num_features) + list(cat_features)
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': np.mean([m.get_feature_importance() for m in models], axis=0)
    }).sort_values('importance', ascending=False)

    print(f"\n  OOF AUC: {roc_auc_score(y, oof_preds):.4f}")

    return models, oof_preds, feature_importance


from sklearn.metrics import roc_auc_score


if __name__ == '__main__':
    print("Baselines module loaded. Use in main training script.")
