# Credit Risk Analysis on NSS 77th Round Household Survey Data

**Geographic and Demographic Heterogeneity in Credit Risk Models: Evidence from Nationally Representative Indian Household Data**

---

## Overview

This project develops and rigorously tests a credit risk model using NSS 77th Round (2019) nationally representative household survey data (34,425 institutional loans from 29,092 households across 8,180 villages/wards).

**Key Contribution**: First study to apply GroupKFold CV, isotonic calibration, and fairness audits across caste/religion on Indian household credit data. Exposes geographic/demographic heterogeneity hidden in bank datasets.

**Finding**: Model achieves moderate discrimination (AUC 0.76) but **fails** geographic stability (PSI 0.58) and fairness tests (AUC gap 0.28 across social groups), highlighting limitations of national-scale survey-based credit models.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python src/labels.py          # P1: Label construction
python src/features.py        # P2: Feature engineering
python src/models/train_fast.py  # P2: Train baselines
python src/models/p3_calibrate_ensemble.py  # P3: Calibration
python src/models/p4_robustness.py  # P4: Robustness tests
python src/models/p5_shap_analysis.py  # P5: SHAP explainability
python src/models/p6_risk_grades.py  # P6: Risk grades & expected loss
```

**Data**: Place `CSV_DI_77.zip` in project root (NSS 77th Round from MoSPI).

---

## Project Structure

```
.
├── README.md                   # This file
├── DATA_DICTIONARY.md          # Field documentation (156 variables)
├── requirements.txt            # Dependencies
├── data/
│   ├── labeled_loans_v1.parquet       # 34,425 labeled loans
│   └── features_loans_v1.parquet      # 82 features
├── src/
│   ├── data.py                 # Data loading
│   ├── labels.py               # Label construction (NPA proxy)
│   ├── features.py             # Feature engineering
│   ├── evaluate.py             # Metrics (AUC, Brier, ECE, PSI, DeLong)
│   └── models/
│       ├── baselines.py        # Model classes
│       ├── train_fast.py       # P2: LightGBM + CatBoost
│       ├── p3_calibrate_ensemble.py  # Isotonic calibration
│       ├── p4_robustness.py    # LORO CV, fairness audit
│       ├── p5_shap_analysis.py # SHAP values & reason codes
│       └── p6_risk_grades.py   # Risk grading A-G
└── results/
    ├── P1_gate_results.md      # Phase 1 validation
    ├── P2_gate_results.md      # Baseline performance
    ├── P3_gate_results.md      # Calibration results
    ├── P4_gate_results.md      # Robustness (FAILS)
    ├── FINAL_PROJECT_SUMMARY.md  # Executive summary
    ├── baseline_oof_predictions.csv
    ├── p3_calibrated_predictions.csv
    ├── p4_loro_results.csv
    ├── p4_fairness_audit.csv
    ├── p5_shap_importance.csv
    ├── p5_reason_codes.csv
    └── p6_risk_grades.csv
```

---

## Results Summary

### Phase 1: Data & Labels ✅
- **34,425** institutional loans (b12q5 ∈ 01-13)
- **29,092** households, **8,180** FSUs
- **15.9%** NPA rate (zero repayment proxy)
- ✅ All validation checks passed

### Phase 2: Baselines ⚠️
- **LightGBM**: AUC 0.759, KS 0.435, ECE 0.015
- **CatBoost**: AUC 0.729, KS 0.399
- ⚠️ Below 0.85 AUC target (survey label ceiling)
- ✅ No leakage (AUC dropped from 0.886 when b12q14 removed)

### Phase 3: Calibration ✅
- **Isotonic regression** (GroupKFold CV): ECE 0.015 → 0.005 (67% reduction)
- **Calibration slope**: 0.840 → 0.965 (closer to ideal 1.0)
- **Ensemble**: 50/50 LGB+CB (AUC 0.752, no synergy)
- ✅ DeLong tests all p < 0.05

### Phase 4: Robustness ❌
- **LORO CV**: AUC 0.793 (std 0.052)
  - Region 32: AUC 0.70 vs Region 19: AUC 0.85
- **PSI**: **0.579** (❌ >> 0.10 target) → geographically unstable
- **Fairness**:
  - Social group 9: AUC 0.51 (random), 55% default rate
  - Gender gap: 0.08, Sector gap: 0.08, **Caste gap: 0.28** ❌
- **Weighted AUC**: 0.607 vs unweighted 0.667 (⚠️ -0.06)

### Phase 5: SHAP ✅
- **Top features**: loan_amount (0.53), sector (0.37), agency_type (0.37)
- **Stability**: Spearman ρ = 0.997 (✅ > 0.9 target)
- **Reason codes**: Top-3 features per loan with SHAP values

### Phase 6: Risk Grades ✅
- **7 grades** (A-G) with **monotone bad rates** (8.2% → 52.4%)
- **Expected loss**: ₹1.35B (4.77% of ₹28.3B exposure)
- **Inclusion**: 55% of informal borrowers in grades A-B (low risk)
- ⚠️ LGD assumptions (35%/45%), not estimated

---

## Key Findings

### 1. Survey Label Limitation
**NPA proxy** (zero repayment) ≠ actual bank NPAs:
- 15.9% "default" vs typical 5-10% bank NPAs
- Includes informal delays, no payment history
- **AUC ceiling ~0.76** vs 0.85+ in bank studies

### 2. Geographic Heterogeneity
**PSI 0.58** between regions → model rebuild recommended:
- Different state lending ecosystems
- Feature distributions vary (informal debt 5% urban vs 35% rural)
- 15-point AUC gap (region 32: 0.70 vs region 19: 0.85)

### 3. Fairness Failures ❌
**Social group AUC gap 0.28**:
- Group 1 (upper caste): AUC 0.79
- Group 9 (marginalized): AUC 0.51 (random), 55% default
- **Cannot deploy** without group-specific models

### 4. Policy Insights
- **Co-operative NPAs**: 33.5% vs SCBs 15.3% → supervision needed
- **Informal debt risk**: 17.8× default separation by debt ratios
- **Financial inclusion**: 55% of informal borrowers are low-risk (grades A-B)

---

## Methodology Highlights

### Leakage Controls ✅
- **GroupKFold by FSU** (geographic clusters) prevents spatial leakage
- **Outcome variables excluded**: b12q12 (repayment), b12q13 (writeoff), b12q14 (survey-date balance)
- **Debt aggregates**: ONLY use b12q15 (30-06-2018 outstanding), ONLY loans with b12q3 == 1
- **Evidence**: AUC dropped 0.886 → 0.759 when leakage fixed

### Calibration
- **Isotonic regression** with GroupKFold CV (unbiased ECE)
- ECE 0.015 → 0.005, slope 0.840 → 0.965
- Trade-off: AUC -0.003 for better probabilities

### Robustness
- **Leave-one-region-out CV** (6 regions)
- **PSI across regions** (max 0.58 ❌)
- **Fairness audit**: caste, religion, gender, rural/urban
- **Weighted vs unweighted metrics**

### Explainability
- **TreeSHAP** global importance
- **Top-3 reason codes** per loan
- **Stability test**: Spearman ρ = 0.997 across splits

---

## Limitations

### Data
1. **Survey proxy label** ≠ actual NPAs
2. No payment history (single snapshot)
3. Unverified codes (tenure, purpose)
4. 2019 pre-COVID data
5. No bank data for comparison

### Model
1. **AUC 0.76** < 0.85 target (survey label ceiling)
2. **PSI 0.58** → geographically unstable
3. **Fairness gap 0.28** → cannot deploy to group 9
4. No neural net (unlikely to help on tabular data)
5. No external validation (L&T data not loaded)

### Scope (Out of Scope)
- No uplift modeling (no intervention data)
- No temporal NNs (no transaction sequences)
- No ML-predicted LGD (no recovery data)
- No pricing policy (no price-response data)

---

## Paper Framing

### NOT
- "Deployment-ready credit risk model"
- "SOTA performance on Indian credit data"

### YES
- "Methodology for nationally representative credit risk analysis with robustness testing"
- "First study to test fairness across caste/religion on Indian household credit data"
- "Exposed geographic/demographic heterogeneity hidden in bank selection bias"
- "Policy insights: co-op supervision, informal lending, financial inclusion challenges"

### Title
"Geographic and Demographic Heterogeneity in Credit Risk Models: Evidence from Nationally Representative Indian Household Data"

### Abstract (Draft)
> We develop a credit risk model using NSS 77th Round nationally representative household survey data (34,425 institutional loans, 2019). Unlike bank studies, our data includes unbanked and informal borrowers, exposing heterogeneity hidden by selection bias. We apply rigorous methodology: GroupKFold by geographic cluster, isotonic calibration, and fairness audits across caste/religion. While achieving moderate discrimination (AUC 0.76), the model fails geographic stability (PSI 0.58) and fairness tests (AUC gap 0.28 across social groups). We document that: (1) co-operative lenders have 2× higher NPAs than commercial banks, (2) informal debt share predicts default, (3) national models require region/group-specific calibration, and (4) survey proxy labels limit predictive performance. Our contribution is methodological rigor and policy insights, not deployment-ready models.

---

## Dependencies

See `requirements.txt`:
- pandas, numpy, scikit-learn
- lightgbm, catboost
- shap, scipy, statsmodels
- matplotlib, seaborn

Python 3.8+

---

## Citation

```bibtex
@article{nss77creditrisk2026,
  title={Geographic and Demographic Heterogeneity in Credit Risk Models: Evidence from Nationally Representative Indian Household Data},
  author={[Your Name]},
  year={2026},
  note={NSS 77th Round analysis with fairness audit}
}
```

---

## Contact

For questions about methodology or data access, please open an issue.

**Data Source**: NSS 77th Round (Jan-Dec 2019), Schedule 18.2, Ministry of Statistics and Programme Implementation (MoSPI), Government of India.

---

**License**: Code MIT, Data subject to MoSPI terms.
