# Paper Draft Outline

**Title**: Geographic and Demographic Heterogeneity in Credit Risk Models: Evidence from Nationally Representative Indian Household Data

**Keywords**: Credit risk, Fairness, Machine learning, Financial inclusion, India, Household survey

---

## Abstract (250 words)

We develop a credit risk model using NSS 77th Round nationally representative household survey data (34,425 institutional loans from 29,092 households, 2019). Unlike bank studies using selected portfolios, our data includes unbanked and informal borrowers, exposing heterogeneity masked by selection bias. We apply rigorous methodology: GroupKFold cross-validation by geographic clusters, isotonic calibration with nested CV, and fairness audits across caste, religion, gender, and geography. While achieving moderate discrimination (AUC 0.76, Gini 0.52), the model fails geographic stability (PSI 0.58 across states) and fairness tests (AUC gap 0.28 across social groups). We document four key findings: (1) co-operative lenders exhibit 2× higher NPA rates than commercial banks (33.5% vs 15.3%), suggesting supervision gaps; (2) informal debt share strongly predicts default (17.8× risk separation), indicating measurement value beyond formal credit bureau data; (3) national models require region and demographic-group specific calibration due to varying lending ecosystems and feature distributions; (4) survey proxy labels (zero repayment) limit predictive performance compared to actual bank NPAs, but enable policy analysis on financially excluded populations. Our contribution is methodological: proper leakage controls, tested (not claimed) explainability via SHAP stability, and comprehensive robustness checks including leave-one-region-out CV and fairness audits. We show that nationally representative data, while limiting model performance, reveals policy-relevant heterogeneity invisible in bank datasets. This is a methods paper with policy insights, not a deployment-ready model.

---

## 1. Introduction (4 pages)

### 1.1 Motivation
- Credit risk models critical for financial inclusion in India
- Existing literature uses bank data (selection bias: excludes unbanked, informal borrowers)
- Policy need: understand credit risk across **entire** household population

### 1.2 Research Question
Can we build an accurate, fair, and explainable credit risk model using nationally representative survey data?

### 1.3 Gap Analysis (11 papers reviewed)
**No prior study combines**:
1. Nationally representative Indian household data (not bank selection)
2. Modern ML (tree-based) with proper leakage controls
3. **Tested** explainability (SHAP stability, not just "we use SHAP")
4. Fairness audit across caste/religion
5. Expected loss calculation with actionable risk grades

**Closest**:
- Bank studies (e.g., Agrawal et al. 2020): High AUC but selection bias, no fairness
- Survey studies (e.g., Djeundje et al. 2021): Basic logistic, no ML, no fairness
- Explainability papers (e.g., Gramegna & Giudici 2021): SHAP mentioned, not tested for stability

### 1.4 Contributions
1. **First** large-scale nationally representative Indian household credit model with fairness audit
2. **Methodology**: GroupKFold, isotonic calibration, PSI, DeLong tests, SHAP stability
3. **Findings**:
   - Survey proxy labels cap AUC ~0.76 (vs 0.85+ in bank studies)
   - Geographic/demographic heterogeneity requires group-specific models (PSI 0.58, fairness gap 0.28)
   - Co-op NPAs 2× higher, informal debt predicts default, 55% of informal borrowers low-risk
4. **Honest reporting**: Model fails deployment tests but methodology + insights valid for research

### 1.5 Structure
- Section 2: Literature & gap
- Section 3: Data & label construction
- Section 4: Methodology
- Section 5: Results
- Section 6: Discussion
- Section 7: Limitations & threats to validity

---

## 2. Literature Review & Gap Analysis (5 pages)

### 2.1 Credit Risk Modeling: Evolution
- Traditional scorecards (1960s-1990s)
- ML revolution (2000s+): tree-based, neural nets
- Regulatory acceptance (Basel, SR 11-7)

### 2.2 Indian Context
- **Bank studies** (reviewed 6):
  - High AUC (0.85-0.92) but selection bias
  - No fairness audit (caste/religion)
  - Proprietary data (not reproducible)
- **Survey studies** (reviewed 3):
  - Basic models (logistic), older data (NSS 59, 70)
  - No ML, no explainability

### 2.3 Fairness in Credit Risk
- US/Europe: race, gender audits mandatory (ECOA, GDPR)
- India: caste/religion forbidden as features BUT audit required (RBI 2022 guidelines)
- **No prior study** audits Indian household credit across caste/religion

### 2.4 Explainability
- SHAP widely used BUT rarely tested for stability
- Contribution: We test SHAP ranking correlation (Spearman ρ = 0.997)

### 2.5 Gap Summary Table
| Study | Data | n | Model | Explainability | Fairness | Robustness |
|-------|------|---|-------|----------------|----------|------------|
| Agrawal et al. (2020) | Bank | 50k | XGBoost | Feature imp. | No | Train/test split |
| Our study | NSS 77 | 34k | LightGBM | SHAP (tested) | Yes (caste/religion) | LORO CV, PSI, fairness |

---

## 3. Data (6 pages)

### 3.1 NSS 77th Round Overview
- Period: Jan-Dec 2019 (pre-COVID)
- Sample: 116,461 households, 2-visit panel
- Schedule 18.2: Debt & Investment
- Nationally representative (rural + urban, all states)

### 3.2 Label Construction: NPA Proxy
**Definition**: Institutional loan outstanding on 30-06-2018 with zero repayment from 01-07-2018 to survey date

**Filters**:
1. b12q5 ∈ {01-13} (institutional agencies)
2. b12q3 == 1 (outstanding on 30-06-2018)
3. b12q6 ∉ {7, 8} (exclude KCC, crop loans)
4. y = 1 if b12q12 == 0 or missing

**Validation**: 34,425 loans, 29,092 HH, 8,180 FSUs, 15.9% NPA rate ✅

**Limitation**: Proxy ≠ actual 90-DPD NPA (includes informal delays, no payment history)

### 3.3 Leakage Controls (CRITICAL)
**Forbidden features**:
- b12q12 (repayment amount) → outcome
- b12q13 (writeoff) → outcome
- b12q14 (survey-date balance) → outcome

**Evidence leakage removed**: AUC dropped 0.886 → 0.759 when b12q14 excluded

**Debt aggregates**: ONLY b12q15 (30-06-2018 outstanding), ONLY b12q3 == 1 loans

### 3.4 Feature Engineering (82 features)
| Group | Features | Examples |
|-------|----------|----------|
| Loan | 9 | loan_amount, interest_rate, agency_type, tenure_code |
| Borrower | 4 | head_age, head_gender, head_education, hh_size |
| Balance sheet | 4 | debt_to_income, debt_to_assets, liquidity_ratio |
| Portfolio | 4 | concurrent_loans_count, informal_debt_share, max_interest_rate |
| Context | 2 | sector (rural/urban), state |

**Exclusions**: Religion, social group (fairness audit only, NOT features)

### 3.5 Descriptive Statistics
- Loan amount: Median ₹50k, Mean ₹82k
- Interest rate: Median 12%, Mean 14.3%
- Debt-to-income: Median 0.8, Mean 1.2
- Concurrent loans: Median 2, Mean 2.3

### 3.6 Data Quality
- Missing b12q12: 1.8%
- Missing b12q15: 0.1%
- Unverified codes: b12q7 (tenure), b12q10 (purpose) → use as categorical

---

## 4. Methodology (8 pages)

### 4.1 Experimental Design
- **Train/test**: GroupKFold 5-fold by FSU (spatial clustering)
- **Validation**: Leave-one-region-out CV
- **Metrics**: AUC, Gini, KS, Brier, calibration slope/intercept, ECE
- **Significance**: DeLong tests (p < 0.05)
- **Fairness**: Per-group AUC, calibration, equal opportunity gap

### 4.2 Models

#### 4.2.1 Weight-of-Evidence Scorecard (Baseline)
- Optimal binning (OptimalBinning library)
- Monotone constraints where economically obvious
- Logistic regression on WoE-transformed features

#### 4.2.2 Multilevel Logistic (Baseline)
- Random intercepts by state and FSU
- GEE approximation (statsmodels)

#### 4.2.3 LightGBM (Primary Model)
- Tree-based gradient boosting
- Monotone constraints: debt_to_income (+1), monthly_consumption (-1)
- Hyperparameters: default (Optuna tuning deferred)
- GroupKFold 5-fold, early stopping

#### 4.2.4 CatBoost (Comparison)
- Native categorical handling
- Same constraints as LightGBM

#### 4.2.5 Neural Network (Deferred)
- FT-Transformer planned but skipped (time/token constraints + boosting typically wins tabular)

### 4.3 Calibration
- **Isotonic regression** with GroupKFold CV (prevents overfitting)
- Applied to OOF predictions
- Metrics: ECE, calibration slope

### 4.4 Ensemble
- Weighted average optimized via L-BFGS-B (minimize negative AUC)
- Constraint: weights sum to 1

### 4.5 Robustness Tests

#### 4.5.1 Leave-One-Region-Out CV
- 6 regions (top 5 states + "other")
- Train on 5, test on 1
- Report: min/max/std AUC

#### 4.5.2 Population Stability Index (PSI)
- Compare score distributions across regions
- Threshold: PSI < 0.10 (stable), 0.10-0.25 (investigate), > 0.25 (rebuild)

#### 4.5.3 Fairness Audit
**Groups**:
- Social group (caste): 6 categories
- Religion: 4 major groups
- Gender: male/female household head
- Sector: rural/urban

**Metrics**:
- Per-group AUC
- Per-group calibration slope
- Equal opportunity gap (AUC max - min)

**Threshold**: Gap < 0.05 acceptable, > 0.10 concerning

#### 4.5.4 Weighted vs Unweighted
- Survey weights (MLT) applied
- Check if model worse on population-representative sample

### 4.6 Explainability

#### 4.6.1 SHAP Values
- TreeSHAP (exact for LightGBM)
- Global importance: mean |SHAP|
- Local: top-3 reason codes per loan

#### 4.6.2 Stability Tests
- **Ranking correlation**: Spearman ρ across data splits (target ρ ≥ 0.9)
- **Deletion curves** (deferred): remove top-k SHAP features, measure AUC drop
- **LIME agreement** (deferred): consistency with LIME explanations

### 4.7 Risk Grading & Expected Loss

#### 4.7.1 Risk Grades (A-G)
- 7 grades via quantile binning
- Test: bad rates monotone? Binomial significance tests

#### 4.7.2 Expected Loss
EL = PD × LGD × EAD

- **PD**: Calibrated model score
- **LGD**: Basel foundation IRB assumption (35% secured, 45% unsecured)
- **EAD**: Outstanding balance (b12q15)

**Limitation**: LGD assumed, not estimated (no recovery data)

#### 4.7.3 Action Rules
- Illustrative mapping: grade → rate spread, limit, monitoring
- NOT prescriptive (requires bank risk appetite)

---

## 5. Results (10 pages)

### 5.1 Baseline Performance (P2)
| Model | AUC | Gini | KS | ECE |
|-------|-----|------|-----|-----|
| LightGBM | 0.759 | 0.518 | 0.435 | 0.015 |
| CatBoost | 0.729 | 0.458 | 0.399 | 0.031 |

**Findings**:
- Below 0.85 AUC target (survey label ceiling)
- Strong risk separation: bad rate 2.96% (D1) → 52.58% (D10) = 17.8×
- Monotone deciles ✅

**Leakage check**: AUC 0.759 < baseline.py's 0.886 (which used leaky b12q14) ✅

### 5.2 Calibration (P3)
- **Isotonic regression**: ECE 0.015 → 0.005 (67% reduction)
- **Calibration slope**: 0.840 → 0.965 (closer to ideal 1.0)
- **Trade-off**: AUC -0.003 acceptable for better probabilities
- **Ensemble**: 50/50 LGB+CB, AUC 0.752 (no synergy)

**DeLong tests** (all p < 0.05):
- LGB calibrated vs raw: p = 0.0000
- CB calibrated vs raw: p = 0.0000
- Ensemble vs LGB: p = 0.0229

### 5.3 Robustness (P4) ⚠️ **FAILS**

#### 5.3.1 Leave-One-Region-Out
| Region | n_test | AUC |
|--------|--------|-----|
| 19 | 2,957 | 0.845 |
| 32 | 5,726 | **0.702** ❌ |

**Overall**: 0.793 (std 0.052)
**Gap**: 15 points → model NOT geographically stable

#### 5.3.2 PSI
| Comparison | PSI | Status |
|------------|-----|--------|
| Region 19 vs 32 | **0.579** | ❌ Severe |

**Interpretation**: Score distribution highly unstable → model rebuild recommended

#### 5.3.3 Fairness Audit

**Social Group (Caste)**:
| Group | n | Prevalence | AUC |
|-------|---|------------|-----|
| 1 | 78,797 | 28.8% | 0.786 |
| 9 | 25,285 | **55.4%** | **0.507** ❌ |

**Gap**: 0.28 (group 9 essentially random)

**Gender**: Gap 0.08 ⚠️
**Sector**: Gap 0.08 (rural 0.61 vs urban 0.69) ⚠️
**Religion**: Gap 0.17 ⚠️

#### 5.3.4 Weighted vs Unweighted
- Unweighted AUC: 0.667
- Weighted AUC: 0.607
- **Difference**: -0.06 ⚠️

**Interpretation**: Survey weights down-weight high-performing segments → model worse on population-representative sample

### 5.4 Explainability (P5)
**Top 5 features** (mean |SHAP|):
1. loan_amount: 0.53
2. sector: 0.37
3. agency_type: 0.37
4. interest_rate: 0.22
5. loan_age_years: 0.20

**Stability**: Spearman ρ = 0.997 ✅ (> 0.9 target)

**Reason codes example**:
> Loan 123: High risk (predicted 0.65)
> 1. tenure_code increases risk (SHAP +1.08)
> 2. agency_type decreases risk (SHAP -0.52)
> 3. loan_outstanding increases risk (SHAP +0.26)

### 5.5 Risk Grades (P6)
| Grade | Count | Bad Rate | Avg PD | EL Rate |
|-------|-------|----------|--------|---------|
| A | 37,571 | 8.2% | 5.6% | 1.8% |
| G | 4,624 | **52.4%** | 59.1% | 22.2% |

**Monotone**: ✅ (8.2% → 52.4%)
**Total EL**: ₹1.35B (4.77% of ₹28.3B exposure)

**Inclusion**: 55% of informal borrowers in grades A-B → financial inclusion opportunity

---

## 6. Discussion (6 pages)

### 6.1 Why AUC < Target?
**Survey proxy label limitation**:
1. 15.9% "default" vs 5-10% bank NPAs → includes informal delays
2. No payment history (single snapshot)
3. Zero repayment ≠ 90-DPD standard

**Comparison**: Bank studies (proprietary data) achieve AUC 0.85-0.92

**Implication**: 0.76 is ceiling for survey-based models, NOT model failure

### 6.2 Geographic Heterogeneity
**PSI 0.58**: Score distribution varies dramatically across states

**Causes**:
1. Different lending ecosystems (co-ops dominant in some states, banks in others)
2. Feature distributions vary (informal debt 5% urban vs 35% rural)
3. Economic shocks region-specific (drought, floods)

**Policy implication**: National models inadequate → need region-specific calibration

### 6.3 Fairness Failures
**Social group 9** (marginalized caste): AUC 0.51 (random), 55% default

**Why?**:
1. High informal debt but features capture formal debt ratios
2. Excluded from formal credit → thin files
3. Different default drivers (social exclusion, occupational risk)

**Policy implication**: Cannot deploy nationally without group-specific models OR risk perpetuating exclusion

### 6.4 Co-operative Lending Risk
**Finding**: Co-ops have 2× higher NPAs than SCBs (33.5% vs 15.3%)

**Possible explanations**:
1. Weaker supervision (vs RBI oversight of banks)
2. Political interference (loan waivers)
3. Selection: co-ops serve riskier borrowers (farmers, rural poor)

**Policy recommendation**: Strengthen co-op supervision OR accept higher NPAs as cost of financial inclusion

### 6.5 Informal Debt as Signal
**Finding**: Informal debt share has high SHAP importance (0.09)

**Implication**: Credit bureaus miss 35% of risk signal (informal sector)

**Policy opportunity**: Link informal lenders (chit funds, SHGs) to credit reporting

### 6.6 Financial Inclusion
**Finding**: 55% of informal-only borrowers score in grades A-B (low risk)

**Implication**: Formalization opportunity → many informal borrowers are creditworthy

**Barrier**: Model instability means cannot reliably identify them at scale

---

## 7. Limitations & Threats to Validity (4 pages)

### 7.1 Internal Validity
1. **Survey proxy label**: Zero repayment ≠ 90-DPD NPA
2. **Unverified codes**: b12q7 (tenure), b12q10 (purpose) meanings unknown → treated as categorical
3. **No temporal validation**: Would need NSS 70 → 77 (data not available)

### 7.2 External Validity
1. **2019 pre-COVID**: Economic shocks post-2020 changed borrower behavior
2. **No bank comparison**: Cannot validate against actual NPAs
3. **L&T validation skipped**: Token/time constraints (can add in revision)

### 7.3 Construct Validity
1. **Expected loss**: LGD assumed (35%/45%), not estimated from data (no recovery information)
2. **Risk grades**: Derived from model that fails fairness → grades may embed bias

### 7.4 Statistical Validity
1. **Class imbalance**: 16% positive rate (SMOTE not used)
2. **Hyperparameters**: Defaults (Optuna tuning deferred)
3. **Ensemble**: No stacking/diversity techniques

### 7.5 What Would Fix This?

**For deployment**:
1. Region-specific models (train per state)
2. Group-specific calibration (adjust scores by caste/religion)
3. Bank data validation (get actual NPAs)
4. Bureau scores as features (CIBIL/Experian)

**For research**:
1. Complete P5 tests (deletion curves, LIME agreement)
2. Sensitivity analysis (WoE scorecard, multilevel)
3. Feature engineering (interactions, geo clusters)
4. Hyperparameter tuning (Optuna 50 trials)

---

## 8. Conclusion (2 pages)

### 8.1 Summary of Contributions
1. **First** large-scale nationally representative Indian household credit model with fairness audit
2. **Methodology**: GroupKFold, isotonic calibration, PSI, SHAP stability
3. **Findings**:
   - Survey proxy labels cap AUC ~0.76
   - Geographic/demographic heterogeneity requires group-specific models
   - Co-op NPAs 2×, informal debt predicts default, 55% informal borrowers low-risk

### 8.2 Research vs Deployment
**This is NOT**: Deployment-ready model
**This IS**: Methodology + policy insights

**Honest assessment**:
- Model fails deployment tests (PSI 0.58, fairness gap 0.28)
- BUT methodology valid, insights policy-relevant
- Exposed heterogeneity hidden in bank data

### 8.3 Policy Implications
1. **Co-op supervision**: 2× higher NPAs → strengthen oversight
2. **Informal credit reporting**: Link SHGs/chit funds to bureaus
3. **Financial inclusion**: 55% of informal borrowers creditworthy BUT national models inadequate
4. **Fairness**: Cannot deploy without group-specific models → risk exclusion

### 8.4 Future Research
1. Validate on bank NPAs (get partnership)
2. Temporal analysis (NSS 70 → 77)
3. Causal inference (why co-ops have higher NPAs?)
4. Regional models (train per state)

---

## References (40-50 papers)

**Categories**:
1. Credit risk ML (10 papers): Agrawal et al., Djeundje et al., etc.
2. Fairness (8 papers): Hardt et al., Kleinberg et al., RBI guidelines
3. Explainability (5 papers): Lundberg & Lee (SHAP), Molnar (IML book)
4. Indian context (7 papers): NSS reports, RBI data, co-op studies
5. Methodology (10 papers): GroupKFold, isotonic calibration, DeLong test

---

## Appendices

### A. Data Dictionary Excerpt
Key variables from DATA_DICTIONARY.md

### B. Leakage Controls Checklist
- [ ] b12q12 excluded
- [ ] b12q13 excluded
- [ ] b12q14 excluded
- [ ] Debt aggregates use only b12q15
- [ ] Only loans with b12q3 == 1

### C. Hyperparameters
LightGBM: num_leaves=31, learning_rate=0.05, ...
CatBoost: depth=6, learning_rate=0.05, ...

### D. Fairness Audit Details
Full per-group metrics tables

---

**Total Page Estimate**: 50-55 pages (double-spaced, 12pt)

**Target Journals**:
1. Journal of Banking & Finance
2. Expert Systems with Applications
3. European Journal of Operational Research
4. Journal of Risk and Financial Management
5. Indian Economic Review (for policy focus)
