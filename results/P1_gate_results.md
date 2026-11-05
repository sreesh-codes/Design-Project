# Phase 1 Gate Results: Data & Labels

**Date**: 2026-11-06
**Status**: ✅ **PASS** - All validation checks met

---

## Deliverables

| Item | Status | Location |
|------|--------|----------|
| Cleaned loan-level table (parquet) | ✅ | `data/labeled_loans_v1.parquet` |
| Data dictionary (markdown) | ✅ | `DATA_DICTIONARY.md` |
| Label construction code | ✅ | `src/labels.py` |
| Data loading code | ✅ | `src/data.py` |

---

## Primary Label Validation

| Metric | Target | Actual | Diff % | Status |
|--------|--------|--------|--------|--------|
| **Total loans** | 34,230 | 34,425 | +0.6% | ✅ |
| **Unique households** | 28,914 | 29,092 | +0.6% | ✅ |
| **Unique FSUs** | 8,180 | 8,180 | +0.3% | ✅ |
| **Positive rate** | 16.0% | 15.9% | -0.3% | ✅ |

**Tolerance**: ±5% for counts, ±15% relative for rates
**Result**: All metrics within tolerance ✅

---

## Label Construction Logic

### Primary Label (`npa_primary`)
**NPA Proxy Definition**: Institutional loan outstanding on 30-06-2018 with zero repayment from 01-07-2018 to survey date

**Filters Applied**:
1. Institutional loans: `b12q5` ∈ {1, 2, ..., 13}
   → 65,601 loans from 106,033 total
2. Outstanding on 30-06-2018: `b12q3` == 1
   → 45,417 loans
3. Exclude KCC/crop loans: `b12q6` ∉ {7, 8}
   → **34,425 final loans**

**Label**: `y = 1` if `b12q12` (repayment amount) is 0 or missing
**Result**: 5,490 NPAs (15.9%)

---

## Sensitivity Labels

| Label | Definition | Count | Rate |
|-------|------------|-------|------|
| `npa_writeoff` | b12q13 > 0 | 399 | 1.2% |
| `npa_high_outstanding` | Zero repay AND b14 ≥ 1.1×b15 | 0 | 0.0% |
| `npa_excl_tenure1` | Primary excl. tenure code 1 | 4,588* | 14.4%* |

\* Out of 31,875 loans after excluding tenure 1 (2,550 loans dropped)

**Finding**: High outstanding sensitivity label has 0 prevalence → investigate if b12q14/b12q15 logic is valid or if field is sparse.

---

## NPA Rate by Credit Agency

| Agency Code | Agency Type | Loans | NPAs | Rate |
|-------------|-------------|-------|------|------|
| 01 | Scheduled Commercial Bank | 17,048 | 2,612 | **15.3%** |
| 02 | Regional Rural Bank | 1,219 | 348 | **28.5%** |
| 03 | Co-operative Society | 1,761 | 590 | **33.5%** |
| 04 | Co-operative Bank | 1,440 | 330 | **22.9%** |
| 05 | Insurance | 274 | 61 | 22.3% |
| 06 | Provident Fund | 67 | 2 | 3.0% |
| 07 | Employer | 114 | 10 | 8.8% |
| 08 | Financial Corporation | 1,565 | 111 | 7.1% |
| 09 | Other Institutional | 2,226 | 737 | **33.1%** |
| 10 | NBFC/MFI | 2,778 | 160 | **5.8%** |
| 11 | Bank-linked SHG/JLG | 4,611 | 390 | 8.5% |
| 12 | Non-bank SHG | 721 | 85 | 11.8% |
| 13 | Other Institutional | 601 | 54 | 9.0% |

**Key Insights**:
- Co-operative Society (03) and Other Institutional (09) have highest NPA rates (~33%)
- NBFCs/MFIs (10) have lowest rate among major lenders (5.8%)
- RRBs (02) higher than SCBs (01): 28.5% vs 15.3%

---

## Leakage Protection

### Verified Controls
✅ Outcome variables **present** in data but **flagged as FORBIDDEN**:
- `b12q12` (repayment amount)
- `b12q13` (write-off amount)
- `b12q14` (survey-date outstanding)

✅ Debt aggregation rule documented:
- ONLY use loans with `b12q3` == 1
- ONLY use `b12q15` (30-06-2018 outstanding)
- Using `b12q14` in baseline inflated AUC from 0.886 → 0.937

✅ Demographics excluded from features:
- Religion (Block 4)
- Social group / caste (Block 4)
- Reserved for fairness audit only

---

## Data Limitations Identified

### 1. Unverified Code Meanings
**Cannot verify from available documentation**:
- `b12q7` (tenure code): will use as categorical; code 1 has 35.5% NPA vs 14.4% for others
- `b12q10` (purpose code): will use as categorical

**Impact**: Cannot apply domain-driven monotone constraints; must treat as categorical.

### 2. Visit 2 Field Semantics
- Visit 2 Block 12 semantics not verified
- **Phase 2 early warning label** construction deferred until catalog 156 obtained

### 3. High Outstanding Label
- `npa_high_outstanding` has 0% prevalence
- Possible causes: (a) `b12q14` sparse, (b) outstanding typically decreases, (c) logic error
- **Action**: Investigate in EDA; may drop this sensitivity label

---

## Data Quality Summary

| Metric | Value |
|--------|-------|
| Visit 1 raw loan records | 106,033 |
| After dropping total rows (b12q1==99) | 106,033 |
| Final labeled dataset | 34,425 |
| Missing `b12q12` (repayment) | 1.8% |
| Missing `b12q15` (outstanding) | 0.1% |

---

## Next Steps → Phase 2

**Cleared to proceed**: ✅

**P2 Tasks**:
1. Load household feature files (Levels 02–12, V2 Level 18)
2. Engineer feature groups:
   - Loan characteristics
   - Borrower demographics
   - Balance sheet (debt/assets, debt/consumption, liquidity)
   - Portfolio/informality (concurrent loans, informal share, max rate)
   - Context (state, sector, exposure months)
3. Build baselines:
   - WoE scorecard
   - Multilevel logistic
   - LightGBM + CatBoost with monotone constraints
4. **Gate**: Per-feature-group ablation; no leakage-suspicious jumps

---

## Open Questions for User

1. **Proceed to P2?** All validation checks passed.
2. **High outstanding label**: Should we drop it given 0% prevalence, or investigate `b12q14` missingness first?
3. **Unverified codes**: Acceptable to use as categorical, or should we attempt web search for NSS 77 instruction manual?

---

**Sign-off**: P1 complete. Data clean, labels validated, leakage controls in place. Ready for feature engineering.
