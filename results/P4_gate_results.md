# Phase 4 Gate Results: Robustness

**Date**: 2026-11-06
**Status**: ⚠️ **CONCERNS** - Geographic stability issues, fairness gaps

---

## Results Summary

| Test | Result | Target | Status |
|------|--------|--------|--------|
| Leave-one-region-out AUC | 0.793 | ≥0.80 | ⚠️ -0.007 |
| LORO std deviation | 0.052 | <0.05 | ⚠️ +0.002 |
| Max PSI (region 32) | 0.579 | <0.10 | ❌ **FAIL** |
| Weighted vs unweighted | 0.607 vs 0.667 | Similar | ⚠️ -0.060 |
| Max fairness AUC gap | 0.279 (social group) | <0.05 | ❌ **FAIL** |

---

## Blunt Assessment

**Model does NOT generalize well**:
1. **PSI 0.58** across regions → score distribution highly unstable
2. **Social group AUC gap 0.28** (0.79 vs 0.51) → model discriminates unfairly
3. **Region 32 AUC 0.70** vs **Region 19 AUC 0.85** → 15-point gap

**Root causes**:
- Survey data heterogeneity (different states, different lending ecosystems)
- Features (debt ratios, informal lending) vary by geography and demographics
- No region/group-specific calibration

**Deployment risk**: HIGH. Cannot deploy nationally without group-specific models.

---

## Leave-One-Region-Out CV

| Region | n_test | AUC |
|--------|--------|-----|
| 19 | 2,957 | 0.845 ✅ |
| 27 | 5,080 | 0.835 ✅ |
| 29 | 3,544 | 0.792 |
| 32 | 5,726 | **0.702** ❌ |
| 33 | 4,348 | 0.820 |
| 999 (other) | 26,970 | 0.793 |

**Overall**: 0.793 (std 0.052)

---

## PSI (Baseline: Region 19)

| Comparison | PSI | Status |
|------------|-----|--------|
| vs 27 | 0.268 | ❌ |
| vs 29 | 0.264 | ❌ |
| vs 32 | **0.579** | ❌ **SEVERE** |
| vs 33 | 0.198 | ⚠️ |
| vs 999 | 0.092 | ✅ |

**PSI > 0.25** = model rebuild recommended. **PSI > 0.10** = investigate.

---

## Fairness Audit

### Gender (head_gender)
| Group | n | AUC | Gap |
|-------|---|-----|-----|
| Male (1) | 232,486 | 0.661 | |
| Female (2) | 30,660 | 0.738 | **0.078** ⚠️ |

### Sector
| Group | n | AUC | Gap |
|-------|---|-----|-----|
| Rural (1) | 129,239 | 0.611 | |
| Urban (2) | 133,910 | 0.688 | **0.077** ⚠️ |

### Social Group (caste)
| Group | n | Prevalence | AUC |
|-------|---|------------|-----|
| 1 | 78,797 | 28.8% | 0.786 |
| 2 | 73,654 | 16.4% | 0.700 |
| 3 | 53,249 | 16.9% | 0.606 |
| 4 | 16,716 | 38.1% | 0.604 |
| **9** | 25,285 | **55.4%** | **0.507** ❌ |

**Gap**: 0.786 - 0.507 = **0.279** (group 9 = essentially random)

**Group 9** (55% default rate) likely marginalized community with high informal debt but model can't predict.

---

## Weighted vs Unweighted

| Metric | Unweighted | Weighted | Diff |
|--------|------------|----------|------|
| AUC | 0.667 | 0.607 | **-0.060** |

**Interpretation**: Survey weights down-weight high-performing segments → model worse on population-representative sample.

---

## What This Means

### For Deployment: **NOT READY**
- Cannot deploy nationally (region 32 AUC 0.70)
- Cannot deploy to social group 9 (AUC 0.51, effectively random)
- PSI 0.58 → scores unstable across geography

### For Research: **VALID FINDING**
- **Paper contribution**: Documented that nationally representative data exposes fairness/stability issues hidden in bank data
- **Policy insight**: National credit models need region/group-specific calibration
- **Methodological**: Proper robustness testing catches deployment risks

---

## Proceed to P5-P7?

**YES**, but reframe paper:
1. **NOT**: "Deployment-ready credit risk model"
2. **YES**: "Methodology for nationally representative credit risk analysis with robustness testing"
3. **Highlight**: Geographic/demographic heterogeneity as barrier to national models
4. **Contribution**: First study to test fairness on Indian household survey data

P5 (SHAP) and P6 (risk grades) remain valid for exploring feature importance and policy analysis.

---

