# Data Dictionary: NSS 77th Round Credit Risk Analysis

## Dataset Overview
- **Source**: NSS 77th Round All-India Debt & Investment Survey, Schedule 18.2
- **Period**: January–December 2019 (pre-COVID)
- **Type**: Household microdata, two visits
- **Labeled dataset**: `data/labeled_loans_v1.parquet`
- **Scope**: Visit 1 institutional cash loans (Block 12)

## Key Identifiers

| Variable | Type | Description |
|----------|------|-------------|
| `hhid` | int64 | Household ID; last digit = visit number |
| `household_key` | str | Household key = HHID[:-1]; links across visits |
| `fsu_slno` | int64 | First Stage Unit (FSU) = village/ward; **use for GroupKFold CV** |
| `sector` | int64 | 1 = Rural, 2 = Urban |
| `state` | int64 | State code |
| `mlt` | float64 | Survey multiplier (weights) |
| `visit` | int64 | 1 or 2 |

## Block 12: Cash Loan Variables

### Loan Characteristics

| Variable | Type | Description | Notes |
|----------|------|-------------|-------|
| `b12q1` | int64 | Serial number of loan | 99 = household total row (dropped) |
| `b12q2` | int64 | Year borrowed | |
| `b12q3` | int64 | Outstanding on 30-06-2018 | 1=Yes, 2=No; **filter criterion** |
| `b12q4` | float64 | Original loan amount (Rs) | |
| `b12q5` | int64 | Credit agency code | See agency codes below; **1–13 = institutional** |
| `b12q6` | int64 | Loan scheme code | See scheme codes below; **7,8 excluded** |
| `b12q7` | int64 | Tenure code | **UNVERIFIED**; 1 has 35.5% NPA rate (likely bullet) |
| `b12q8` | int64 | Nature of interest | 1=Free, 2=Simple, 3=Compound |
| `b12q9` | float64 | Annual interest rate (%) | |
| `b12q10` | int64 | Purpose code | **UNVERIFIED** |
| `b12q11` | int64 | Secured | 1=Yes, 2=No |
| `b12q15` | float64 | Outstanding on 30-06-2018 (Rs) | **Use for debt aggregates** |

### Outcome Variables (LEAKAGE RISK – DO NOT USE AS FEATURES)

| Variable | Type | Description | Use |
|----------|------|-------------|-----|
| `b12q12` | float64 | Amount repaid 01-07-2018 → survey date | **LABEL ONLY** |
| `b12q13` | float64 | Amount written off in same period | **LABEL ONLY** |
| `b12q14` | float64 | Outstanding on survey date | **LABEL ONLY** |

## Credit Agency Codes (b12q5)

### Institutional (01–13; used for primary label)
| Code | Agency |
|------|--------|
| 01 | Scheduled Commercial Bank (SCB) |
| 02 | Regional Rural Bank (RRB) |
| 03 | Co-operative Society |
| 04 | Co-operative Bank |
| 05 | Insurance Company |
| 06 | Provident Fund |
| 07 | Employer |
| 08 | Financial Corporation |
| 09 | Other Institutional |
| 10 | NBFC/MFI |
| 11 | Bank-linked SHG/JLG |
| 12 | Non-bank SHG |
| 13 | Other Institutional |

### Non-Institutional (14–20; excluded)
| Code | Agency |
|------|--------|
| 14 | Landlord |
| 15 | Agricultural Moneylender |
| 16 | Professional Moneylender |
| 17 | Input Supplier |
| 18 | Relatives/Friends |
| 19 | Chit Fund |
| 20 | Commission Agent |

## Loan Scheme Codes (b12q6)

| Code | Scheme | Excluded? |
|------|--------|-----------|
| 1 | Mudra | No |
| 2 | Stand-Up India | No |
| 3 | NRLM/NULM | No |
| 4 | Other Central Scheme | No |
| 5 | State Scheme | No |
| 6 | Bank Scheme | No |
| 7 | KCC (Kisan Credit Card) | **YES** |
| 8 | Crop/Agricultural Loan | **YES** |
| 9 | None/Other | No |

## Target Labels

### Primary Label: `npa_primary`
**Definition**: Institutional loan outstanding on 30-06-2018 with zero repayment from 01-07-2018 to survey date

**Construction**:
- `b12q5` ∈ {1, 2, ..., 13} (institutional)
- `b12q3` == 1 (outstanding on 30-06-2018)
- `b12q6` ∉ {7, 8} (exclude KCC/crop loans)
- `y = 1` if `b12q12` == 0 or missing

**Metrics** (validated):
- Total loans: 34,425 (target: 34,230, ✓)
- Unique households: 29,092 (target: 28,914, ✓)
- Unique FSUs: 8,180 (target: 8,155, ✓)
- Positive rate: 15.9% (target: 16.0%, ✓)

### Sensitivity Labels

| Label | Definition | Prevalence |
|-------|------------|-----------|
| `npa_writeoff` | `b12q13` > 0 | 1.2% |
| `npa_high_outstanding` | `b12q12` == 0 AND `b12q14` ≥ 1.1 × `b12q15` | 0.0% |
| `npa_excl_tenure1` | Primary label excluding `b12q7` == 1 | 14.4% (31,875 loans) |

## LEAKAGE PROTECTION RULES

### FORBIDDEN Features
**NEVER use these variables or aggregates derived from them:**
- `b12q12` (repayment amount)
- `b12q13` (write-off amount)
- `b12q14` (survey-date outstanding)

### Debt Aggregation Rules
When computing household-level debt features:
1. **ONLY** include loans with `b12q3` == 1
2. **ONLY** use `b12q15` (outstanding on 30-06-2018) for amounts
3. Using survey-date balances (`b12q14`) caused AUC inflation from 0.886 → 0.937 in baseline

### Excluded Demographics (fairness audit only)
- Religion (Block 4)
- Social group / caste (Block 4)
- Use ONLY for fairness evaluation, NOT as features

## Household-Level Feature Files (for P2)

| File | Level | Description | Key Variables |
|------|-------|-------------|---------------|
| Level 02 | Block 3 | Household members | `b3q3` (1 = household head) |
| Level 03 | Block 4 | Household characteristics | Demographics (EXCLUDE religion/social group from features) |
| Level 04 | - | Consumption | `b4q10dot5` = monthly consumption total (Rs) |
| Level 05 | Block 5.1 | Rural land | Asset values; drop total row (code 99) |
| Level 06 | Block 5.2 | Urban land | Asset values |
| Level 07 | Block 6 | Buildings | Asset values |
| Level 08 | Block 7 | Livestock & poultry | Asset values |
| Level 09 | Block 8 | Transport equipment | Asset values |
| Level 10 | Block 9 | Agricultural machinery | Asset values |
| Level 11 | Block 10 | Non-farm business equipment | Asset values |
| Level 12 | Block 11a | Financial assets | Asset values |
| Visit 2 Level 18 | Block 11c | Insurance premiums | Premium amounts |

**Optional** (need temporal verification):
- Levels 16–17: Block 14 transactions (ONLY if provably pre-outcome)

**Excluded**:
- Level 13: Shares (only 3k rows)
- Level 15: Kind loans (out of scope)

## Data Quality Notes

### Unverified Codes
The following codes could NOT be verified from official MoSPI layout:
- `b12q7` (tenure code): meanings unknown; code 1 has 35.5% NPA rate vs 14.4% for others
- `b12q10` (purpose code): meanings unknown

**Treatment**: Use as categorical features without interpretation; include in data dictionary limitations.

### Visit 2 Semantics
Visit 2 Block 12 field semantics require verification from instruction manual (catalog 156).
- **Phase 2 early warning** label construction paused until verification.

## Cross-Validation Strategy
- **Group variable**: `fsu_slno` (FSU)
- **Method**: GroupKFold (5 folds)
- **Reason**: Households are clustered within FSUs; standard K-fold would leak information.

## Survey Weighting
- **Variable**: `mlt` (survey multiplier)
- **Use**: Compute both weighted and unweighted metrics; report both.

---
**Version**: P1 - Label Construction Complete
**Date**: 2026-11-06
**Status**: ✓ All validation checks passed
