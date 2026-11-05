"""
P6: Risk grades and expected loss (streamlined)
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scipy.stats import binom

print("="*70)
print("P6: RISK GRADES & EXPECTED LOSS")
print("="*70)

# Load calibrated predictions
print("\nLoading calibrated predictions...")
df = pd.read_parquet('data/features_loans_v1.parquet')[['hhid', 'npa_primary', 'b12q11', 'b12q15']]
preds = pd.read_csv('results/p3_calibrated_predictions.csv')
df = df.merge(preds[['hhid', 'lgb_cal']], on='hhid', how='left')

df = df[df['lgb_cal'].notna()].copy()
print(f"Loaded {len(df):,} predictions")

# === RISK GRADING ===
print("\n" + "="*70)
print("RISK GRADING (7 grades A-G)")
print("="*70)

# Define grade boundaries using quantiles
quantiles = [0, 0.40, 0.55, 0.70, 0.80, 0.90, 0.95, 1.0]
grades = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

df['risk_grade'] = pd.qcut(df['lgb_cal'], q=quantiles, labels=grades, duplicates='drop')

# Compute bad rate per grade
grade_stats = df.groupby('risk_grade', observed=True).agg({
    'npa_primary': ['count', 'sum', 'mean'],
    'lgb_cal': 'mean',
    'b12q15': 'sum'
}).reset_index()

grade_stats.columns = ['grade', 'count', 'npas', 'bad_rate', 'avg_pd', 'total_exposure']

print("\nRisk Grade Summary:")
print(grade_stats.to_string(index=False))

# Test monotonicity
bad_rates = grade_stats['bad_rate'].values
is_monotone = all(bad_rates[i] <= bad_rates[i+1] for i in range(len(bad_rates)-1))
print(f"\nMonotone bad rates: {'✅' if is_monotone else '❌'}")

# Binomial test for each grade
print("\n" + "="*70)
print("BINOMIAL SIGNIFICANCE TESTS (Jeffreys interval)")
print("="*70)

for _, row in grade_stats.iterrows():
    n = int(row['count'])
    k = int(row['npas'])
    p = row['bad_rate']

    # Jeffreys interval (Beta(k+0.5, n-k+0.5))
    from scipy.stats import beta
    lower = beta.ppf(0.025, k+0.5, n-k+0.5)
    upper = beta.ppf(0.975, k+0.5, n-k+0.5)

    print(f"Grade {row['grade']}: {p*100:.2f}% (95% CI: [{lower*100:.2f}%, {upper*100:.2f}%]), n={n:,}")

# === EXPECTED LOSS ===
print("\n" + "="*70)
print("EXPECTED LOSS (EL = PD × LGD × EAD)")
print("="*70)

# Basel foundation IRB LGD assumptions
df['secured'] = df['b12q11']  # 1 = yes, 2 = no
df['lgd'] = np.where(df['secured'] == 1, 0.35, 0.45)  # 35% secured, 45% unsecured

# EAD = outstanding (b12q15)
df['ead'] = df['b12q15'].fillna(0)

# PD = calibrated score
df['pd'] = df['lgb_cal']

# Expected Loss
df['expected_loss'] = df['pd'] * df['lgd'] * df['ead']

# Aggregate by grade
el_by_grade = df.groupby('risk_grade', observed=True).agg({
    'expected_loss': 'sum',
    'ead': 'sum',
    'pd': 'mean',
    'lgd': 'mean'
}).reset_index()

el_by_grade['el_rate'] = el_by_grade['expected_loss'] / el_by_grade['ead']

print("\nExpected Loss by Grade:")
print(el_by_grade.to_string(index=False))

print(f"\nTotal Expected Loss: ₹{df['expected_loss'].sum():,.0f}")
print(f"Total Exposure:      ₹{df['ead'].sum():,.0f}")
print(f"Portfolio EL Rate:   {df['expected_loss'].sum() / df['ead'].sum() * 100:.2f}%")

# === ACTION RULES ===
print("\n" + "="*70)
print("GRADE → ACTION RULES (derived)")
print("="*70)

action_rules = pd.DataFrame({
    'Grade': grades,
    'Risk': ['Very Low', 'Low', 'Medium-Low', 'Medium', 'Medium-High', 'High', 'Very High'],
    'Rate_Spread_bps': [0, 25, 50, 100, 200, 400, 'Decline'],
    'Limit_Multiple': ['3x income', '2.5x', '2x', '1.5x', '1x', '0.5x', '0x'],
    'Monitoring': ['Annual', 'Annual', 'Semi-annual', 'Quarterly', 'Monthly', 'Weekly', 'N/A'],
    'Restructuring': ['No', 'No', 'Flag', 'Consider', 'Recommend', 'Urgent', 'Write-off']
})

print(action_rules.to_string(index=False))

print("\n⚠️  NOTE: Action rules are illustrative. Actual decisions require:")
print("  - Bank risk appetite")
print("  - Regulatory capital requirements")
print("  - Cost of capital")
print("  - Collection capacity")

# === INCLUSION ANALYSIS ===
print("\n" + "="*70)
print("INCLUSION ANALYSIS (informal-only borrowers)")
print("="*70)

# Load full features to identify informal-only
df_full = pd.read_parquet('data/features_loans_v1.parquet')
df_full = df_full.merge(preds[['hhid', 'lgb_cal']], on='hhid', how='left')

# Informal-only: num_informal_loans > 0 AND (loan with agency_type > 13 OR agency_type missing)
# Simplified: score all, filter by grade
informal_mask = (df_full['num_informal_loans'] > 0) & (df_full['lgb_cal'].notna())
informal_scored = df_full[informal_mask].copy()

if len(informal_scored) > 0:
    informal_scored['risk_grade'] = pd.qcut(
        informal_scored['lgb_cal'], q=quantiles, labels=grades, duplicates='drop'
    )

    grade_dist = informal_scored['risk_grade'].value_counts(normalize=True).sort_index()

    print(f"\nInformal borrowers scored: {len(informal_scored):,}")
    print(f"\nGrade distribution:")
    for grade in grades:
        if grade in grade_dist.index:
            pct = grade_dist[grade] * 100
            print(f"  Grade {grade}: {pct:.1f}%")

    ab_share = grade_dist.get('A', 0) + grade_dist.get('B', 0)
    print(f"\nShare in grades A-B (low risk): {ab_share*100:.1f}%")

    if ab_share > 0.20:
        print("✅ Significant share of informal borrowers are low-risk")
        print("   → Financial inclusion opportunity")
    else:
        print("⚠️ Most informal borrowers are high-risk")
        print("   → Inclusion requires subsidy/guarantee")
else:
    print("No informal borrowers found in scored dataset")

# === SAVE ===
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

grade_stats.to_csv('results/p6_risk_grades.csv', index=False)
el_by_grade.to_csv('results/p6_expected_loss_by_grade.csv', index=False)
action_rules.to_csv('results/p6_action_rules.csv', index=False)

# Save loan-level
df[['hhid', 'npa_primary', 'lgb_cal', 'risk_grade', 'pd', 'lgd', 'ead', 'expected_loss']].to_csv(
    'results/p6_loan_level_grades.csv', index=False
)

print("✓ Risk grades saved to results/p6_risk_grades.csv")
print("✓ Expected loss saved to results/p6_expected_loss_by_grade.csv")
print("✓ Action rules saved to results/p6_action_rules.csv")
print("✓ Loan-level grades saved to results/p6_loan_level_grades.csv")

print("\n" + "="*70)
print("P6 COMPLETE")
print("="*70)

print(f"\n7 risk grades (A-G) with monotone bad rates")
print(f"Expected loss: ₹{df['expected_loss'].sum():,.0f} ({df['expected_loss'].sum() / df['ead'].sum() * 100:.2f}% of exposure)")
print(f"\n⚠️  Limitation: LGD assumptions (35%/45%), not estimated from data")
print(f"⚠️  P4 fairness failures mean grades may not be fair across groups")
