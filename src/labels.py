"""
P1: Label construction for NSS 77th Round Credit Risk Analysis
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import load_and_merge_loans


def construct_primary_label(loans):
    """
    Construct PRIMARY NPA proxy label:
    - Institutional loans (b12q5 in 01–13)
    - With b12q3 == 1 (outstanding on 30-06-2018)
    - Excluding b12q6 in {7, 8} (KCC and crop/agri loan schemes)
    - y = 1 if b12q12 is 0 or missing (zero repayment from 01-07-2018 to survey date)

    Target metrics:
    - ~34,230 loans
    - ~28,914 households
    - ~8,155 FSUs
    - ~16.0% positive rate
    """
    df = loans.copy()

    # Filter to institutional loans
    institutional_agencies = list(range(1, 14))  # 01-13
    df = df[df['b12q5'].isin(institutional_agencies)].copy()
    print(f"After filtering to institutional loans (b12q5 in 01-13): {len(df):,} loans")

    # Filter to outstanding on 30-06-2018
    df = df[df['b12q3'] == 1].copy()
    print(f"After filtering to outstanding on 30-06-2018 (b12q3==1): {len(df):,} loans")

    # Exclude KCC and crop/agri loan schemes
    excluded_schemes = [7, 8]
    df = df[~df['b12q6'].isin(excluded_schemes)].copy()
    print(f"After excluding KCC/crop loan schemes (b12q6 not in {{7,8}}): {len(df):,} loans")

    # Construct label: y=1 if b12q12 is 0 or missing
    df['npa_primary'] = ((df['b12q12'] == 0) | (df['b12q12'].isna())).astype(int)

    # Validation metrics
    n_loans = len(df)
    n_households = df['household_key'].nunique()
    n_fsus = df['fsu_slno'].nunique()
    positive_rate = df['npa_primary'].mean() * 100

    print(f"\n=== PRIMARY LABEL VALIDATION ===")
    print(f"Total loans:           {n_loans:,}")
    print(f"Unique households:     {n_households:,}")
    print(f"Unique FSUs:           {n_fsus:,}")
    print(f"Positive rate:         {positive_rate:.1f}%")

    # Target checks
    checks = {
        'Total loans': (n_loans, 34230, 0.05),  # ±5% tolerance
        'Unique households': (n_households, 28914, 0.05),
        'Unique FSUs': (n_fsus, 8155, 0.05),
        'Positive rate (%)': (positive_rate, 16.0, 0.15),  # ±15% relative tolerance
    }

    print(f"\n=== TARGET VALIDATION ===")
    all_pass = True
    for metric, (actual, target, tolerance) in checks.items():
        diff_pct = abs(actual - target) / target
        status = '✓' if diff_pct <= tolerance else '✗'
        all_pass = all_pass and (diff_pct <= tolerance)
        print(f"{status} {metric:25s} Actual: {actual:>10,.1f}  Target: {target:>10,.1f}  Diff: {diff_pct*100:>6.1f}%")

    if not all_pass:
        print("\n⚠️  WARNING: Some metrics outside tolerance. Investigate before proceeding.")
    else:
        print("\n✓ All validation checks passed!")

    return df


def construct_sensitivity_labels(df):
    """
    Construct SENSITIVITY labels:
    (a) Write-off: b12q13 > 0
    (b) High outstanding: zero repayment AND b12q14 >= 1.1 × b12q15
    (c) Exclude tenure 1: primary label excluding loans with b12q7 == 1
    """
    # (a) Write-off
    df['npa_writeoff'] = (df['b12q13'] > 0).astype(int)

    # (b) High outstanding: zero repayment AND current > 1.1 × baseline
    zero_repayment = (df['b12q12'] == 0) | (df['b12q12'].isna())
    high_outstanding = (df['b12q14'] >= 1.1 * df['b12q15'])
    df['npa_high_outstanding'] = (zero_repayment & high_outstanding).astype(int)

    # (c) Exclude tenure 1
    df['npa_excl_tenure1'] = df['npa_primary'].copy()
    df.loc[df['b12q7'] == 1, 'npa_excl_tenure1'] = np.nan  # Mark as excluded

    # Report sensitivity label stats
    print(f"\n=== SENSITIVITY LABELS ===")
    print(f"(a) Write-off (b12q13>0):                {df['npa_writeoff'].sum():,} ({df['npa_writeoff'].mean()*100:.1f}%)")
    print(f"(b) High outstanding (0 repay + 1.1x):   {df['npa_high_outstanding'].sum():,} ({df['npa_high_outstanding'].mean()*100:.1f}%)")

    # For tenure exclusion, report on the subset
    excl_t1 = df[df['b12q7'] != 1].copy()
    print(f"(c) Excl. tenure 1:")
    print(f"    Remaining loans:                     {len(excl_t1):,}")
    print(f"    Positive rate:                       {excl_t1['npa_primary'].mean()*100:.1f}%")
    print(f"    (Expected ~35.5% if tenure 1 has lower default rate)")

    return df


def verify_leakage_protection(df):
    """Verify that outcome variables are not leaked into features"""
    leakage_vars = ['b12q12', 'b12q13', 'b12q14']

    print(f"\n=== LEAKAGE PROTECTION CHECK ===")
    print("The following variables are FORBIDDEN as features:")
    for var in leakage_vars:
        if var in df.columns:
            print(f"  - {var}: present in data (must NOT be used as feature)")
        else:
            print(f"  - {var}: not in data")

    print("\nNote: Any debt aggregates must use only b12q15 (outstanding on 30-06-2018)")
    print("      and only for loans with b12q3 == 1")


def create_labeled_dataset():
    """Main function to create labeled loan-level dataset"""
    print("="*70)
    print("P1: LABEL CONSTRUCTION - NSS 77th Round Credit Risk")
    print("="*70 + "\n")

    # Load loans
    print("Loading loan data...")
    loans = load_and_merge_loans()

    print(f"\n--- Visit 1 only for primary labels ---")
    loans_v1 = loans[loans['visit'] == 1].copy()
    print(f"Visit 1 loans: {len(loans_v1):,}")

    # Construct primary label
    print("\n" + "="*70)
    print("CONSTRUCTING PRIMARY LABEL")
    print("="*70)
    labeled = construct_primary_label(loans_v1)

    # Construct sensitivity labels
    labeled = construct_sensitivity_labels(labeled)

    # Verify leakage protection
    verify_leakage_protection(labeled)

    # Save to parquet
    output_path = 'data/labeled_loans_v1.parquet'
    labeled.to_parquet(output_path, index=False)
    print(f"\n✓ Saved labeled dataset to: {output_path}")

    return labeled


if __name__ == '__main__':
    labeled_df = create_labeled_dataset()

    # Quick EDA
    print("\n" + "="*70)
    print("QUICK EDA")
    print("="*70)
    print(f"\nLabel distribution:")
    print(labeled_df['npa_primary'].value_counts())
    print(f"\nBy credit agency (b12q5):")
    print(labeled_df.groupby('b12q5')['npa_primary'].agg(['count', 'sum', 'mean']).round(3))
