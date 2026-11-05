"""
P2: Feature engineering for NSS 77th Round Credit Risk Analysis
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import NSS77DataLoader


class FeatureEngineer:
    """Engineer features from NSS 77 household microdata"""

    def __init__(self, data_dir='data/CSV_DI_77/CSV_DI_77'):
        self.loader = NSS77DataLoader(data_dir)
        self.excluded_vars = ['b12q12', 'b12q13', 'b12q14']  # Leakage

    def load_household_features(self):
        """Load all household-level feature files"""
        print("Loading household feature files...")

        # Level 02: Members (demographics)
        members = self.loader.load_level02()
        print(f"  Level 02 (members): {len(members):,} rows")

        # Level 03: Household characteristics
        hh_char = self.loader.load_level03()
        print(f"  Level 03 (HH char): {len(hh_char):,} rows")

        # Level 04: Consumption - need to find it
        files = list(Path(self.loader.data_dir).glob('*Level*04*'))
        if files:
            consumption = pd.read_csv(files[0], low_memory=False)
            consumption.columns = [c.lower() for c in consumption.columns]
            print(f"  Level 04 (consumption): {len(consumption):,} rows")
        else:
            consumption = None
            print("  Level 04 (consumption): NOT FOUND")

        # Asset files: Levels 05-12
        asset_files = {
            'land_rural': 'Visit1  Level - 05 (Block 5pt1) - RURAL LAND owned by the household .csv',
            'land_urban': 'Visit1  Level - 06 (Block 5_2) -URBAN LAND owned by the household.csv',
            'buildings': 'Visit1  Level - 07 (Block 6) - Buildings and other constructions owned by the household.csv',
            'livestock': 'Visit1  Level - 08 (Block 7) - Livestock and poultry owned by the household.csv',
            'transport': 'Visit1  Level - 09 (Block 8) - Transport equipment owned by the household.csv',
            'agri_machinery': 'Visit1  Level - 10 (Block 9) - Agricultural machinery and implements owned by the household.csv',
            'business_equip': 'Visit1  Level - 11 (Block 10) - Non-farm business equipment fully owned by the household.csv',
            'financial_assets': 'Visit1  Level - 12 (Block 11a) - Financial assets including receivables (other than shares and related instruments) owned by the household_.csv',
        }

        assets = {}
        for name, fname in asset_files.items():
            try:
                df = pd.read_csv(self.loader.data_dir / fname, low_memory=False)
                df.columns = [c.lower() for c in df.columns]
                assets[name] = df
                print(f"  {name}: {len(df):,} rows")
            except FileNotFoundError:
                print(f"  {name}: NOT FOUND")
                assets[name] = None

        # Visit 2 Level 18: Insurance
        try:
            insurance = pd.read_csv(
                self.loader.data_dir / 'Visit2  Level - 18 (Block 11c) - premium paid by the household for life and non-life insurance.csv',
                low_memory=False
            )
            insurance.columns = [c.lower() for c in insurance.columns]
            print(f"  Visit2 Level 18 (insurance): {len(insurance):,} rows")
        except FileNotFoundError:
            insurance = None
            print("  Visit2 Level 18 (insurance): NOT FOUND")

        return {
            'members': members,
            'hh_char': hh_char,
            'consumption': consumption,
            'assets': assets,
            'insurance': insurance
        }

    def engineer_loan_features(self, loans):
        """Engineer loan-level features"""
        df = loans.copy()

        print("\n=== Engineering Loan Features ===")

        # Original amount (already present as b12q4)
        df['loan_amount'] = df['b12q4'].fillna(0)

        # Outstanding amount (b12q15 - safe to use)
        df['loan_outstanding'] = df['b12q15'].fillna(0)

        # Interest rate
        df['interest_rate'] = df['b12q9'].fillna(0)
        df['interest_rate_missing'] = df['b12q9'].isna().astype(int)

        # Year borrowed → exposure months (from 30-06-2018 backward)
        # Assume loans taken mid-year
        df['year_borrowed'] = df['b12q2']
        df['loan_age_years'] = 2018 - df['b12q2'].fillna(2018)
        df['exposure_months'] = df['loan_age_years'] * 12

        # Categorical features
        df['agency_type'] = df['b12q5'].astype('category')
        df['loan_scheme'] = df['b12q6'].astype('category')
        df['tenure_code'] = df['b12q7'].astype('category')
        df['interest_type'] = df['b12q8'].astype('category')
        df['purpose_code'] = df['b12q10'].astype('category')
        df['secured'] = df['b12q11'].astype('category')

        # Derived: institutional vs informal (already filtered to institutional for labels)
        df['is_institutional'] = (df['b12q5'] <= 13).astype(int)

        # Sector and state
        df['sector'] = df['sector'].astype('category')  # 1=rural, 2=urban
        df['state'] = df['state'].astype('category')

        print(f"  Loan features engineered: {len(df):,} rows")
        return df

    def engineer_borrower_features(self, loans, members, hh_char):
        """Engineer borrower (household head) features"""
        print("\n=== Engineering Borrower Features ===")

        # Create household key in members
        if 'hhid' in members.columns:
            members['household_key'] = members['hhid'].astype(str).str[:-1]

        # Find household head (b3q3 == 1)
        heads = members[members['b3q3'] == 1].copy()
        print(f"  Household heads identified: {len(heads):,}")

        # Extract head demographics (EXCLUDE religion and social group)
        head_features = heads[['household_key']].copy()

        # Age
        if 'b3q5' in heads.columns:
            head_features['head_age'] = heads['b3q5']
        # Gender
        if 'b3q4' in heads.columns:
            head_features['head_gender'] = heads['b3q4'].astype('category')
        # Education
        if 'b3q7' in heads.columns:
            head_features['head_education'] = heads['b3q7'].astype('category')
        # Marital status
        if 'b3q6' in heads.columns:
            head_features['head_marital_status'] = heads['b3q6'].astype('category')

        # Household size
        hh_size = members.groupby('household_key').size().reset_index(name='hh_size')
        head_features = head_features.merge(hh_size, on='household_key', how='left')

        # Merge with loans
        loans = loans.merge(head_features, on='household_key', how='left')
        print(f"  Borrower features merged: {len(loans):,} rows")

        return loans

    def engineer_balance_sheet_features(self, loans, consumption, assets):
        """Engineer balance sheet features: debt/assets, debt/consumption, liquidity"""
        print("\n=== Engineering Balance Sheet Features ===")

        # Aggregate household debt (SAFE: only b12q3==1, use b12q15)
        # Use all loans for same household, not just labeled subset
        debt_agg = loans.groupby('household_key').agg({
            'b12q15': 'sum',  # Total outstanding
            'b12q4': 'sum',   # Total original
            'b12q5': 'count'  # Number of loans
        }).reset_index()
        debt_agg.columns = ['household_key', 'total_debt_outstanding', 'total_debt_original', 'num_loans']

        print(f"  Debt aggregates computed: {len(debt_agg):,} households")

        # Monthly consumption (b4q10dot5 or similar)
        if consumption is not None and 'household_key' not in consumption.columns:
            consumption['household_key'] = consumption['hhid'].astype(str).str[:-1]

        monthly_cons = None
        if consumption is not None:
            # Find consumption total column
            cons_cols = [c for c in consumption.columns if 'b4q' in c and 'dot5' in c]
            if not cons_cols:
                cons_cols = [c for c in consumption.columns if 'b4q10' in c]
            if cons_cols:
                consumption['monthly_consumption'] = consumption[cons_cols[0]]
                monthly_cons = consumption[['household_key', 'monthly_consumption']].copy()
                print(f"  Monthly consumption found: {len(monthly_cons):,} households")
            else:
                print("  Monthly consumption column not found")
        else:
            print("  Consumption data not available")

        # Total assets from asset files
        asset_totals = []
        for name, df in assets.items():
            if df is not None and 'household_key' not in df.columns and 'hhid' in df.columns:
                df['household_key'] = df['hhid'].astype(str).str[:-1]

            if df is not None:
                # Find value columns (typically end with value, worth, etc.)
                value_cols = [c for c in df.columns if 'value' in c.lower() or 'worth' in c.lower()]
                if value_cols:
                    # Drop total rows if present
                    asset_df = df.copy()
                    for col in df.columns:
                        if 'q1' in col.lower() or 'srl' in col.lower():
                            asset_df = asset_df[asset_df[col] != 99]

                    # Sum by household
                    asset_sum = asset_df.groupby('household_key')[value_cols].sum().sum(axis=1).reset_index(name=f'{name}_value')
                    asset_totals.append(asset_sum)
                    print(f"  {name} aggregated: {len(asset_sum):,} households")

        # Merge all asset totals
        total_assets = None
        if asset_totals:
            total_assets = asset_totals[0]
            for at in asset_totals[1:]:
                total_assets = total_assets.merge(at, on='household_key', how='outer')
            total_assets['total_assets'] = total_assets[[c for c in total_assets.columns if c != 'household_key']].fillna(0).sum(axis=1)
            total_assets = total_assets[['household_key', 'total_assets']]
            print(f"  Total assets computed: {len(total_assets):,} households")

        # Merge balance sheet features
        balance_sheet = debt_agg.copy()
        if monthly_cons is not None:
            balance_sheet = balance_sheet.merge(monthly_cons, on='household_key', how='left')
            balance_sheet['monthly_consumption'] = balance_sheet['monthly_consumption'].fillna(0)
        else:
            balance_sheet['monthly_consumption'] = 0

        if total_assets is not None:
            balance_sheet = balance_sheet.merge(total_assets, on='household_key', how='left')
            balance_sheet['total_assets'] = balance_sheet['total_assets'].fillna(0)
        else:
            balance_sheet['total_assets'] = 0

        # Debt to income (proxy: 12 * monthly consumption)
        balance_sheet['annual_consumption'] = balance_sheet['monthly_consumption'] * 12
        balance_sheet['debt_to_income'] = np.where(
            balance_sheet['annual_consumption'] > 0,
            balance_sheet['total_debt_outstanding'] / balance_sheet['annual_consumption'],
            np.nan
        )

        # Debt to assets
        balance_sheet['debt_to_assets'] = np.where(
            balance_sheet['total_assets'] > 0,
            balance_sheet['total_debt_outstanding'] / balance_sheet['total_assets'],
            np.nan
        )

        # Liquid assets to debt (use financial assets)
        liquid_assets = None
        if 'financial_assets' in assets and assets['financial_assets'] is not None:
            fa = assets['financial_assets'].copy()
            if 'household_key' not in fa.columns and 'hhid' in fa.columns:
                fa['household_key'] = fa['hhid'].astype(str).str[:-1]
            value_cols = [c for c in fa.columns if 'value' in c.lower()]
            if value_cols:
                liquid_assets = fa.groupby('household_key')[value_cols].sum().sum(axis=1).reset_index(name='liquid_assets')
                balance_sheet = balance_sheet.merge(liquid_assets, on='household_key', how='left')
                balance_sheet['liquid_assets'] = balance_sheet['liquid_assets'].fillna(0)
                balance_sheet['liquidity_ratio'] = np.where(
                    balance_sheet['total_debt_outstanding'] > 0,
                    balance_sheet['liquid_assets'] / balance_sheet['total_debt_outstanding'],
                    np.nan
                )
                print(f"  Liquidity ratio computed")

        # Merge with loans
        loans = loans.merge(balance_sheet, on='household_key', how='left')
        print(f"  Balance sheet features merged: {len(loans):,} rows")

        return loans

    def engineer_portfolio_features(self, loans_all, loans):
        """
        Engineer portfolio/informality features
        loans_all: full loan dataset including informal
        loans: institutional loans only (labeled subset)
        """
        print("\n=== Engineering Portfolio/Informality Features ===")

        # Load ALL loans (not just institutional)
        # We need to reload to include informal loans
        from src.data import load_and_merge_loans
        all_loans = load_and_merge_loans()
        all_loans = all_loans[all_loans['visit'] == 1].copy()  # Visit 1 only
        all_loans = all_loans[all_loans['b12q3'] == 1].copy()  # Outstanding on 30-06-2018

        print(f"  All outstanding loans (institutional + informal): {len(all_loans):,}")

        # Count concurrent loans by household
        loan_counts = all_loans.groupby('household_key').agg({
            'b12q1': 'count',
            'b12q15': 'sum'
        }).reset_index()
        loan_counts.columns = ['household_key', 'concurrent_loans_count', 'total_debt_all']

        # Informal loan indicators
        all_loans['is_informal'] = (all_loans['b12q5'] > 13).astype(int)
        all_loans['is_moneylender'] = all_loans['b12q5'].isin([15, 16]).astype(int)

        informal_agg = all_loans.groupby('household_key').agg({
            'is_informal': 'sum',
            'is_moneylender': 'sum',
            'b12q9': 'max'  # Max interest rate
        }).reset_index()
        informal_agg.columns = ['household_key', 'num_informal_loans', 'num_moneylender_loans', 'max_interest_rate']

        # Informal debt share
        informal_debt = all_loans[all_loans['is_informal'] == 1].groupby('household_key')['b12q15'].sum().reset_index(name='informal_debt')
        informal_agg = informal_agg.merge(informal_debt, on='household_key', how='left')
        informal_agg['informal_debt'] = informal_agg['informal_debt'].fillna(0)

        informal_agg = informal_agg.merge(loan_counts[['household_key', 'total_debt_all']], on='household_key', how='left')
        informal_agg['informal_debt_share'] = np.where(
            informal_agg['total_debt_all'] > 0,
            informal_agg['informal_debt'] / informal_agg['total_debt_all'],
            0
        )

        # Merge portfolio features
        portfolio = loan_counts.merge(informal_agg, on='household_key', how='left')
        portfolio[['num_informal_loans', 'num_moneylender_loans']] = portfolio[['num_informal_loans', 'num_moneylender_loans']].fillna(0)
        portfolio['max_interest_rate'] = portfolio['max_interest_rate'].fillna(0)

        loans = loans.merge(portfolio, on='household_key', how='left')
        print(f"  Portfolio features merged: {len(loans):,} rows")

        return loans


def create_feature_dataset():
    """Main function to create feature-engineered dataset"""
    print("="*70)
    print("P2: FEATURE ENGINEERING")
    print("="*70 + "\n")

    # Load labeled loans
    print("Loading labeled loans...")
    loans = pd.read_parquet('data/labeled_loans_v1.parquet')
    print(f"Labeled loans: {len(loans):,}\n")

    # Initialize feature engineer
    fe = FeatureEngineer()

    # Load household features
    hh_features = fe.load_household_features()

    # Engineer features
    loans = fe.engineer_loan_features(loans)
    loans = fe.engineer_borrower_features(loans, hh_features['members'], hh_features['hh_char'])
    loans = fe.engineer_balance_sheet_features(loans, hh_features['consumption'], hh_features['assets'])
    loans = fe.engineer_portfolio_features(None, loans)

    # Save
    output_path = 'data/features_loans_v1.parquet'
    loans.to_parquet(output_path, index=False)
    print(f"\n✓ Saved feature dataset to: {output_path}")

    return loans


if __name__ == '__main__':
    feature_df = create_feature_dataset()

    print("\n" + "="*70)
    print("FEATURE SUMMARY")
    print("="*70)
    print(f"\nTotal features: {len(feature_df.columns)}")
    print(f"Total rows: {len(feature_df):,}")
    print("\nFeature groups:")
    print(f"  Loan: loan_amount, loan_outstanding, interest_rate, exposure_months, etc.")
    print(f"  Borrower: head_age, head_gender, head_education, hh_size")
    print(f"  Balance sheet: debt_to_income, debt_to_assets, liquidity_ratio")
    print(f"  Portfolio: concurrent_loans_count, informal_debt_share, max_interest_rate")
    print(f"  Context: state, sector")
