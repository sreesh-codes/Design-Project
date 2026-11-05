"""
P1: Data loading and cleaning for NSS 77th Round Credit Risk Analysis
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class NSS77DataLoader:
    """Load and clean NSS 77th Round Debt & Investment Survey data"""

    def __init__(self, data_dir='data/CSV_DI_77/CSV_DI_77'):
        self.data_dir = Path(data_dir)

    def load_block12_visit1(self):
        """Load Visit 1 Block 12 (cash loans)"""
        df = pd.read_csv(self.data_dir / 'Visit1_Level_14_Block_12.csv', low_memory=False)

        # Rename columns to lowercase for consistency first
        df.columns = [c.lower() for c in df.columns]

        # Drop household total rows (b12q1 == 99)
        df = df[df['b12q1'] != 99].copy()

        # Add visit indicator
        df['visit'] = 1

        return df

    def load_block12_visit2(self):
        """Load Visit 2 Block 12 (cash loans)"""
        df = pd.read_csv(self.data_dir / 'Visit2  Level - 14 (Block 12).csv', low_memory=False)

        # Rename columns to lowercase
        df.columns = [c.lower() for c in df.columns]

        # Drop household total rows if present
        if 'b12q1' in df.columns:
            df = df[df['b12q1'] != 99].copy()

        # Add visit indicator
        df['visit'] = 2

        return df

    def load_level01_visit1(self):
        """Load Visit 1 Level 01 (household identification)"""
        df = pd.read_csv(self.data_dir / 'Visit1  Level - 01 (Blocks 1 and 2) - Identification of sample household and  particulars of field operations.csv', low_memory=False)
        df.columns = [c.lower() for c in df.columns]
        df['visit'] = 1
        return df

    def load_level01_visit2(self):
        """Load Visit 2 Level 01 (household identification)"""
        df = pd.read_csv(self.data_dir / 'Visit2  Level - 01 (Blocks 1 and 2) - Identification of sample household and  particulars of field operations.csv', low_memory=False)
        df.columns = [c.lower() for c in df.columns]
        df['visit'] = 2
        return df

    def load_level02(self):
        """Load Level 02 (household members)"""
        df = pd.read_csv(self.data_dir / 'Visit1  Level - 02 (Block 3) - Demographic and other particulars of household members.csv', low_memory=False)
        df.columns = [c.lower() for c in df.columns]
        return df

    def load_level03(self):
        """Load Level 03 (household characteristics)"""
        df = pd.read_csv(self.data_dir / 'Visit1  Level - 03 (Block 4) - Household characteristics.csv', low_memory=False)
        df.columns = [c.lower() for c in df.columns]
        return df

    def load_level04(self):
        """Load Level 04 (consumption)"""
        # Find the file with consumption data
        files = list(self.data_dir.glob('*Level*04*'))
        if not files:
            # Try alternate naming
            files = list(self.data_dir.glob('*Block*4*'))
            # Filter for the one that's not already loaded
            files = [f for f in files if 'Level - 03' not in f.name]

        if files:
            df = pd.read_csv(files[0])
            df.columns = [c.lower() for c in df.columns]
            return df
        return None


def create_household_key(df):
    """Create household key by removing last digit of HHID"""
    if 'hhid' in df.columns:
        df['household_key'] = df['hhid'].astype(str).str[:-1]
    return df


def load_and_merge_loans():
    """Load both visits of loan data and merge with identifiers"""
    loader = NSS77DataLoader()

    # Load Block 12 data
    print("Loading Visit 1 Block 12...")
    v1_loans = loader.load_block12_visit1()
    print(f"  Raw rows: {len(v1_loans):,}")

    print("Loading Visit 2 Block 12...")
    v2_loans = loader.load_block12_visit2()
    print(f"  Raw rows: {len(v2_loans):,}")

    # Load Level 01 for identifiers
    print("Loading household identifiers...")
    v1_id = loader.load_level01_visit1()
    v2_id = loader.load_level01_visit2()

    # Create household keys
    v1_loans = create_household_key(v1_loans)
    v2_loans = create_household_key(v2_loans)
    v1_id = create_household_key(v1_id)
    v2_id = create_household_key(v2_id)

    # Merge with identifiers to get FSU, sector, state, multiplier
    id_cols = ['hhid', 'household_key', 'fsu_slno', 'sector', 'state', 'mlt']

    # Check which columns exist
    available_cols_v1 = [c for c in id_cols if c in v1_id.columns]
    available_cols_v2 = [c for c in id_cols if c in v2_id.columns]

    print(f"\nAvailable ID columns V1: {available_cols_v1}")
    print(f"Available ID columns V2: {available_cols_v2}")

    # Merge
    v1_merged = v1_loans.merge(v1_id[available_cols_v1], on='hhid', how='left', suffixes=('', '_id'))
    v2_merged = v2_loans.merge(v2_id[available_cols_v2], on='hhid', how='left', suffixes=('', '_id'))

    # Combine both visits
    loans = pd.concat([v1_merged, v2_merged], ignore_index=True)

    print(f"\nCombined loan records: {len(loans):,}")
    print(f"Unique households: {loans['household_key'].nunique():,}")
    print(f"Unique FSUs: {loans['fsu_slno'].nunique():,}")

    return loans


if __name__ == '__main__':
    # Test loading
    loans = load_and_merge_loans()
    print("\nColumn names:")
    print(loans.columns.tolist())
    print(f"\nSample data:")
    print(loans.head())
    print(f"\nData types:")
    print(loans.dtypes)
