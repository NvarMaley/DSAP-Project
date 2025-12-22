import pandas as pd
import os
from pathlib import Path

# Get the project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'

def clean_economic_data():
    """
    Clean all economic indicator CSV files using linear interpolation
    to fill missing values
    """
    
    # Create processed folder if it doesn't exist
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    
    # List of economic indicator files to clean
    economic_files = [
        'taux_interet_38_pays.csv',
        'inflation_38_pays.csv',
        'chomage_38_pays.csv',
        'croissance_pib_38_pays.csv',
        'dette_publique_38_pays.csv',
        'solde_budgetaire_38_pays.csv',
        'balance_compte_courant_38_pays.csv',
        'reserves_change_38_pays.csv'
    ]
    
    print("=== CLEANING ECONOMIC INDICATORS ===\n")
    
    for file in economic_files:
        # Load CSV file
        df = pd.read_csv(DATA_RAW / file)
        
        # Count missing values before interpolation
        missing_before = df.isnull().sum().sum()
        
        # Apply linear interpolation to fill missing values
        # axis=1 means interpolate across columns (years)
        # limit_direction='both' fills gaps at beginning and end
        df_cleaned = df.copy()
        df_cleaned.iloc[:, 1:] = df_cleaned.iloc[:, 1:].interpolate(
            axis=1, 
            method='linear', 
            limit_direction='both'
        )
        
        # Count missing values after interpolation
        missing_after = df_cleaned.isnull().sum().sum()
        
        # Save cleaned file
        output_name = file.replace('_38_pays.csv', '_cleaned.csv')
        df_cleaned.to_csv(DATA_PROCESSED / output_name, index=False)
        
        # Print results
        print(f'✓ {file}')
        print(f'  Missing values: {missing_before} → {missing_after}')
        print(f'  Saved: data/processed/{output_name}\n')
    
    print("=== ECONOMIC INDICATORS CLEANED ===\n")


def clean_credit_ratings():
    """
    Clean credit ratings data and convert text ratings to numerical values
    """
    
    # Load credit ratings CSV
    df_ratings = pd.read_csv(DATA_RAW / 'notations_credit_38_pays.csv')
    
    print("=== CLEANING CREDIT RATINGS ===\n")
    print(f"Missing values: {df_ratings.isnull().sum().sum()}")
    
    # Create mapping dictionary: AAA=1 (best) to CC=20 (worst)
    rating_map = {
        'AAA': 1, 'AA+': 2, 'AA': 3, 'AA-': 4,
        'A+': 5, 'A': 6, 'A-': 7,
        'BBB+': 8, 'BBB': 9, 'BBB-': 10,
        'BB+': 11, 'BB': 12, 'BB-': 13,
        'B+': 14, 'B': 15, 'B-': 16,
        'CCC+': 17, 'CCC': 18, 'CCC-': 19, 'CC': 20
    }
    
    # Save text version (original ratings)
    df_ratings.to_csv(DATA_PROCESSED / 'notations_credit_cleaned.csv', index=False)
    print("✓ Text ratings saved: notations_credit_cleaned.csv")
    
    # Convert ratings to numerical values
    df_ratings_numerical = df_ratings.copy()
    for col in df_ratings_numerical.columns[1:]:
        df_ratings_numerical[col] = df_ratings_numerical[col].map(rating_map)
    
    # Save numerical version
    df_ratings_numerical.to_csv(DATA_PROCESSED / 'notations_credit_numerical.csv', index=False)
    print("✓ Numerical ratings saved: notations_credit_numerical.csv")
    print("\n=== CREDIT RATINGS CLEANED ===\n")


def merge_all_data():
    """
    Merge all cleaned CSV files into a single dataset
    Transforms from wide format (years as columns) to long format (years as rows)
    """
    
    print("=== MERGING ALL CLEANED DATA ===\n")
    
    # Load all cleaned CSV files
    print("Loading cleaned files...")
    df_interest = pd.read_csv(DATA_PROCESSED / 'taux_interet_cleaned.csv')
    df_inflation = pd.read_csv(DATA_PROCESSED / 'inflation_cleaned.csv')
    df_unemployment = pd.read_csv(DATA_PROCESSED / 'chomage_cleaned.csv')
    df_gdp = pd.read_csv(DATA_PROCESSED / 'croissance_pib_cleaned.csv')
    df_debt = pd.read_csv(DATA_PROCESSED / 'dette_publique_cleaned.csv')
    df_budget = pd.read_csv(DATA_PROCESSED / 'solde_budgetaire_cleaned.csv')
    df_current_account = pd.read_csv(DATA_PROCESSED / 'balance_compte_courant_cleaned.csv')
    df_reserves = pd.read_csv(DATA_PROCESSED / 'reserves_change_cleaned.csv')
    df_ratings = pd.read_csv(DATA_PROCESSED / 'notations_credit_numerical.csv')
    print("✓ All files loaded\n")
    
    # Helper function to transform from wide to long format
    def wide_to_long(df, value_name):
        # Melt: convert year columns to rows
        df_long = df.melt(id_vars=['Country'], var_name='Year', value_name=value_name)
        df_long['Year'] = df_long['Year'].astype(int)
        return df_long
    
    # Transform each dataframe
    print("Transforming data format...")
    df_interest_long = wide_to_long(df_interest, 'Interest_Rate')
    df_inflation_long = wide_to_long(df_inflation, 'Inflation')
    df_unemployment_long = wide_to_long(df_unemployment, 'Unemployment')
    df_gdp_long = wide_to_long(df_gdp, 'GDP_Growth')
    df_debt_long = wide_to_long(df_debt, 'Public_Debt')
    df_budget_long = wide_to_long(df_budget, 'Budget_Balance')
    df_current_account_long = wide_to_long(df_current_account, 'Current_Account')
    df_reserves_long = wide_to_long(df_reserves, 'FX_Reserves')
    df_ratings_long = wide_to_long(df_ratings, 'Credit_Rating')
    print("✓ Data transformed to long format\n")
    
    # Merge all dataframes on Country and Year
    print("Merging all datasets...")
    df_merged = df_interest_long
    df_merged = df_merged.merge(df_inflation_long, on=['Country', 'Year'])
    df_merged = df_merged.merge(df_unemployment_long, on=['Country', 'Year'])
    df_merged = df_merged.merge(df_gdp_long, on=['Country', 'Year'])
    df_merged = df_merged.merge(df_debt_long, on=['Country', 'Year'])
    df_merged = df_merged.merge(df_budget_long, on=['Country', 'Year'])
    df_merged = df_merged.merge(df_current_account_long, on=['Country', 'Year'])
    df_merged = df_merged.merge(df_reserves_long, on=['Country', 'Year'])
    df_merged = df_merged.merge(df_ratings_long, on=['Country', 'Year'])
    print("✓ All datasets merged\n")
    
    # Save merged dataset
    df_merged.to_csv(DATA_PROCESSED / 'merged_dataset.csv', index=False)
    
    print("=== MERGE COMPLETE ===")
    print(f"Final dataset shape: {df_merged.shape}")
    print(f"  - {df_merged.shape[0]} rows (country-year observations)")
    print(f"  - {df_merged.shape[1]} columns\n")
    print(f"Saved to: data/processed/merged_dataset.csv\n")
    
    return df_merged


def create_merged_dataset_labels():
    """
    Create merged dataset with credit rating LABELS (AAA, BB+, etc.) instead of numeric scores
    For Phase 2: Classification
    """
    
    print("=== CREATING MERGED DATASET WITH LABELS ===\n")
    
    # Load existing merged dataset (with numeric ratings)
    df_numeric = pd.read_csv(DATA_PROCESSED / 'merged_dataset.csv')
    print(f"✓ Loaded merged_dataset.csv (numeric ratings)")
    print(f"  Shape: {df_numeric.shape}")
    
    # Load credit ratings (labels)
    df_ratings = pd.read_csv(DATA_PROCESSED / 'notations_credit_cleaned.csv')
    print(f"✓ Loaded notations_credit_cleaned.csv (labels)")
    
    # Transform ratings to long format
    df_ratings_long = df_ratings.melt(id_vars=['Country'], var_name='Year', value_name='Credit_Rating_Label')
    df_ratings_long['Year'] = df_ratings_long['Year'].astype(int)
    print(f"✓ Transformed ratings to long format: {df_ratings_long.shape}")
    
    # Drop numeric Credit_Rating from df_numeric, keep only features
    df_features = df_numeric.drop('Credit_Rating', axis=1)
    print(f"✓ Dropped numeric Credit_Rating column")
    
    # Merge features with labels
    df_merged = pd.merge(df_features, df_ratings_long, on=['Country', 'Year'], how='inner')
    print(f"✓ Merged features with labels: {df_merged.shape}")
    
    # Reorder columns (Country, Year, features, then label)
    cols = ['Country', 'Year'] + [c for c in df_merged.columns if c not in ['Country', 'Year', 'Credit_Rating_Label']] + ['Credit_Rating_Label']
    df_merged = df_merged[cols]
    
    # Save to CSV
    df_merged.to_csv(DATA_PROCESSED / 'merged_dataset_labels.csv', index=False)
    
    print("\n=== MERGE WITH LABELS COMPLETE ===")
    print(f"Final dataset shape: {df_merged.shape}")
    print(f"  - {df_merged.shape[0]} rows (country-year observations)")
    print(f"  - {df_merged.shape[1]} columns")
    print(f"  - {df_merged['Credit_Rating_Label'].nunique()} unique credit rating labels")
    print(f"Saved to: data/processed/merged_dataset_labels.csv\n")
    
    return df_merged


def clean_all_data():
    """
    Main function to clean all data files
    """
    print("\n" + "="*50)
    print("STARTING DATA CLEANING PROCESS")
    print("="*50 + "\n")
    
    # Clean economic indicators
    clean_economic_data()
    
    # Clean credit ratings
    clean_credit_ratings()
    
    print("="*50)
    print("ALL DATA CLEANING COMPLETED")
    print("="*50 + "\n")


def process_all_data():
    """
    Complete pipeline: clean and merge all data
    Creates both numeric and label versions for Phase 1 (Regression) and Phase 2 (Classification)
    """
    # Clean all data
    clean_all_data()
    
    # Merge into single dataset (numeric ratings for Phase 1: Regression)
    merge_all_data()
    
    # Create dataset with labels (for Phase 2: Classification)
    create_merged_dataset_labels()


if __name__ == "__main__":
    # Run complete pipeline when script is executed directly
    process_all_data()
