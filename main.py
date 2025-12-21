"""
Main entry point for the Sovereign Risk Prediction project
"""

from src.data_loader import process_all_data
from src.models import run_linear_regression_pipeline


def main():
    """
    Main function to run the complete project pipeline
    """
    print("\n" + "="*60)
    print("SOVEREIGN RISK PREDICTION PROJECT")
    print("="*60 + "\n")
    
    # Step 1: Data cleaning and merging
    print("STEP 1: Data Cleaning and Merging")
    print("-" * 60)
    process_all_data()
    
    # Step 2: Phase 1 - Linear Regression
    print("\nSTEP 2: Phase 1 - Linear Regression")
    print("-" * 60)
    results = run_linear_regression_pipeline()
    
    print("\n" + "="*60)
    print("PROJECT PIPELINE COMPLETE")
    print("="*60 + "\n")
    
    return results


if __name__ == "__main__":
    # Run main pipeline
    main()
