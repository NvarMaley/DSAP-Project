"""
Main entry point for the Sovereign Risk Prediction project
"""

from src.data_loader import process_all_data
from src.models import run_linear_regression, run_linear_regression_kfold


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
    
    # Step 2: Phase 1 - Linear Regression (Simple Split)
    print("\nSTEP 2: Phase 1 - Linear Regression (Simple Split)")
    print("-" * 60)
    model_simple = run_linear_regression()
    
    # Step 3: Linear Regression with K-Fold Cross-Validation
    print("\nSTEP 3: Linear Regression (K-Fold CV)")
    print("-" * 60)
    model_kfold, rmse_scores, mae_scores, r2_scores = run_linear_regression_kfold(k=5)
    
    print("\n" + "="*60)
    print("PROJECT PIPELINE COMPLETE")
    print("="*60 + "\n")
    
    return model_simple, model_kfold


if __name__ == "__main__":
    # Run main pipeline
    main()
