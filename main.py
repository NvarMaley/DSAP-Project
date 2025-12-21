"""
Main entry point for the Sovereign Risk Prediction project
"""

from src.data_loader import process_all_data
from src.models import run_linear_regression, run_linear_regression_kfold, run_polynomial_regression


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
    
    # Step 4: Polynomial Regression (degree 2)
    print("\nSTEP 4: Polynomial Regression (degree=2, K-Fold CV)")
    print("-" * 60)
    model_poly2, poly2, rmse2, mae2, r2_2 = run_polynomial_regression(degree=2, k=5)
    
    # Step 5: Polynomial Regression (degree 3)
    print("\nSTEP 5: Polynomial Regression (degree=3, K-Fold CV)")
    print("-" * 60)
    model_poly3, poly3, rmse3, mae3, r2_3 = run_polynomial_regression(degree=3, k=5)
    
    print("\n" + "="*60)
    print("PROJECT PIPELINE COMPLETE")
    print("="*60 + "\n")
    
    return model_simple, model_kfold, model_poly2, model_poly3


if __name__ == "__main__":
    # Run main pipeline
    main()
