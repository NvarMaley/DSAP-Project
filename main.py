"""
Main entry point for the Sovereign Risk Prediction project
"""

from src.data_loader import process_all_data
from src.models import run_linear_regression, run_linear_regression_kfold, run_polynomial_regression, run_ridge_regression, run_ridge_polynomial_regression


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
    
    # Step 6: Ridge Regression (alpha=0.01)
    print("\nSTEP 6: Ridge Regression (alpha=0.01, K-Fold CV)")
    print("-" * 60)
    model_ridge_001, scaler_001, rmse_r001, mae_r001, r2_r001 = run_ridge_regression(alpha=0.01, k=5)
    
    # Step 7: Ridge Regression (alpha=0.1)
    print("\nSTEP 7: Ridge Regression (alpha=0.1, K-Fold CV)")
    print("-" * 60)
    model_ridge_01, scaler_01, rmse_r01, mae_r01, r2_r01 = run_ridge_regression(alpha=0.1, k=5)
    
    # Step 8: Ridge Regression (alpha=1.0)
    print("\nSTEP 8: Ridge Regression (alpha=1.0, K-Fold CV)")
    print("-" * 60)
    model_ridge_1, scaler_1, rmse_r1, mae_r1, r2_r1 = run_ridge_regression(alpha=1.0, k=5)
    
    # Step 9: Ridge Regression (alpha=10.0)
    print("\nSTEP 9: Ridge Regression (alpha=10.0, K-Fold CV)")
    print("-" * 60)
    model_ridge_10, scaler_10, rmse_r10, mae_r10, r2_r10 = run_ridge_regression(alpha=10.0, k=5)
    
    # Step 10: Ridge + Polynomial (alpha=0.01, deg=2)
    print("\nSTEP 10: Ridge + Polynomial (alpha=0.01, deg=2, K-Fold CV)")
    print("-" * 60)
    model_rp_001, poly_rp_001, scaler_rp_001, rmse_rp001, mae_rp001, r2_rp001 = run_ridge_polynomial_regression(alpha=0.01, degree=2, k=5)
    
    # Step 11: Ridge + Polynomial (alpha=0.1, deg=2)
    print("\nSTEP 11: Ridge + Polynomial (alpha=0.1, deg=2, K-Fold CV)")
    print("-" * 60)
    model_rp_01, poly_rp_01, scaler_rp_01, rmse_rp01, mae_rp01, r2_rp01 = run_ridge_polynomial_regression(alpha=0.1, degree=2, k=5)
    
    # Step 12: Ridge + Polynomial (alpha=1.0, deg=2)
    print("\nSTEP 12: Ridge + Polynomial (alpha=1.0, deg=2, K-Fold CV)")
    print("-" * 60)
    model_rp_1, poly_rp_1, scaler_rp_1, rmse_rp1, mae_rp1, r2_rp1 = run_ridge_polynomial_regression(alpha=1.0, degree=2, k=5)
    
    # Step 13: Ridge + Polynomial (alpha=10.0, deg=2) - BEST MODEL
    print("\nSTEP 13: Ridge + Polynomial (alpha=10.0, deg=2, K-Fold CV) - BEST MODEL")
    print("-" * 60)
    model_rp_10, poly_rp_10, scaler_rp_10, rmse_rp10, mae_rp10, r2_rp10 = run_ridge_polynomial_regression(alpha=10.0, degree=2, k=5)
    
    print("\n" + "="*60)
    print("PROJECT PIPELINE COMPLETE")
    print("="*60 + "\n")
    
    return model_simple, model_kfold, model_poly2, model_poly3, model_ridge_10, model_rp_10


if __name__ == "__main__":
    # Run main pipeline
    main()
