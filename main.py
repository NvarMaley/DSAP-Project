"""
Main entry point for the Sovereign Risk Prediction project
"""

from src.data_loader import process_all_data
from src.models import run_linear_regression, run_linear_regression_kfold, run_polynomial_regression, run_ridge_regression, run_ridge_polynomial_regression
from src.classification import run_knn_classification, run_naive_bayes_classification


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
    
    # PHASE 2: CLASSIFICATION
    print("\n" + "="*60)
    print("PHASE 2: CLASSIFICATION")
    print("="*60 + "\n")
    
    # Step 14: k-NN (k=3)
    print("\nSTEP 14: k-NN Classification (k=3, K-Fold CV)")
    print("-" * 60)
    model_knn_k3, scaler_knn_k3, acc_k3, std_acc_k3, f1_k3, std_f1_k3 = run_knn_classification(k=3, cv=5)
    
    # Step 15: k-NN (k=5)
    print("\nSTEP 15: k-NN Classification (k=5, K-Fold CV)")
    print("-" * 60)
    model_knn_k5, scaler_knn_k5, acc_k5, std_acc_k5, f1_k5, std_f1_k5 = run_knn_classification(k=5, cv=5)
    
    # Step 16: k-NN (k=7)
    print("\nSTEP 16: k-NN Classification (k=7, K-Fold CV)")
    print("-" * 60)
    model_knn_k7, scaler_knn_k7, acc_k7, std_acc_k7, f1_k7, std_f1_k7 = run_knn_classification(k=7, cv=5)
    
    # Step 17: k-NN (k=9)
    print("\nSTEP 17: k-NN Classification (k=9, K-Fold CV)")
    print("-" * 60)
    model_knn_k9, scaler_knn_k9, acc_k9, std_acc_k9, f1_k9, std_f1_k9 = run_knn_classification(k=9, cv=5)
    
    # Step 18: Naive Bayes
    print("\nSTEP 18: Naive Bayes Classification (K-Fold CV)")
    print("-" * 60)
    model_nb, acc_nb, std_acc_nb, f1_nb, std_f1_nb = run_naive_bayes_classification(cv=5)
    
    print("\n" + "="*60)
    print("PROJECT PIPELINE COMPLETE")
    print("="*60 + "\n")
    print("Phase 1 (Regression) - Best Model: Ridge + Polynomial (alpha=10, deg=2)")
    print(f"  R² = {r2_rp10:.4f}")
    print("\nPhase 2 (Classification) - Best Model: k-NN (k=3)")
    print(f"  Accuracy = {acc_k3:.4f}")
    print(f"  F1-Weighted = {f1_k3:.4f}")
    print("\n")
    
    return model_simple, model_kfold, model_poly2, model_poly3, model_ridge_10, model_rp_10, model_knn_k3, model_nb


if __name__ == "__main__":
    # Run main pipeline
    main()
