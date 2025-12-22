"""
Main entry point for the Sovereign Risk Prediction project
"""

from src.data_loader import process_all_data
from src.models import run_linear_regression, run_linear_regression_kfold, run_polynomial_regression, run_ridge_regression, run_ridge_polynomial_regression
from src.classification import run_knn_classification, run_naive_bayes_classification, run_decision_tree_classification, run_random_forest_classification
from src.unsupervised import find_optimal_k, run_kmeans_clustering, compare_clusters_with_ratings, visualize_clusters_2d, run_pca_analysis, visualize_pca_3d, create_biplot, analyze_feature_correlations
from src.deep_learning import run_mlp_simple, run_mlp_improved, compare_mlp_with_classical, compare_optimizers, compare_learning_rates, compare_l2_regularization


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
    
    # Step 19: Decision Tree (depth=3)
    print("\nSTEP 19: Decision Tree Classification (depth=3, K-Fold CV)")
    print("-" * 60)
    model_dt3, acc_dt3, std_acc_dt3, f1_dt3, std_f1_dt3 = run_decision_tree_classification(max_depth=3, cv=5)
    
    # Step 20: Decision Tree (depth=5)
    print("\nSTEP 20: Decision Tree Classification (depth=5, K-Fold CV)")
    print("-" * 60)
    model_dt5, acc_dt5, std_acc_dt5, f1_dt5, std_f1_dt5 = run_decision_tree_classification(max_depth=5, cv=5)
    
    # Step 21: Decision Tree (depth=10)
    print("\nSTEP 21: Decision Tree Classification (depth=10, K-Fold CV)")
    print("-" * 60)
    model_dt10, acc_dt10, std_acc_dt10, f1_dt10, std_f1_dt10 = run_decision_tree_classification(max_depth=10, cv=5)
    
    # Step 22: Random Forest (n=50)
    print("\nSTEP 22: Random Forest Classification (n=50, K-Fold CV)")
    print("-" * 60)
    model_rf50, acc_rf50, std_acc_rf50, f1_rf50, std_f1_rf50 = run_random_forest_classification(n_estimators=50, cv=5)
    
    # Step 23: Random Forest (n=100)
    print("\nSTEP 23: Random Forest Classification (n=100, K-Fold CV)")
    print("-" * 60)
    model_rf100, acc_rf100, std_acc_rf100, f1_rf100, std_f1_rf100 = run_random_forest_classification(n_estimators=100, cv=5)
    
    # Step 24: Random Forest (n=200)
    print("\nSTEP 24: Random Forest Classification (n=200, K-Fold CV)")
    print("-" * 60)
    model_rf200, acc_rf200, std_acc_rf200, f1_rf200, std_f1_rf200 = run_random_forest_classification(n_estimators=200, cv=5)
    
    # PHASE 3: UNSUPERVISED LEARNING
    print("\n" + "="*60)
    print("PHASE 3: UNSUPERVISED LEARNING")
    print("="*60 + "\n")
    
    # Step 25: Find Optimal K for K-Means
    print("\nSTEP 25: K-Means Elbow Method (Find Optimal K)")
    print("-" * 60)
    optimal_k = find_optimal_k(k_range=[2, 3, 4, 5, 6, 7, 8])
    
    # Step 26: K-Means Clustering
    print(f"\nSTEP 26: K-Means Clustering (K={optimal_k})")
    print("-" * 60)
    kmeans_model, cluster_results = run_kmeans_clustering(n_clusters=optimal_k)
    
    # Step 27: Compare Clusters with Ratings
    print(f"\nSTEP 27: Compare Clusters with Credit Ratings")
    print("-" * 60)
    comparison_metrics = compare_clusters_with_ratings(n_clusters=optimal_k)
    
    # Step 28: Visualize Clusters in 2D
    print(f"\nSTEP 28: Visualize Clusters in 2D (PCA)")
    print("-" * 60)
    visualize_clusters_2d(n_clusters=optimal_k)
    
    # Step 29: PCA Analysis
    print("\nSTEP 29: PCA Analysis (Variance Explained & Loadings)")
    print("-" * 60)
    pca_model, X_pca, variance_explained = run_pca_analysis()
    
    # Step 30: PCA 3D Visualization
    print("\nSTEP 30: PCA 3D Visualization")
    print("-" * 60)
    visualize_pca_3d()
    
    # Step 31: Biplot
    print("\nSTEP 31: PCA Biplot (Observations + Features)")
    print("-" * 60)
    create_biplot()
    
    # Step 32: Feature Correlations
    print("\nSTEP 32: Feature Correlations Analysis")
    print("-" * 60)
    analyze_feature_correlations()
    
    # Step 33: MLP Simple
    print("\nSTEP 33: Deep Learning - MLP Simple")
    print("-" * 60)
    model_mlp_simple, history_simple, metrics_simple, le_simple = run_mlp_simple()
    acc_mlp_simple = metrics_simple['Accuracy'].values[0]
    f1_mlp_simple = metrics_simple['F1_Weighted'].values[0]
    
    # Step 34: MLP Improved
    print("\nSTEP 34: Deep Learning - MLP Improved")
    print("-" * 60)
    model_mlp_improved, history_improved, metrics_improved, le_improved = run_mlp_improved()
    acc_mlp_improved = metrics_improved['Accuracy'].values[0]
    f1_mlp_improved = metrics_improved['F1_Weighted'].values[0]
    
    # Step 35: Compare MLP with Classical ML
    print("\nSTEP 35: Compare MLP with Classical ML")
    print("-" * 60)
    comparison_ml = compare_mlp_with_classical()
    
    # Step 36: Compare Optimizers
    print("\nSTEP 36: Training Techniques - Compare Optimizers")
    print("-" * 60)
    results_optimizers = compare_optimizers()
    best_optimizer = results_optimizers.iloc[0]['Optimizer']
    best_opt_acc = results_optimizers.iloc[0]['Accuracy']
    
    # Step 37: Compare Learning Rates
    print("\nSTEP 37: Training Techniques - Compare Learning Rates")
    print("-" * 60)
    results_lr = compare_learning_rates()
    best_lr = results_lr.iloc[0]['Learning_Rate']
    best_lr_acc = results_lr.iloc[0]['Accuracy']
    
    # Step 38: Compare L2 Regularization
    print("\nSTEP 38: Training Techniques - Compare L2 Regularization")
    print("-" * 60)
    results_l2 = compare_l2_regularization()
    best_l2 = results_l2.iloc[0]['L2_Strength']
    best_l2_acc = results_l2.iloc[0]['Accuracy']
    
    print("\n" + "="*60)
    print("PROJECT PIPELINE COMPLETE")
    print("="*60 + "\n")
    print("Phase 1 (Regression) - Best Model: Ridge + Polynomial (alpha=10, deg=2)")
    print(f"  R² = {r2_rp10.mean():.4f}")
    print("\nPhase 2 (Classification) - Best Model: Random Forest (n=200)")
    print(f"  Accuracy = {acc_rf200:.4f}")
    print(f"  F1-Weighted = {f1_rf200:.4f}")
    print(f"\nPhase 3 (Unsupervised Learning):")
    print(f"  K-Means: Optimal K = {optimal_k}, ARI = {comparison_metrics.get('ARI', 0):.4f}")
    print(f"  PCA: 5 components explain 80% variance")
    print(f"\nPhase 4 (Deep Learning):")
    print(f"  4.1 MLP Architectures:")
    print(f"    - MLP Simple: Accuracy = {acc_mlp_simple:.4f}")
    print(f"    - MLP Improved: Accuracy = {acc_mlp_improved:.4f}")
    print(f"  4.2 Training Techniques:")
    print(f"    - Best Optimizer: {best_optimizer} (Acc = {best_opt_acc:.4f})")
    print(f"    - Best Learning Rate: {best_lr} (Acc = {best_lr_acc:.4f})")
    print(f"    - Best L2 Regularization: {best_l2} (Acc = {best_l2_acc:.4f})")
    print(f"\nBest Overall: Random Forest (n=200) - Accuracy = {acc_rf200:.4f}")
    print("\n")
    
    return model_simple, model_kfold, model_poly2, model_poly3, model_ridge_10, model_rp_10, model_knn_k3, model_nb, model_dt10, model_rf200, kmeans_model, pca_model, model_mlp_simple, model_mlp_improved


if __name__ == "__main__":
    # Run main pipeline
    main()
