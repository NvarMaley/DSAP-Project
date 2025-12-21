import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def run_linear_regression():
    """
    Simple Linear Regression with results saved to CSV
    """
    print('=== LINEAR REGRESSION ===\n')
    
    # Load data
    df = pd.read_csv('../data/processed/merged_dataset.csv')
    
    # Prepare X and y
    X = df.drop(['Country', 'Year', 'Credit_Rating'], axis=1)
    y = df['Credit_Rating']
    
    # Split train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Print results
    print('Training Set:')
    print(f'  RMSE: {train_rmse:.4f}')
    print(f'  MAE:  {train_mae:.4f}')
    print(f'  R²:   {train_r2:.4f}\n')
    
    print('Test Set:')
    print(f'  RMSE: {test_rmse:.4f}')
    print(f'  MAE:  {test_mae:.4f}')
    print(f'  R²:   {test_r2:.4f}\n')
    
    # Save metrics to CSV
    os.makedirs('../results', exist_ok=True)
    
    metrics_df = pd.DataFrame({
        'Model': ['Linear Regression'],
        'Train_RMSE': [train_rmse],
        'Train_MAE': [train_mae],
        'Train_R2': [train_r2],
        'Test_RMSE': [test_rmse],
        'Test_MAE': [test_mae],
        'Test_R2': [test_r2]
    })
    metrics_df.to_csv('../results/regression_metrics.csv', index=False)
    print('✓ Metrics saved to: results/regression_metrics.csv\n')
    
    # Save coefficients to CSV with Model column
    coefficients_df = pd.DataFrame({
        'Model': ['Linear Regression'] * len(X.columns),
        'Feature': X.columns,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    coefficients_df.to_csv('../results/coefficients.csv', index=False)
    print('✓ Coefficients saved to: results/coefficients.csv\n')
    
    # Create visualization
    create_visualization(y_test, y_test_pred, coefficients_df)
    
    print('=== COMPLETE ===\n')
    
    return model


def create_visualization(y_true, y_pred, coefficients_df):
    """
    Create simple visualization with 2 plots
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Actual vs Predicted
    ax1.scatter(y_true, y_pred, alpha=0.6, edgecolors='k')
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax1.set_xlabel('Actual Credit Rating', fontsize=12)
    ax1.set_ylabel('Predicted Credit Rating', fontsize=12)
    ax1.set_title('Actual vs Predicted', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Feature Importance (Coefficients)
    colors = ['green' if x < 0 else 'red' for x in coefficients_df['Coefficient']]
    ax2.barh(coefficients_df['Feature'], coefficients_df['Coefficient'], color=colors, alpha=0.7)
    ax2.set_xlabel('Coefficient Value', fontsize=12)
    ax2.set_title('Feature Importance (Coefficients)', fontsize=14, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    # Save
    os.makedirs('../results', exist_ok=True)
    plt.savefig('../results/regression_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print('✓ Visualization saved to: results/regression_visualization.png\n')



def create_kfold_visualization(rmse_scores, mae_scores, r2_scores, k=5):
    """
    Create visualization for K-fold CV results
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: R² scores per fold
    folds = [f'Fold {i+1}' for i in range(k)]
    ax1.bar(folds, r2_scores, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axhline(y=r2_scores.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {r2_scores.mean():.4f}')
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_title('R² Score per Fold (K-Fold CV)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Box plot of all metrics
    metrics_data = [rmse_scores, mae_scores, r2_scores]
    ax2.boxplot(metrics_data, labels=['RMSE', 'MAE', 'R²'])
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Distribution of Metrics (K-Fold CV)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    os.makedirs('../results', exist_ok=True)
    plt.savefig('../results/kfold_cv_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print('✓ K-fold visualization saved to: results/kfold_cv_visualization.png\n')


def run_linear_regression_kfold(k=5):
    """
    Linear Regression with K-fold Cross-Validation
    """
    print(f'=== LINEAR REGRESSION (K-FOLD CV, K={k}) ===\n')
    
    # Load data
    df = pd.read_csv('../data/processed/merged_dataset.csv')
    
    # Prepare X and y
    X = df.drop(['Country', 'Year', 'Credit_Rating'], axis=1)
    y = df['Credit_Rating']
    
    # Create model
    model = LinearRegression()
    
    # K-fold cross-validation
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    
    # Calculate scores for each metric
    print(f'Running {k}-fold cross-validation...\n')
    
    rmse_scores = -cross_val_score(model, X, y, cv=kfold, scoring='neg_root_mean_squared_error')
    mae_scores = -cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_absolute_error')
    r2_scores = cross_val_score(model, X, y, cv=kfold, scoring='r2')
    
    # Print results for each fold
    print('Results per fold:')
    for i in range(k):
        print(f'  Fold {i+1}: RMSE={rmse_scores[i]:.4f}, MAE={mae_scores[i]:.4f}, R²={r2_scores[i]:.4f}')
    
    # Calculate mean and std
    print(f'\nAverage across {k} folds:')
    print(f'  RMSE: {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}')
    print(f'  MAE:  {mae_scores.mean():.4f} ± {mae_scores.std():.4f}')
    print(f'  R²:   {r2_scores.mean():.4f} ± {r2_scores.std():.4f}\n')
    
    # Train final model on all data for coefficients
    model.fit(X, y)
    
    # Save metrics to CSV (append mode)
    os.makedirs('../results', exist_ok=True)
    
    metrics_df = pd.DataFrame({
        'Model': [f'Linear Regression (K-Fold CV, K={k})'],
        'Mean_RMSE': [rmse_scores.mean()],
        'Mean_MAE': [mae_scores.mean()],
        'Mean_R2': [r2_scores.mean()],
        'Std_RMSE': [rmse_scores.std()],
        'Std_MAE': [mae_scores.std()],
        'Std_R2': [r2_scores.std()]
    })
    
    # Append to existing file
    if os.path.exists('../results/regression_metrics.csv'):
        existing_df = pd.read_csv('../results/regression_metrics.csv')
        metrics_df = pd.concat([existing_df, metrics_df], ignore_index=True)
    
    metrics_df.to_csv('../results/regression_metrics.csv', index=False)
    print('✓ Metrics saved to: results/regression_metrics.csv\n')
    
    # Save coefficients to CSV (append mode)
    coefficients_df = pd.DataFrame({
        'Model': [f'Linear Regression (K-Fold CV, K={k})'] * len(X.columns),
        'Feature': X.columns,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    # Append to existing file
    if os.path.exists('../results/coefficients.csv'):
        existing_coef = pd.read_csv('../results/coefficients.csv')
        coefficients_df = pd.concat([existing_coef, coefficients_df], ignore_index=True)
    
    coefficients_df.to_csv('../results/coefficients.csv', index=False)
    print('✓ Coefficients saved to: results/coefficients.csv\n')
    
    # Create K-fold visualization
    create_kfold_visualization(rmse_scores, mae_scores, r2_scores, k)
    
    print('=== COMPLETE ===\n')
    
    return model, rmse_scores, mae_scores, r2_scores


def run_polynomial_regression(degree=2, k=5):
    """
    Polynomial Regression with K-fold Cross-Validation
    """
    print(f'=== POLYNOMIAL REGRESSION (DEGREE={degree}, K-FOLD CV, K={k}) ===\n')
    
    # Load data
    df = pd.read_csv('../data/processed/merged_dataset.csv')
    
    # Prepare X and y
    X = df.drop(['Country', 'Year', 'Credit_Rating'], axis=1)
    y = df['Credit_Rating']
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    print(f'Original features: {X.shape[1]}')
    print(f'Polynomial features (degree={degree}): {X_poly.shape[1]}\n')
    
    # Create model
    model = LinearRegression()
    
    # K-fold cross-validation
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    
    # Calculate scores for each metric
    print(f'Running {k}-fold cross-validation...\n')
    
    rmse_scores = -cross_val_score(model, X_poly, y, cv=kfold, scoring='neg_root_mean_squared_error')
    mae_scores = -cross_val_score(model, X_poly, y, cv=kfold, scoring='neg_mean_absolute_error')
    r2_scores = cross_val_score(model, X_poly, y, cv=kfold, scoring='r2')
    
    # Print results for each fold
    print('Results per fold:')
    for i in range(k):
        print(f'  Fold {i+1}: RMSE={rmse_scores[i]:.4f}, MAE={mae_scores[i]:.4f}, R²={r2_scores[i]:.4f}')
    
    # Calculate mean and std
    print(f'\nAverage across {k} folds:')
    print(f'  RMSE: {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}')
    print(f'  MAE:  {mae_scores.mean():.4f} ± {mae_scores.std():.4f}')
    print(f'  R²:   {r2_scores.mean():.4f} ± {r2_scores.std():.4f}\n')
    
    # Train final model on all data for coefficients
    model.fit(X_poly, y)
    
    # Save metrics to CSV (append mode)
    os.makedirs('../results', exist_ok=True)
    
    metrics_df = pd.DataFrame({
        'Model': [f'Polynomial Regression (deg={degree}, K-Fold CV, K={k})'],
        'Mean_RMSE': [rmse_scores.mean()],
        'Mean_MAE': [mae_scores.mean()],
        'Mean_R2': [r2_scores.mean()],
        'Std_RMSE': [rmse_scores.std()],
        'Std_MAE': [mae_scores.std()],
        'Std_R2': [r2_scores.std()]
    })
    
    # Append to existing file
    if os.path.exists('../results/regression_metrics.csv'):
        existing_df = pd.read_csv('../results/regression_metrics.csv')
        metrics_df = pd.concat([existing_df, metrics_df], ignore_index=True)
    
    metrics_df.to_csv('../results/regression_metrics.csv', index=False)
    print('✓ Metrics saved to: results/regression_metrics.csv\n')
    
    # Save top 10 most important coefficients only (too many for polynomial)
    feature_names = poly.get_feature_names_out(X.columns)
    coef_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False).head(10)
    
    coefficients_df = pd.DataFrame({
        'Model': [f'Polynomial Regression (deg={degree})'] * 10,
        'Feature': coef_importance['Feature'].values,
        'Coefficient': coef_importance['Coefficient'].values
    })
    
    # Append to existing file
    if os.path.exists('../results/coefficients.csv'):
        existing_coef = pd.read_csv('../results/coefficients.csv')
        coefficients_df = pd.concat([existing_coef, coefficients_df], ignore_index=True)
    
    coefficients_df.to_csv('../results/coefficients.csv', index=False)
    print('✓ Top 10 coefficients saved to: results/coefficients.csv\n')
    
    # Create visualization
    create_polynomial_visualization(r2_scores, degree, k)
    
    print('=== COMPLETE ===\n')
    
    return model, poly, rmse_scores, mae_scores, r2_scores


def create_polynomial_visualization(r2_scores, degree, k=5):
    """
    Create visualization for Polynomial Regression results
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Bar plot: R² scores per fold
    folds = [f'Fold {i+1}' for i in range(k)]
    ax.bar(folds, r2_scores, alpha=0.7, color='coral', edgecolor='black')
    ax.axhline(y=r2_scores.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {r2_scores.mean():.4f}')
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title(f'Polynomial Regression (Degree={degree})\nR² Score per Fold (K-Fold CV)', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    os.makedirs('../results', exist_ok=True)
    plt.savefig(f'../results/polynomial_deg{degree}_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'✓ Visualization saved to: results/polynomial_deg{degree}_visualization.png\n')


def run_ridge_regression(alpha=1.0, k=5):
    """
    Ridge Regression on Linear features with K-fold Cross-Validation
    """
    print(f'=== RIDGE REGRESSION (ALPHA={alpha}, K-FOLD CV, K={k}) ===\n')
    
    # Load data
    df = pd.read_csv('../data/processed/merged_dataset.csv')
    
    # Prepare X and y
    X = df.drop(['Country', 'Year', 'Credit_Rating'], axis=1)
    y = df['Credit_Rating']
    
    # Normalize features (REQUIRED for Ridge!)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f'Features: {X.shape[1]} (normalized)')
    print(f'Regularization strength (alpha): {alpha}\n')
    
    # Create Ridge model
    model = Ridge(alpha=alpha)
    
    # K-fold cross-validation
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    
    # Calculate scores for each metric
    print(f'Running {k}-fold cross-validation...\n')
    
    rmse_scores = -cross_val_score(model, X_scaled, y, cv=kfold, scoring='neg_root_mean_squared_error')
    mae_scores = -cross_val_score(model, X_scaled, y, cv=kfold, scoring='neg_mean_absolute_error')
    r2_scores = cross_val_score(model, X_scaled, y, cv=kfold, scoring='r2')
    
    # Print results for each fold
    print('Results per fold:')
    for i in range(k):
        print(f'  Fold {i+1}: RMSE={rmse_scores[i]:.4f}, MAE={mae_scores[i]:.4f}, R²={r2_scores[i]:.4f}')
    
    # Calculate mean and std
    print(f'\nAverage across {k} folds:')
    print(f'  RMSE: {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}')
    print(f'  MAE:  {mae_scores.mean():.4f} ± {mae_scores.std():.4f}')
    print(f'  R²:   {r2_scores.mean():.4f} ± {r2_scores.std():.4f}\n')
    
    # Train final model on all data for coefficients
    model.fit(X_scaled, y)
    
    # Save metrics to CSV (append mode)
    os.makedirs('../results', exist_ok=True)
    
    metrics_df = pd.DataFrame({
        'Model': [f'Ridge Regression (alpha={alpha}, K-Fold CV, K={k})'],
        'Mean_RMSE': [rmse_scores.mean()],
        'Mean_MAE': [mae_scores.mean()],
        'Mean_R2': [r2_scores.mean()],
        'Std_RMSE': [rmse_scores.std()],
        'Std_MAE': [mae_scores.std()],
        'Std_R2': [r2_scores.std()]
    })
    
    # Append to existing file
    if os.path.exists('../results/regression_metrics.csv'):
        existing_df = pd.read_csv('../results/regression_metrics.csv')
        metrics_df = pd.concat([existing_df, metrics_df], ignore_index=True)
    
    metrics_df.to_csv('../results/regression_metrics.csv', index=False)
    print('✓ Metrics saved to: results/regression_metrics.csv\n')
    
    # Save top 10 coefficients
    coefficients_df = pd.DataFrame({
        'Model': [f'Ridge Regression (alpha={alpha})'] * len(X.columns),
        'Feature': X.columns,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False).head(10)
    
    # Append to existing file
    if os.path.exists('../results/coefficients.csv'):
        existing_coef = pd.read_csv('../results/coefficients.csv')
        coefficients_df = pd.concat([existing_coef, coefficients_df], ignore_index=True)
    
    coefficients_df.to_csv('../results/coefficients.csv', index=False)
    print('✓ Coefficients saved to: results/coefficients.csv\n')
    
    print('=== COMPLETE ===\n')
    
    return model, scaler, rmse_scores, mae_scores, r2_scores


def run_ridge_polynomial_regression(alpha=1.0, degree=2, k=5):
    """
    Ridge Regression on Polynomial features with K-fold Cross-Validation
    """
    print(f'=== RIDGE + POLYNOMIAL (DEGREE={degree}, ALPHA={alpha}, K-FOLD CV, K={k}) ===\n')
    
    # Load data
    df = pd.read_csv('../data/processed/merged_dataset.csv')
    
    # Prepare X and y
    X = df.drop(['Country', 'Year', 'Credit_Rating'], axis=1)
    y = df['Credit_Rating']
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    # Normalize polynomial features (REQUIRED for Ridge!)
    scaler = StandardScaler()
    X_poly_scaled = scaler.fit_transform(X_poly)
    
    print(f'Original features: {X.shape[1]}')
    print(f'Polynomial features (degree={degree}): {X_poly.shape[1]}')
    print(f'Regularization strength (alpha): {alpha}\n')
    
    # Create Ridge model
    model = Ridge(alpha=alpha)
    
    # K-fold cross-validation
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    
    # Calculate scores for each metric
    print(f'Running {k}-fold cross-validation...\n')
    
    rmse_scores = -cross_val_score(model, X_poly_scaled, y, cv=kfold, scoring='neg_root_mean_squared_error')
    mae_scores = -cross_val_score(model, X_poly_scaled, y, cv=kfold, scoring='neg_mean_absolute_error')
    r2_scores = cross_val_score(model, X_poly_scaled, y, cv=kfold, scoring='r2')
    
    # Print results for each fold
    print('Results per fold:')
    for i in range(k):
        print(f'  Fold {i+1}: RMSE={rmse_scores[i]:.4f}, MAE={mae_scores[i]:.4f}, R²={r2_scores[i]:.4f}')
    
    # Calculate mean and std
    print(f'\nAverage across {k} folds:')
    print(f'  RMSE: {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}')
    print(f'  MAE:  {mae_scores.mean():.4f} ± {mae_scores.std():.4f}')
    print(f'  R²:   {r2_scores.mean():.4f} ± {r2_scores.std():.4f}\n')
    
    # Train final model on all data for coefficients
    model.fit(X_poly_scaled, y)
    
    # Save metrics to CSV (append mode)
    os.makedirs('../results', exist_ok=True)
    
    metrics_df = pd.DataFrame({
        'Model': [f'Ridge + Polynomial (deg={degree}, alpha={alpha}, K-Fold CV, K={k})'],
        'Mean_RMSE': [rmse_scores.mean()],
        'Mean_MAE': [mae_scores.mean()],
        'Mean_R2': [r2_scores.mean()],
        'Std_RMSE': [rmse_scores.std()],
        'Std_MAE': [mae_scores.std()],
        'Std_R2': [r2_scores.std()]
    })
    
    # Append to existing file
    if os.path.exists('../results/regression_metrics.csv'):
        existing_df = pd.read_csv('../results/regression_metrics.csv')
        metrics_df = pd.concat([existing_df, metrics_df], ignore_index=True)
    
    metrics_df.to_csv('../results/regression_metrics.csv', index=False)
    print('✓ Metrics saved to: results/regression_metrics.csv\n')
    
    # Save top 10 coefficients
    feature_names = poly.get_feature_names_out(X.columns)
    coef_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False).head(10)
    
    coefficients_df = pd.DataFrame({
        'Model': [f'Ridge + Polynomial (deg={degree}, alpha={alpha})'] * 10,
        'Feature': coef_importance['Feature'].values,
        'Coefficient': coef_importance['Coefficient'].values
    })
    
    # Append to existing file
    if os.path.exists('../results/coefficients.csv'):
        existing_coef = pd.read_csv('../results/coefficients.csv')
        coefficients_df = pd.concat([existing_coef, coefficients_df], ignore_index=True)
    
    coefficients_df.to_csv('../results/coefficients.csv', index=False)
    print('✓ Top 10 coefficients saved to: results/coefficients.csv\n')
    
    print('=== COMPLETE ===\n')
    
    return model, poly, scaler, rmse_scores, mae_scores, r2_scores


if __name__ == "__main__":
    # Run both versions for comparison
    print('\n' + '='*60)
    print('VERSION 1: SIMPLE TRAIN/TEST SPLIT (80/20)')
    print('='*60 + '\n')
    model_simple = run_linear_regression()
    
    print('\n' + '='*60)
    print('VERSION 2: K-FOLD CROSS-VALIDATION (K=5)')
    print('='*60 + '\n')
    model_kfold, rmse_scores, mae_scores, r2_scores = run_linear_regression_kfold(k=5)
    
    print('\n' + '='*60)
    print('COMPARISON COMPLETE - Check results/regression_metrics.csv')
    print('='*60 + '\n')
