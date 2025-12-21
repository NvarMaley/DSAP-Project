
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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


if __name__ == "__main__":
    run_linear_regression()
