import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

# Get the project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
RESULTS_DIR = PROJECT_ROOT / 'results'


def run_knn_classification(k=5, cv=5):
    """
    k-Nearest Neighbors Classification with K-Fold Cross-Validation
    
    Parameters:
    - k: Number of neighbors (default 5)
    - cv: Number of folds for cross-validation (default 5)
    
    Returns:
    - model: Trained k-NN model
    - scaler: StandardScaler used for normalization
    - accuracy_mean: Mean accuracy across folds
    - accuracy_std: Standard deviation of accuracy
    - f1_weighted_mean: Mean weighted F1-score
    - f1_weighted_std: Standard deviation of weighted F1-score
    """
    
    print(f'=== k-NN CLASSIFICATION (k={k}) ===\n')
    
    # Load data
    df = pd.read_csv(DATA_PROCESSED / 'merged_dataset_labels.csv')
    
    # Prepare X and y
    X = df.drop(['Country', 'Year', 'Credit_Rating_Label'], axis=1)
    y = df['Credit_Rating_Label']
    
    print(f'Dataset shape: {X.shape}')
    print(f'Number of classes: {y.nunique()}')
    print(f'Class distribution:\n{y.value_counts().sort_index()}\n')
    
    # Normalize features (MANDATORY for k-NN)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print('✓ Features normalized with StandardScaler\n')
    
    # Create k-NN model
    model = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    
    # Stratified K-Fold to maintain class distribution
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Cross-validation scores
    print(f'Running {cv}-fold cross-validation...\n')
    
    # Accuracy
    accuracy_scores = cross_val_score(model, X_scaled, y, cv=skf, scoring='accuracy')
    accuracy_mean = accuracy_scores.mean()
    accuracy_std = accuracy_scores.std()
    
    # Weighted F1-Score (important for imbalanced classes)
    f1_weighted_scores = cross_val_score(model, X_scaled, y, cv=skf, scoring='f1_weighted')
    f1_weighted_mean = f1_weighted_scores.mean()
    f1_weighted_std = f1_weighted_scores.std()
    
    # Macro F1-Score (all classes treated equally)
    f1_macro_scores = cross_val_score(model, X_scaled, y, cv=skf, scoring='f1_macro')
    f1_macro_mean = f1_macro_scores.mean()
    f1_macro_std = f1_macro_scores.std()
    
    # Print results
    print('Cross-Validation Results:')
    print(f'  Accuracy:      {accuracy_mean:.4f} ± {accuracy_std:.4f}')
    print(f'  F1-Weighted:   {f1_weighted_mean:.4f} ± {f1_weighted_std:.4f}')
    print(f'  F1-Macro:      {f1_macro_mean:.4f} ± {f1_macro_std:.4f}\n')
    
    # Train final model on full dataset
    model.fit(X_scaled, y)
    print('✓ Final model trained on full dataset\n')
    
    # Save metrics to CSV
    os.makedirs('results', exist_ok=True)
    
    metrics_df = pd.DataFrame({
        'Model': [f'k-NN (k={k})'],
        'Mean_Accuracy': [accuracy_mean],
        'Std_Accuracy': [accuracy_std],
        'Mean_F1_Weighted': [f1_weighted_mean],
        'Std_F1_Weighted': [f1_weighted_std],
        'Mean_F1_Macro': [f1_macro_mean],
        'Std_F1_Macro': [f1_macro_std]
    })
    
    # Append to existing file or create new one
    metrics_file = RESULTS_DIR / 'classification_metrics.csv'
    if os.path.exists(metrics_file):
        existing_df = pd.read_csv(metrics_file)
        metrics_df = pd.concat([existing_df, metrics_df], ignore_index=True)
    
    metrics_df.to_csv(metrics_file, index=False)
    print(f'✓ Metrics saved to: results/classification_metrics.csv\n')
    
    # Create confusion matrix
    create_confusion_matrix(model, X_scaled, y, skf, k)
    
    print(f'=== k-NN (k={k}) COMPLETE ===\n')
    
    return model, scaler, accuracy_mean, accuracy_std, f1_weighted_mean, f1_weighted_std


def create_confusion_matrix(model, X_scaled, y, skf, k):
    """
    Create and save confusion matrix visualization
    
    Parameters:
    - model: Trained k-NN model
    - X_scaled: Normalized features
    - y: True labels
    - skf: StratifiedKFold object
    - k: Number of neighbors
    """
    
    print('Creating confusion matrix...')
    
    # Get predictions using cross_val_predict
    y_pred = cross_val_predict(model, X_scaled, y, cv=skf)
    
    # Compute confusion matrix
    cm = confusion_matrix(y, y_pred)
    
    # Get unique labels in sorted order
    labels = sorted(y.unique())
    
    # Create figure
    plt.figure(figsize=(14, 12))
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    
    plt.title(f'k-NN (k={k}) Confusion Matrix\nAccuracy: {accuracy_score(y, y_pred):.4f}', 
              fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save figure
    os.makedirs('results/confusion_matrices', exist_ok=True)
    output_path = f'results/confusion_matrices/knn_k{k}_confusion_matrix.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'✓ Confusion matrix saved to: results/confusion_matrices/knn_k{k}_confusion_matrix.png\n')
    
    # Print classification report
    print('Classification Report (per-class metrics):')
    print(classification_report(y, y_pred, zero_division=0))


def run_naive_bayes_classification(cv=5):
    """
    Gaussian Naive Bayes Classification with K-Fold Cross-Validation
    
    Parameters:
    - cv: Number of folds for cross-validation (default 5)
    
    Returns:
    - model: Trained Naive Bayes model
    - accuracy_mean: Mean accuracy across folds
    - accuracy_std: Standard deviation of accuracy
    - f1_weighted_mean: Mean weighted F1-score
    - f1_weighted_std: Standard deviation of weighted F1-score
    """
    
    print('=== NAIVE BAYES CLASSIFICATION ===\n')
    
    # Load data
    df = pd.read_csv(DATA_PROCESSED / 'merged_dataset_labels.csv')
    
    # Prepare X and y
    X = df.drop(['Country', 'Year', 'Credit_Rating_Label'], axis=1)
    y = df['Credit_Rating_Label']
    
    print(f'Dataset shape: {X.shape}')
    print(f'Number of classes: {y.nunique()}')
    print(f'Class distribution:\n{y.value_counts().sort_index()}\n')
    
    # Create Gaussian Naive Bayes model (no normalization needed)
    model = GaussianNB()
    
    # Stratified K-Fold to maintain class distribution
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Cross-validation scores
    print(f'Running {cv}-fold cross-validation...\n')
    
    # Accuracy
    accuracy_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    accuracy_mean = accuracy_scores.mean()
    accuracy_std = accuracy_scores.std()
    
    # Weighted F1-Score (important for imbalanced classes)
    f1_weighted_scores = cross_val_score(model, X, y, cv=skf, scoring='f1_weighted')
    f1_weighted_mean = f1_weighted_scores.mean()
    f1_weighted_std = f1_weighted_scores.std()
    
    # Macro F1-Score (all classes treated equally)
    f1_macro_scores = cross_val_score(model, X, y, cv=skf, scoring='f1_macro')
    f1_macro_mean = f1_macro_scores.mean()
    f1_macro_std = f1_macro_scores.std()
    
    # Print results
    print('Cross-Validation Results:')
    print(f'  Accuracy:      {accuracy_mean:.4f} ± {accuracy_std:.4f}')
    print(f'  F1-Weighted:   {f1_weighted_mean:.4f} ± {f1_weighted_std:.4f}')
    print(f'  F1-Macro:      {f1_macro_mean:.4f} ± {f1_macro_std:.4f}\n')
    
    # Train final model on full dataset
    model.fit(X, y)
    print('✓ Final model trained on full dataset\n')
    
    # Save metrics to CSV
    os.makedirs('results', exist_ok=True)
    
    metrics_df = pd.DataFrame({
        'Model': ['Naive Bayes'],
        'Mean_Accuracy': [accuracy_mean],
        'Std_Accuracy': [accuracy_std],
        'Mean_F1_Weighted': [f1_weighted_mean],
        'Std_F1_Weighted': [f1_weighted_std],
        'Mean_F1_Macro': [f1_macro_mean],
        'Std_F1_Macro': [f1_macro_std]
    })
    
    # Append to existing file or create new one
    metrics_file = RESULTS_DIR / 'classification_metrics.csv'
    if os.path.exists(metrics_file):
        existing_df = pd.read_csv(metrics_file)
        metrics_df = pd.concat([existing_df, metrics_df], ignore_index=True)
    
    metrics_df.to_csv(metrics_file, index=False)
    print(f'✓ Metrics saved to: results/classification_metrics.csv\n')
    
    # Create confusion matrix
    create_confusion_matrix_nb(model, X, y, skf)
    
    print('=== NAIVE BAYES COMPLETE ===\n')
    
    return model, accuracy_mean, accuracy_std, f1_weighted_mean, f1_weighted_std


def create_confusion_matrix_nb(model, X, y, skf):
    """
    Create and save confusion matrix visualization for Naive Bayes
    
    Parameters:
    - model: Trained Naive Bayes model
    - X: Features
    - y: True labels
    - skf: StratifiedKFold object
    """
    
    print('Creating confusion matrix...')
    
    # Get predictions using cross_val_predict
    y_pred = cross_val_predict(model, X, y, cv=skf)
    
    # Compute confusion matrix
    cm = confusion_matrix(y, y_pred)
    
    # Get unique labels in sorted order
    labels = sorted(y.unique())
    
    # Create figure
    plt.figure(figsize=(14, 12))
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    
    plt.title(f'Naive Bayes Confusion Matrix\nAccuracy: {accuracy_score(y, y_pred):.4f}', 
              fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save figure
    os.makedirs('results/confusion_matrices', exist_ok=True)
    output_path = 'results/confusion_matrices/naive_bayes_confusion_matrix.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'✓ Confusion matrix saved to: results/confusion_matrices/naive_bayes_confusion_matrix.png\n')
    
    # Print classification report
    print('Classification Report (per-class metrics):')
    print(classification_report(y, y_pred, zero_division=0))


def run_decision_tree_classification(max_depth=5, cv=5):
    """
    Decision Tree Classification with K-Fold Cross-Validation
    
    Parameters:
    - max_depth: Maximum depth of the tree (default 5)
    - cv: Number of folds for cross-validation (default 5)
    
    Returns:
    - model: Trained Decision Tree model
    - accuracy_mean: Mean accuracy across folds
    - accuracy_std: Standard deviation of accuracy
    - f1_weighted_mean: Mean weighted F1-score
    - f1_weighted_std: Standard deviation of weighted F1-score
    """
    
    print(f'=== DECISION TREE CLASSIFICATION (max_depth={max_depth}) ===\n')
    
    # Load data
    df = pd.read_csv(DATA_PROCESSED / 'merged_dataset_labels.csv')
    
    # Prepare X and y
    X = df.drop(['Country', 'Year', 'Credit_Rating_Label'], axis=1)
    y = df['Credit_Rating_Label']
    
    print(f'Dataset shape: {X.shape}')
    print(f'Number of classes: {y.nunique()}')
    print(f'Class distribution:\n{y.value_counts().sort_index()}\n')
    
    # Create Decision Tree model (no normalization needed)
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=42,
        class_weight='balanced'
    )
    
    # Stratified K-Fold to maintain class distribution
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Cross-validation scores
    print(f'Running {cv}-fold cross-validation...\n')
    
    # Accuracy
    accuracy_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    accuracy_mean = accuracy_scores.mean()
    accuracy_std = accuracy_scores.std()
    
    # Weighted F1-Score
    f1_weighted_scores = cross_val_score(model, X, y, cv=skf, scoring='f1_weighted')
    f1_weighted_mean = f1_weighted_scores.mean()
    f1_weighted_std = f1_weighted_scores.std()
    
    # Macro F1-Score
    f1_macro_scores = cross_val_score(model, X, y, cv=skf, scoring='f1_macro')
    f1_macro_mean = f1_macro_scores.mean()
    f1_macro_std = f1_macro_scores.std()
    
    # Print results
    print('Cross-Validation Results:')
    print(f'  Accuracy:      {accuracy_mean:.4f} ± {accuracy_std:.4f}')
    print(f'  F1-Weighted:   {f1_weighted_mean:.4f} ± {f1_weighted_std:.4f}')
    print(f'  F1-Macro:      {f1_macro_mean:.4f} ± {f1_macro_std:.4f}\n')
    
    # Train final model on full dataset
    model.fit(X, y)
    print('✓ Final model trained on full dataset\n')
    
    # Save metrics to CSV
    os.makedirs('results', exist_ok=True)
    
    metrics_df = pd.DataFrame({
        'Model': [f'Decision Tree (depth={max_depth})'],
        'Mean_Accuracy': [accuracy_mean],
        'Std_Accuracy': [accuracy_std],
        'Mean_F1_Weighted': [f1_weighted_mean],
        'Std_F1_Weighted': [f1_weighted_std],
        'Mean_F1_Macro': [f1_macro_mean],
        'Std_F1_Macro': [f1_macro_std]
    })
    
    # Append to existing file
    metrics_file = RESULTS_DIR / 'classification_metrics.csv'
    if os.path.exists(metrics_file):
        existing_df = pd.read_csv(metrics_file)
        metrics_df = pd.concat([existing_df, metrics_df], ignore_index=True)
    
    metrics_df.to_csv(metrics_file, index=False)
    print(f'✓ Metrics saved to: results/classification_metrics.csv\n')
    
    # Create confusion matrix
    create_confusion_matrix_dt(model, X, y, skf, max_depth)
    
    # Create feature importance visualization
    create_feature_importance(model, X, max_depth)
    
    print(f'=== DECISION TREE (depth={max_depth}) COMPLETE ===\n')
    
    return model, accuracy_mean, accuracy_std, f1_weighted_mean, f1_weighted_std


def create_confusion_matrix_dt(model, X, y, skf, max_depth):
    """
    Create and save confusion matrix visualization for Decision Tree
    """
    
    print('Creating confusion matrix...')
    
    y_pred = cross_val_predict(model, X, y, cv=skf)
    cm = confusion_matrix(y, y_pred)
    labels = sorted(y.unique())
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    
    plt.title(f'Decision Tree (depth={max_depth}) Confusion Matrix\nAccuracy: {accuracy_score(y, y_pred):.4f}', 
              fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    os.makedirs('results/confusion_matrices', exist_ok=True)
    output_path = f'results/confusion_matrices/decision_tree_depth{max_depth}_confusion_matrix.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'✓ Confusion matrix saved to: results/confusion_matrices/decision_tree_depth{max_depth}_confusion_matrix.png\n')
    
    print('Classification Report (per-class metrics):')
    print(classification_report(y, y_pred, zero_division=0))


def create_feature_importance(model, X, max_depth):
    """
    Create and save feature importance visualization for Decision Tree
    """
    
    print('Creating feature importance visualization...')
    
    feature_importance = model.feature_importances_
    feature_names = X.columns
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=True)
    
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue')
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Decision Tree (depth={max_depth}) - Feature Importance', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    os.makedirs('results/feature_importance', exist_ok=True)
    output_path = f'results/feature_importance/decision_tree_depth{max_depth}_feature_importance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'✓ Feature importance saved to: results/feature_importance/decision_tree_depth{max_depth}_feature_importance.png\n')
    
    print('Feature Importance Ranking:')
    print('='*50)
    for idx, row in importance_df.sort_values('Importance', ascending=False).iterrows():
        print(f"  {row['Feature']:20s}: {row['Importance']:.4f}")


def run_random_forest_classification(n_estimators=100, cv=5):
    """
    Random Forest Classification with K-Fold Cross-Validation
    
    Parameters:
    - n_estimators: Number of trees in the forest (default 100)
    - cv: Number of folds for cross-validation (default 5)
    
    Returns:
    - model: Trained Random Forest model
    - accuracy_mean: Mean accuracy across folds
    - accuracy_std: Standard deviation of accuracy
    - f1_weighted_mean: Mean weighted F1-score
    - f1_weighted_std: Standard deviation of weighted F1-score
    """
    
    print(f'=== RANDOM FOREST CLASSIFICATION (n_estimators={n_estimators}) ===\n')
    
    # Load data
    df = pd.read_csv(DATA_PROCESSED / 'merged_dataset_labels.csv')
    
    # Prepare X and y
    X = df.drop(['Country', 'Year', 'Credit_Rating_Label'], axis=1)
    y = df['Credit_Rating_Label']
    
    print(f'Dataset shape: {X.shape}')
    print(f'Number of classes: {y.nunique()}')
    print(f'Class distribution:\n{y.value_counts().sort_index()}\n')
    
    # Create Random Forest model (no normalization needed)
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    
    # Stratified K-Fold to maintain class distribution
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Cross-validation scores
    print(f'Running {cv}-fold cross-validation with {n_estimators} trees...\n')
    
    # Accuracy
    accuracy_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    accuracy_mean = accuracy_scores.mean()
    accuracy_std = accuracy_scores.std()
    
    # Weighted F1-Score
    f1_weighted_scores = cross_val_score(model, X, y, cv=skf, scoring='f1_weighted')
    f1_weighted_mean = f1_weighted_scores.mean()
    f1_weighted_std = f1_weighted_scores.std()
    
    # Macro F1-Score
    f1_macro_scores = cross_val_score(model, X, y, cv=skf, scoring='f1_macro')
    f1_macro_mean = f1_macro_scores.mean()
    f1_macro_std = f1_macro_scores.std()
    
    # Print results
    print('Cross-Validation Results:')
    print(f'  Accuracy:      {accuracy_mean:.4f} ± {accuracy_std:.4f}')
    print(f'  F1-Weighted:   {f1_weighted_mean:.4f} ± {f1_weighted_std:.4f}')
    print(f'  F1-Macro:      {f1_macro_mean:.4f} ± {f1_macro_std:.4f}\n')
    
    # Train final model on full dataset
    model.fit(X, y)
    print('✓ Final model trained on full dataset\n')
    
    # Save metrics to CSV
    os.makedirs('results', exist_ok=True)
    
    metrics_df = pd.DataFrame({
        'Model': [f'Random Forest (n={n_estimators})'],
        'Mean_Accuracy': [accuracy_mean],
        'Std_Accuracy': [accuracy_std],
        'Mean_F1_Weighted': [f1_weighted_mean],
        'Std_F1_Weighted': [f1_weighted_std],
        'Mean_F1_Macro': [f1_macro_mean],
        'Std_F1_Macro': [f1_macro_std]
    })
    
    # Append to existing file
    metrics_file = RESULTS_DIR / 'classification_metrics.csv'
    if os.path.exists(metrics_file):
        existing_df = pd.read_csv(metrics_file)
        metrics_df = pd.concat([existing_df, metrics_df], ignore_index=True)
    
    metrics_df.to_csv(metrics_file, index=False)
    print(f'✓ Metrics saved to: results/classification_metrics.csv\n')
    
    # Create confusion matrix
    create_confusion_matrix_rf(model, X, y, skf, n_estimators)
    
    # Create feature importance visualization
    create_feature_importance_rf(model, X, n_estimators)
    
    print(f'=== RANDOM FOREST (n={n_estimators}) COMPLETE ===\n')
    
    return model, accuracy_mean, accuracy_std, f1_weighted_mean, f1_weighted_std


def create_confusion_matrix_rf(model, X, y, skf, n_estimators):
    """
    Create and save confusion matrix visualization for Random Forest
    """
    
    print('Creating confusion matrix...')
    
    y_pred = cross_val_predict(model, X, y, cv=skf)
    cm = confusion_matrix(y, y_pred)
    labels = sorted(y.unique())
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    
    plt.title(f'Random Forest (n={n_estimators}) Confusion Matrix\nAccuracy: {accuracy_score(y, y_pred):.4f}', 
              fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    os.makedirs('results/confusion_matrices', exist_ok=True)
    output_path = f'results/confusion_matrices/random_forest_n{n_estimators}_confusion_matrix.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'✓ Confusion matrix saved to: results/confusion_matrices/random_forest_n{n_estimators}_confusion_matrix.png\n')
    
    print('Classification Report (per-class metrics):')
    print(classification_report(y, y_pred, zero_division=0))


def create_feature_importance_rf(model, X, n_estimators):
    """
    Create and save feature importance visualization for Random Forest
    """
    
    print('Creating feature importance visualization...')
    
    feature_importance = model.feature_importances_
    feature_names = X.columns
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=True)
    
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='forestgreen')
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Random Forest (n={n_estimators}) - Feature Importance', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    os.makedirs('results/feature_importance', exist_ok=True)
    output_path = f'results/feature_importance/random_forest_n{n_estimators}_feature_importance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'✓ Feature importance saved to: results/feature_importance/random_forest_n{n_estimators}_feature_importance.png\n')
    
    print('Feature Importance Ranking:')
    print('='*50)
    for idx, row in importance_df.sort_values('Importance', ascending=False).iterrows():
        print(f"  {row['Feature']:20s}: {row['Importance']:.4f}")


def create_comparison_visualization():
    """
    Create comprehensive comparison visualization of all classification models
    
    Generates a 6-panel figure comparing all models by:
    - Accuracy (with std)
    - F1-Weighted (with std)
    - F1-Macro (with std)
    - Top 5 models
    - Best model by type
    - Accuracy vs Stability scatter plot
    
    Saves to: results/classification_models_comparison.png
    """
    
    print('=== CREATING CLASSIFICATION MODELS COMPARISON VISUALIZATION ===\n')
    
    # Load classification metrics
    df = pd.read_csv(RESULTS_DIR / 'classification_metrics.csv')
    print(f'Loaded {len(df)} models from classification_metrics.csv\n')
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Color mapping by model type
    colors = []
    for model in df['Model']:
        if 'k-NN' in model:
            colors.append('#1f77b4')  # Blue
        elif 'Naive Bayes' in model:
            colors.append('#ff7f0e')  # Orange
        elif 'Decision Tree' in model:
            colors.append('#2ca02c')  # Green
        elif 'Random Forest' in model:
            colors.append('#d62728')  # Red
        else:
            colors.append('gray')
    
    # 1. Accuracy Comparison (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    x_pos = np.arange(len(df))
    ax1.barh(x_pos, df['Mean_Accuracy'], xerr=df['Std_Accuracy'], 
             color=colors, alpha=0.7, capsize=3)
    ax1.set_yticks(x_pos)
    ax1.set_yticklabels(df['Model'], fontsize=9)
    ax1.set_xlabel('Accuracy', fontsize=11, fontweight='bold')
    ax1.set_title('Accuracy Comparison (with std)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.axvline(x=0.8, color='red', linestyle='--', alpha=0.5, linewidth=2, label='80% threshold')
    ax1.legend()
    
    # 2. F1-Weighted Comparison (Top Right)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.barh(x_pos, df['Mean_F1_Weighted'], xerr=df['Std_F1_Weighted'], 
             color=colors, alpha=0.7, capsize=3)
    ax2.set_yticks(x_pos)
    ax2.set_yticklabels(df['Model'], fontsize=9)
    ax2.set_xlabel('F1-Weighted', fontsize=11, fontweight='bold')
    ax2.set_title('F1-Weighted Comparison (with std)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.axvline(x=0.8, color='red', linestyle='--', alpha=0.5, linewidth=2, label='80% threshold')
    ax2.legend()
    
    # 3. F1-Macro Comparison (Middle Left)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.barh(x_pos, df['Mean_F1_Macro'], xerr=df['Std_F1_Macro'], 
             color=colors, alpha=0.7, capsize=3)
    ax3.set_yticks(x_pos)
    ax3.set_yticklabels(df['Model'], fontsize=9)
    ax3.set_xlabel('F1-Macro', fontsize=11, fontweight='bold')
    ax3.set_title('F1-Macro Comparison (with std)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Top 5 Models (Middle Right)
    ax4 = fig.add_subplot(gs[1, 1])
    top5 = df.nlargest(5, 'Mean_Accuracy')
    x_pos_top5 = np.arange(len(top5))
    colors_top5 = [colors[df.index.get_loc(idx)] for idx in top5.index]
    ax4.bar(x_pos_top5, top5['Mean_Accuracy'], yerr=top5['Std_Accuracy'],
            color=colors_top5, alpha=0.7, capsize=5, edgecolor='black', linewidth=1.5)
    ax4.set_xticks(x_pos_top5)
    ax4.set_xticklabels(top5['Model'], rotation=45, ha='right', fontsize=10)
    ax4.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax4.set_title('Top 5 Models by Accuracy', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, linewidth=2)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(top5.iterrows()):
        ax4.text(i, row['Mean_Accuracy'] + 0.02, f"{row['Mean_Accuracy']:.2%}", 
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 5. Model Type Comparison (Bottom Left)
    ax5 = fig.add_subplot(gs[2, 0])
    model_types = {
        'k-NN': df[df['Model'].str.contains('k-NN')]['Mean_Accuracy'].max(),
        'Naive Bayes': df[df['Model'].str.contains('Naive Bayes')]['Mean_Accuracy'].max(),
        'Decision Tree': df[df['Model'].str.contains('Decision Tree')]['Mean_Accuracy'].max(),
        'Random Forest': df[df['Model'].str.contains('Random Forest')]['Mean_Accuracy'].max()
    }
    type_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = ax5.bar(model_types.keys(), model_types.values(), color=type_colors, alpha=0.7, 
                   edgecolor='black', linewidth=2)
    ax5.set_ylabel('Best Accuracy', fontsize=11, fontweight='bold')
    ax5.set_title('Best Model by Type', fontsize=13, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_ylim(0, 1)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.2%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 6. Accuracy vs Stability (Bottom Right)
    ax6 = fig.add_subplot(gs[2, 1])
    scatter = ax6.scatter(df['Std_Accuracy']*100, df['Mean_Accuracy']*100, 
                          c=colors, s=200, alpha=0.7, edgecolors='black', linewidth=1.5)
    ax6.set_xlabel('Standard Deviation (%)', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Mean Accuracy (%)', fontsize=11, fontweight='bold')
    ax6.set_title('Accuracy vs Stability', fontsize=13, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # Annotate best models
    for idx, row in df.iterrows():
        if row['Mean_Accuracy'] > 0.75:
            ax6.annotate(row['Model'], 
                        (row['Std_Accuracy']*100, row['Mean_Accuracy']*100),
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    
    # Add legend for model types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', label='k-NN'),
        Patch(facecolor='#ff7f0e', label='Naive Bayes'),
        Patch(facecolor='#2ca02c', label='Decision Tree'),
        Patch(facecolor='#d62728', label='Random Forest')
    ]
    ax6.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    # Overall title
    fig.suptitle('Classification Models Comparison - Phase 2', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    os.makedirs('results', exist_ok=True)
    output_path = 'results/classification_models_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'✓ Comparison visualization saved to: {output_path}\n')
    print('=== COMPARISON VISUALIZATION COMPLETE ===\n')
