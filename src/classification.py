import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score


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
    df = pd.read_csv('data/processed/merged_dataset_labels.csv')
    
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
    metrics_file = 'results/classification_metrics.csv'
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
    df = pd.read_csv('data/processed/merged_dataset_labels.csv')
    
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
    metrics_file = 'results/classification_metrics.csv'
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
