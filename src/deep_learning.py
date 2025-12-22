"""
Deep Learning models for credit rating classification
Phase 4: Multi-Layer Perceptron (MLP)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

# Get the project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
RESULTS_DIR = PROJECT_ROOT / 'results'


def run_mlp_simple():
    """
    MLP Simple Architecture (Baseline)
    - 1 hidden layer (64 neurons)
    - Dropout 0.3
    - Adam optimizer
    """
    
    print('=== MLP SIMPLE (BASELINE) ===\n')
    
    # Load data
    df = pd.read_csv(DATA_PROCESSED / 'merged_dataset_labels.csv')
    
    # Prepare features and labels
    X = df.drop(['Country', 'Year', 'Credit_Rating_Label'], axis=1)
    y = df['Credit_Rating_Label']
    
    feature_names = X.columns.tolist()
    
    print(f'Dataset: {X.shape[0]} observations, {X.shape[1]} features')
    print(f'Classes: {y.nunique()} credit ratings\n')
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    n_classes = len(le.classes_)
    
    print(f'Label encoding: {n_classes} classes')
    print(f'Classes: {le.classes_[:5]}... → {list(range(5))}...\n')
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data: 60% train, 20% validation, 20% test
    # Note: Some classes have very few samples, so we cannot use stratify
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y_encoded, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    print(f'Data split:')
    print(f'  Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)')
    print(f'  Validation: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)')
    print(f'  Test: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)\n')
    
    # Build model
    print('Building MLP Simple architecture...')
    
    tf.random.set_seed(42)
    np.random.seed(42)
    
    model = keras.Sequential([
        layers.Input(shape=(X.shape[1],)),
        layers.Dense(64, activation='relu', name='hidden_layer'),
        layers.Dropout(0.3, name='dropout'),
        layers.Dense(n_classes, activation='softmax', name='output_layer')
    ], name='MLP_Simple')
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print('\nModel architecture:')
    model.summary()
    
    # Callbacks
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train model
    print('\nTraining MLP Simple...\n')
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Evaluate on test set
    print('\n' + '='*60)
    print('EVALUATION ON TEST SET')
    print('='*60 + '\n')
    
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f'Test Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)')
    print(f'F1-Weighted:      {f1_weighted:.4f}')
    print(f'F1-Macro:         {f1_macro:.4f}')
    print(f'Precision:        {precision:.4f}')
    print(f'Recall:           {recall:.4f}\n')
    
    # Save metrics
    os.makedirs('results/deep_learning', exist_ok=True)
    
    metrics_df = pd.DataFrame({
        'Model': ['MLP_Simple'],
        'Accuracy': [accuracy],
        'F1_Weighted': [f1_weighted],
        'F1_Macro': [f1_macro],
        'Precision': [precision],
        'Recall': [recall],
        'Epochs_Trained': [len(history.history['loss'])],
        'Best_Epoch': [len(history.history['loss']) - early_stop.patience]
    })
    
    metrics_df.to_csv('results/deep_learning/mlp_simple_metrics.csv', index=False)
    print('✓ Metrics saved to: results/deep_learning/mlp_simple_metrics.csv')
    
    # Save model
    model.save('results/deep_learning/mlp_simple_model.h5')
    print('✓ Model saved to: results/deep_learning/mlp_simple_model.h5')
    
    # Create learning curves
    create_learning_curves(history, 'MLP Simple', 'results/deep_learning/mlp_simple_learning_curves.png')
    
    # Create confusion matrix
    create_confusion_matrix_dl(y_test, y_pred, le.classes_, 'MLP Simple', 
                                'results/deep_learning/mlp_simple_confusion_matrix.png')
    
    print('\n=== MLP SIMPLE COMPLETE ===\n')
    
    return model, history, metrics_df, le


def run_mlp_improved():
    """
    MLP Improved Architecture
    - 3 hidden layers [128, 64, 32]
    - Batch Normalization
    - Dropout 0.3
    - Early Stopping + ReduceLROnPlateau
    """
    
    print('=== MLP IMPROVED ===\n')
    
    # Load data
    df = pd.read_csv(DATA_PROCESSED / 'merged_dataset_labels.csv')
    
    # Prepare features and labels
    X = df.drop(['Country', 'Year', 'Credit_Rating_Label'], axis=1)
    y = df['Credit_Rating_Label']
    
    feature_names = X.columns.tolist()
    
    print(f'Dataset: {X.shape[0]} observations, {X.shape[1]} features')
    print(f'Classes: {y.nunique()} credit ratings\n')
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    n_classes = len(le.classes_)
    
    print(f'Label encoding: {n_classes} classes\n')
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data: 60% train, 20% validation, 20% test
    # Note: Some classes have very few samples, so we cannot use stratify
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y_encoded, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    print(f'Data split:')
    print(f'  Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)')
    print(f'  Validation: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)')
    print(f'  Test: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)\n')
    
    # Build model
    print('Building MLP Improved architecture...')
    
    tf.random.set_seed(42)
    np.random.seed(42)
    
    model = keras.Sequential([
        layers.Input(shape=(X.shape[1],)),
        
        # Layer 1
        layers.Dense(128, activation='relu', name='hidden_layer_1'),
        layers.BatchNormalization(name='batch_norm_1'),
        layers.Dropout(0.3, name='dropout_1'),
        
        # Layer 2
        layers.Dense(64, activation='relu', name='hidden_layer_2'),
        layers.BatchNormalization(name='batch_norm_2'),
        layers.Dropout(0.3, name='dropout_2'),
        
        # Layer 3
        layers.Dense(32, activation='relu', name='hidden_layer_3'),
        layers.Dropout(0.2, name='dropout_3'),
        
        # Output
        layers.Dense(n_classes, activation='softmax', name='output_layer')
    ], name='MLP_Improved')
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print('\nModel architecture:')
    model.summary()
    
    # Callbacks
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=0.00001,
        verbose=1
    )
    
    # Train model
    print('\nTraining MLP Improved...\n')
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # Evaluate on test set
    print('\n' + '='*60)
    print('EVALUATION ON TEST SET')
    print('='*60 + '\n')
    
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f'Test Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)')
    print(f'F1-Weighted:      {f1_weighted:.4f}')
    print(f'F1-Macro:         {f1_macro:.4f}')
    print(f'Precision:        {precision:.4f}')
    print(f'Recall:           {recall:.4f}\n')
    
    # Save metrics
    os.makedirs('results/deep_learning', exist_ok=True)
    
    metrics_df = pd.DataFrame({
        'Model': ['MLP_Improved'],
        'Accuracy': [accuracy],
        'F1_Weighted': [f1_weighted],
        'F1_Macro': [f1_macro],
        'Precision': [precision],
        'Recall': [recall],
        'Epochs_Trained': [len(history.history['loss'])],
        'Best_Epoch': [len(history.history['loss']) - early_stop.patience]
    })
    
    metrics_df.to_csv('results/deep_learning/mlp_improved_metrics.csv', index=False)
    print('✓ Metrics saved to: results/deep_learning/mlp_improved_metrics.csv')
    
    # Save model
    model.save('results/deep_learning/mlp_improved_model.h5')
    print('✓ Model saved to: results/deep_learning/mlp_improved_model.h5')
    
    # Create learning curves
    create_learning_curves(history, 'MLP Improved', 'results/deep_learning/mlp_improved_learning_curves.png')
    
    # Create confusion matrix
    create_confusion_matrix_dl(y_test, y_pred, le.classes_, 'MLP Improved', 
                                'results/deep_learning/mlp_improved_confusion_matrix.png')
    
    print('\n=== MLP IMPROVED COMPLETE ===\n')
    
    return model, history, metrics_df, le


def create_learning_curves(history, model_name, output_path):
    """
    Create learning curves (loss and accuracy)
    """
    
    print(f'Creating learning curves for {model_name}...')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss
    axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0].set_title(f'{model_name} - Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    axes[1].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[1].set_title(f'{model_name} - Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'✓ Learning curves saved to: {output_path}')


def create_confusion_matrix_dl(y_test, y_pred, class_names, model_name, output_path):
    """
    Create confusion matrix for deep learning model
    """
    
    print(f'Creating confusion matrix for {model_name}...')
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'},
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Rating', fontsize=12, fontweight='bold')
    plt.ylabel('True Rating', fontsize=12, fontweight='bold')
    plt.title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'✓ Confusion matrix saved to: {output_path}')


def compare_mlp_with_classical():
    """
    Compare MLP models with classical ML models
    """
    
    print('=== COMPARING MLP WITH CLASSICAL ML ===\n')
    
    # Load MLP metrics
    mlp_simple = pd.read_csv('results/deep_learning/mlp_simple_metrics.csv')
    mlp_improved = pd.read_csv('results/deep_learning/mlp_improved_metrics.csv')
    
    # Load classical ML metrics
    classical_metrics = pd.read_csv('results/classification_metrics.csv')
    
    # Get best classical models
    best_classical = classical_metrics.nlargest(3, 'Mean_Accuracy')[['Model', 'Mean_Accuracy', 'Mean_F1_Weighted']]
    
    # Combine results
    comparison_data = []
    
    # Add best classical models
    for _, row in best_classical.iterrows():
        comparison_data.append({
            'Model': row['Model'],
            'Type': 'Classical ML',
            'Accuracy': row['Mean_Accuracy'],
            'F1_Weighted': row['Mean_F1_Weighted']
        })
    
    # Add MLP models
    comparison_data.append({
        'Model': 'MLP Simple',
        'Type': 'Deep Learning',
        'Accuracy': mlp_simple['Accuracy'].values[0],
        'F1_Weighted': mlp_simple['F1_Weighted'].values[0]
    })
    
    comparison_data.append({
        'Model': 'MLP Improved',
        'Type': 'Deep Learning',
        'Accuracy': mlp_improved['Accuracy'].values[0],
        'F1_Weighted': mlp_improved['F1_Weighted'].values[0]
    })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
    
    print('Model Comparison:')
    print('='*80)
    print(comparison_df.to_string(index=False))
    print()
    
    # Save comparison
    comparison_df.to_csv('results/deep_learning/mlp_comparison.csv', index=False)
    print('✓ Comparison saved to: results/deep_learning/mlp_comparison.csv')
    
    # Create comparison visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Accuracy
    colors = ['steelblue' if t == 'Classical ML' else 'coral' for t in comparison_df['Type']]
    axes[0].barh(comparison_df['Model'], comparison_df['Accuracy'], color=colors, alpha=0.8)
    axes[0].set_xlabel('Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_title('Model Comparison - Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlim(0.7, comparison_df['Accuracy'].max() * 1.05)
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (model, acc) in enumerate(zip(comparison_df['Model'], comparison_df['Accuracy'])):
        axes[0].text(acc + 0.005, i, f'{acc:.4f}', va='center', fontsize=9)
    
    # Plot 2: F1-Weighted
    axes[1].barh(comparison_df['Model'], comparison_df['F1_Weighted'], color=colors, alpha=0.8)
    axes[1].set_xlabel('F1-Weighted Score', fontsize=12, fontweight='bold')
    axes[1].set_title('Model Comparison - F1-Weighted', fontsize=14, fontweight='bold')
    axes[1].set_xlim(0.7, comparison_df['F1_Weighted'].max() * 1.05)
    axes[1].grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (model, f1) in enumerate(zip(comparison_df['Model'], comparison_df['F1_Weighted'])):
        axes[1].text(f1 + 0.005, i, f'{f1:.4f}', va='center', fontsize=9)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', alpha=0.8, label='Classical ML'),
        Patch(facecolor='coral', alpha=0.8, label='Deep Learning')
    ]
    axes[1].legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig('results/deep_learning/mlp_comparison_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print('✓ Comparison chart saved to: results/deep_learning/mlp_comparison_chart.png\n')
    
    # Print insights
    best_model = comparison_df.iloc[0]
    print('='*80)
    print('KEY INSIGHTS')
    print('='*80)
    print(f'\nBest Model Overall: {best_model["Model"]} ({best_model["Type"]})')
    print(f'  Accuracy: {best_model["Accuracy"]:.4f}')
    print(f'  F1-Weighted: {best_model["F1_Weighted"]:.4f}')
    
    mlp_improved_acc = comparison_df[comparison_df['Model'] == 'MLP Improved']['Accuracy'].values[0]
    best_classical_acc = comparison_df[comparison_df['Type'] == 'Classical ML']['Accuracy'].max()
    
    diff = mlp_improved_acc - best_classical_acc
    if diff > 0:
        print(f'\n✓ MLP Improved outperforms best Classical ML by {diff:.4f} ({diff*100:.2f}%)')
    elif diff < 0:
        print(f'\n✗ MLP Improved underperforms best Classical ML by {abs(diff):.4f} ({abs(diff)*100:.2f}%)')
    else:
        print(f'\n= MLP Improved matches best Classical ML performance')
    
    print('\n=== COMPARISON COMPLETE ===\n')
    
    return comparison_df


def compare_optimizers():
    """
    Compare different optimizers: SGD, Adam, RMSprop, Adagrad
    """
    
    print('=== COMPARING OPTIMIZERS ===\n')
    
    # Load data
    df = pd.read_csv(DATA_PROCESSED / 'merged_dataset_labels.csv')
    X = df.drop(['Country', 'Year', 'Credit_Rating_Label'], axis=1)
    y = df['Credit_Rating_Label']
    
    # Encode and normalize
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    n_classes = len(le.classes_)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_encoded, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f'Testing 4 optimizers on MLP Improved architecture...\n')
    
    # Define optimizers
    optimizers_config = [
        {'name': 'SGD', 'optimizer': keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)},
        {'name': 'Adam', 'optimizer': keras.optimizers.Adam(learning_rate=0.001)},
        {'name': 'RMSprop', 'optimizer': keras.optimizers.RMSprop(learning_rate=0.001)},
        {'name': 'Adagrad', 'optimizer': keras.optimizers.Adagrad(learning_rate=0.01)}
    ]
    
    results = []
    
    for config in optimizers_config:
        print(f"\nTraining with {config['name']}...")
        print('-' * 60)
        
        tf.random.set_seed(42)
        np.random.seed(42)
        
        # Build model
        model = keras.Sequential([
            layers.Input(shape=(X.shape[1],)),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(n_classes, activation='softmax')
        ])
        
        # Compile with specific optimizer
        model.compile(
            optimizer=config['optimizer'],
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train
        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0)
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )
        
        # Evaluate
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        accuracy = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        epochs_trained = len(history.history['loss'])
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        
        results.append({
            'Optimizer': config['name'],
            'Accuracy': accuracy,
            'F1_Weighted': f1_weighted,
            'Epochs': epochs_trained,
            'Train_Loss': final_train_loss,
            'Val_Loss': final_val_loss
        })
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Weighted: {f1_weighted:.4f}")
        print(f"  Epochs: {epochs_trained}")
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Accuracy', ascending=False)
    
    print('\n' + '='*80)
    print('OPTIMIZER COMPARISON RESULTS')
    print('='*80)
    print(results_df.to_string(index=False))
    print()
    
    # Save results
    os.makedirs('results/deep_learning', exist_ok=True)
    results_df.to_csv('results/deep_learning/optimizer_comparison.csv', index=False)
    print('✓ Results saved to: results/deep_learning/optimizer_comparison.csv')
    
    # Visualize comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    axes[0].bar(results_df['Optimizer'], results_df['Accuracy'], color='steelblue', alpha=0.8)
    axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_title('Optimizer Comparison - Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0.5, results_df['Accuracy'].max() * 1.05)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    for i, (opt, acc) in enumerate(zip(results_df['Optimizer'], results_df['Accuracy'])):
        axes[0].text(i, acc + 0.01, f'{acc:.4f}', ha='center', fontsize=10, fontweight='bold')
    
    # Convergence (Epochs)
    axes[1].bar(results_df['Optimizer'], results_df['Epochs'], color='coral', alpha=0.8)
    axes[1].set_ylabel('Epochs to Converge', fontsize=12, fontweight='bold')
    axes[1].set_title('Optimizer Comparison - Convergence Speed', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for i, (opt, epochs) in enumerate(zip(results_df['Optimizer'], results_df['Epochs'])):
        axes[1].text(i, epochs + 1, f'{epochs}', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/deep_learning/optimizer_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print('✓ Visualization saved to: results/deep_learning/optimizer_comparison.png\n')
    
    best_optimizer = results_df.iloc[0]
    print(f"Best Optimizer: {best_optimizer['Optimizer']} (Accuracy: {best_optimizer['Accuracy']:.4f})\n")
    
    print('=== OPTIMIZER COMPARISON COMPLETE ===\n')
    
    return results_df


def compare_learning_rates():
    """
    Compare different learning rates: 0.0001, 0.001, 0.01, 0.1
    """
    
    print('=== COMPARING LEARNING RATES ===\n')
    
    # Load data
    df = pd.read_csv(DATA_PROCESSED / 'merged_dataset_labels.csv')
    X = df.drop(['Country', 'Year', 'Credit_Rating_Label'], axis=1)
    y = df['Credit_Rating_Label']
    
    # Encode and normalize
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    n_classes = len(le.classes_)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_encoded, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f'Testing 4 learning rates with Adam optimizer...\n')
    
    # Define learning rates
    learning_rates = [0.0001, 0.001, 0.01, 0.1]
    
    results = []
    
    for lr in learning_rates:
        print(f"\nTraining with learning_rate={lr}...")
        print('-' * 60)
        
        tf.random.set_seed(42)
        np.random.seed(42)
        
        # Build model
        model = keras.Sequential([
            layers.Input(shape=(X.shape[1],)),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(n_classes, activation='softmax')
        ])
        
        # Compile with specific learning rate
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train
        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0)
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )
        
        # Evaluate
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        accuracy = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        epochs_trained = len(history.history['loss'])
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        
        results.append({
            'Learning_Rate': lr,
            'Accuracy': accuracy,
            'F1_Weighted': f1_weighted,
            'Epochs': epochs_trained,
            'Train_Loss': final_train_loss,
            'Val_Loss': final_val_loss
        })
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Weighted: {f1_weighted:.4f}")
        print(f"  Epochs: {epochs_trained}")
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Accuracy', ascending=False)
    
    print('\n' + '='*80)
    print('LEARNING RATE COMPARISON RESULTS')
    print('='*80)
    print(results_df.to_string(index=False))
    print()
    
    # Save results
    os.makedirs('results/deep_learning', exist_ok=True)
    results_df.to_csv('results/deep_learning/learning_rate_comparison.csv', index=False)
    print('✓ Results saved to: results/deep_learning/learning_rate_comparison.csv')
    
    # Visualize comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy vs LR
    axes[0].plot(results_df['Learning_Rate'], results_df['Accuracy'], 'bo-', linewidth=2, markersize=10)
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Learning Rate (log scale)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_title('Learning Rate vs Accuracy', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    for lr, acc in zip(results_df['Learning_Rate'], results_df['Accuracy']):
        axes[0].text(lr, acc + 0.01, f'{acc:.4f}', ha='center', fontsize=9)
    
    # Convergence speed
    axes[1].bar([str(lr) for lr in results_df['Learning_Rate']], results_df['Epochs'], color='coral', alpha=0.8)
    axes[1].set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Epochs to Converge', fontsize=12, fontweight='bold')
    axes[1].set_title('Learning Rate vs Convergence Speed', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for i, (lr, epochs) in enumerate(zip(results_df['Learning_Rate'], results_df['Epochs'])):
        axes[1].text(i, epochs + 1, f'{epochs}', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/deep_learning/learning_rate_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print('✓ Visualization saved to: results/deep_learning/learning_rate_comparison.png\n')
    
    best_lr = results_df.iloc[0]
    print(f"Best Learning Rate: {best_lr['Learning_Rate']} (Accuracy: {best_lr['Accuracy']:.4f})\n")
    
    print('=== LEARNING RATE COMPARISON COMPLETE ===\n')
    
    return results_df


def compare_l2_regularization():
    """
    Compare different L2 regularization strengths: 0, 0.001, 0.01, 0.1
    """
    
    print('=== COMPARING L2 REGULARIZATION ===\n')
    
    from tensorflow.keras import regularizers
    
    # Load data
    df = pd.read_csv(DATA_PROCESSED / 'merged_dataset_labels.csv')
    X = df.drop(['Country', 'Year', 'Credit_Rating_Label'], axis=1)
    y = df['Credit_Rating_Label']
    
    # Encode and normalize
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    n_classes = len(le.classes_)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_encoded, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f'Testing 4 L2 regularization strengths...\n')
    
    # Define L2 strengths
    l2_strengths = [0, 0.001, 0.01, 0.1]
    
    results = []
    
    for l2 in l2_strengths:
        print(f"\nTraining with L2={l2}...")
        print('-' * 60)
        
        tf.random.set_seed(42)
        np.random.seed(42)
        
        # Build model with L2 regularization
        if l2 == 0:
            model = keras.Sequential([
                layers.Input(shape=(X.shape[1],)),
                layers.Dense(128, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(32, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(n_classes, activation='softmax')
            ])
        else:
            model = keras.Sequential([
                layers.Input(shape=(X.shape[1],)),
                layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2)),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2)),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2)),
                layers.Dropout(0.2),
                layers.Dense(n_classes, activation='softmax')
            ])
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train
        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0)
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )
        
        # Evaluate
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        accuracy = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        epochs_trained = len(history.history['loss'])
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        overfitting_gap = final_train_loss - final_val_loss
        
        results.append({
            'L2_Strength': l2,
            'Accuracy': accuracy,
            'F1_Weighted': f1_weighted,
            'Epochs': epochs_trained,
            'Train_Loss': final_train_loss,
            'Val_Loss': final_val_loss,
            'Overfitting_Gap': overfitting_gap
        })
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Weighted: {f1_weighted:.4f}")
        print(f"  Overfitting Gap: {overfitting_gap:.4f}")
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Accuracy', ascending=False)
    
    print('\n' + '='*80)
    print('L2 REGULARIZATION COMPARISON RESULTS')
    print('='*80)
    print(results_df.to_string(index=False))
    print()
    
    # Save results
    os.makedirs('results/deep_learning', exist_ok=True)
    results_df.to_csv('results/deep_learning/l2_regularization_comparison.csv', index=False)
    print('✓ Results saved to: results/deep_learning/l2_regularization_comparison.csv')
    
    # Visualize comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy vs L2
    axes[0].plot([str(l2) for l2 in results_df['L2_Strength']], results_df['Accuracy'], 'go-', linewidth=2, markersize=10)
    axes[0].set_xlabel('L2 Regularization Strength', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_title('L2 Regularization vs Accuracy', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    for i, (l2, acc) in enumerate(zip(results_df['L2_Strength'], results_df['Accuracy'])):
        axes[0].text(i, acc + 0.01, f'{acc:.4f}', ha='center', fontsize=9)
    
    # Overfitting gap
    axes[1].bar([str(l2) for l2 in results_df['L2_Strength']], results_df['Overfitting_Gap'], color='coral', alpha=0.8)
    axes[1].set_xlabel('L2 Regularization Strength', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Overfitting Gap (Train - Val Loss)', fontsize=12, fontweight='bold')
    axes[1].set_title('L2 Regularization vs Overfitting', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    for i, (l2, gap) in enumerate(zip(results_df['L2_Strength'], results_df['Overfitting_Gap'])):
        axes[1].text(i, gap + 0.01, f'{gap:.4f}', ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/deep_learning/l2_regularization_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print('✓ Visualization saved to: results/deep_learning/l2_regularization_comparison.png\n')
    
    best_l2 = results_df.iloc[0]
    print(f"Best L2 Strength: {best_l2['L2_Strength']} (Accuracy: {best_l2['Accuracy']:.4f})\n")
    
    print('=== L2 REGULARIZATION COMPARISON COMPLETE ===\n')
    
    return results_df
