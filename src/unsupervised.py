"""
Unsupervised Learning Module for Sovereign Risk Prediction
Phase 3: K-Means Clustering and PCA Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
import warnings
warnings.filterwarnings('ignore')


def find_optimal_k(k_range=[2, 3, 4, 5, 6, 7, 8]):
    """
    Find optimal number of clusters using Elbow Method and Silhouette Score
    
    Parameters:
    - k_range: List of K values to test (default [2,3,4,5,6,7,8])
    
    Returns:
    - optimal_k: Recommended number of clusters
    """
    
    print('=== FINDING OPTIMAL K FOR K-MEANS CLUSTERING ===\n')
    
    # Load data
    df = pd.read_csv('data/processed/merged_dataset_labels.csv')
    
    # Extract features (8 economic indicators)
    X = df.drop(['Country', 'Year', 'Credit_Rating_Label'], axis=1)
    
    print(f'Dataset: {X.shape[0]} observations, {X.shape[1]} features')
    print(f'Testing K values: {k_range}\n')
    
    # Normalize features (important for K-Means)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Store results
    inertias = []
    silhouette_scores = []
    
    # Test each K value
    for k in k_range:
        # Create and fit K-Means model
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Calculate metrics
        inertia = kmeans.inertia_
        silhouette = silhouette_score(X_scaled, clusters)
        
        inertias.append(inertia)
        silhouette_scores.append(silhouette)
        
        print(f'K={k}: Inertia={inertia:.2f}, Silhouette={silhouette:.4f}')
    
    # Find optimal K (highest silhouette score)
    optimal_idx = np.argmax(silhouette_scores)
    optimal_k = k_range[optimal_idx]
    
    print(f'\n✓ Optimal K = {optimal_k} (Silhouette Score: {silhouette_scores[optimal_idx]:.4f})\n')
    
    # Create Elbow Method visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Inertia (Elbow Method)
    axes[0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12, fontweight='bold')
    axes[0].set_title('Elbow Method - Inertia', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(x=optimal_k, color='red', linestyle='--', alpha=0.5, label=f'Optimal K={optimal_k}')
    axes[0].legend()
    
    # Plot 2: Silhouette Score
    axes[1].plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
    axes[1].set_title('Silhouette Score by K', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Good threshold (0.5)')
    axes[1].axvline(x=optimal_k, color='red', linestyle='--', alpha=0.5, label=f'Optimal K={optimal_k}')
    axes[1].legend()
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs('results/clustering', exist_ok=True)
    output_path = 'results/clustering/elbow_method.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'✓ Elbow method visualization saved to: {output_path}\n')
    print('=== OPTIMAL K DETERMINATION COMPLETE ===\n')
    
    return optimal_k


def run_kmeans_clustering(n_clusters=4):
    """
    Apply K-Means clustering and analyze results
    
    Parameters:
    - n_clusters: Number of clusters (default 4)
    
    Returns:
    - kmeans_model: Trained K-Means model
    - results_df: DataFrame with cluster assignments
    """
    
    print(f'=== K-MEANS CLUSTERING (K={n_clusters}) ===\n')
    
    # Load data
    df = pd.read_csv('data/processed/merged_dataset_labels.csv')
    
    # Keep identifiers separate
    identifiers = df[['Country', 'Year']]
    labels = df['Credit_Rating_Label']
    
    # Extract features
    X = df.drop(['Country', 'Year', 'Credit_Rating_Label'], axis=1)
    feature_names = X.columns.tolist()
    
    print(f'Dataset: {X.shape[0]} observations, {X.shape[1]} features')
    print(f'Features: {feature_names}\n')
    
    # Normalize features
    print('Normalizing features...')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply K-Means
    print(f'Running K-Means with K={n_clusters}...')
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Calculate metrics
    inertia = kmeans.inertia_
    silhouette = silhouette_score(X_scaled, clusters)
    
    print(f'✓ Clustering complete')
    print(f'  Inertia: {inertia:.2f}')
    print(f'  Silhouette Score: {silhouette:.4f}\n')
    
    # Calculate individual silhouette scores and distances
    from sklearn.metrics import silhouette_samples
    silhouette_vals = silhouette_samples(X_scaled, clusters)
    
    distances_to_center = []
    for i, cluster_id in enumerate(clusters):
        center = kmeans.cluster_centers_[cluster_id]
        distance = np.linalg.norm(X_scaled[i] - center)
        distances_to_center.append(distance)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'Country': identifiers['Country'],
        'Year': identifiers['Year'],
        'Credit_Rating_Label': labels,
        'Cluster': clusters,
        'Silhouette_Score': silhouette_vals,
        'Distance_to_Center': distances_to_center
    })
    
    # Add original features
    for col in feature_names:
        results_df[col] = df[col].values
    
    # Analyze cluster profiles
    print('Cluster Profiles:')
    print('='*80)
    
    for cluster_id in range(n_clusters):
        cluster_data = results_df[results_df['Cluster'] == cluster_id]
        n_countries = len(cluster_data)
        
        print(f'\nCluster {cluster_id}: {n_countries} observations')
        print('-'*80)
        
        # Calculate mean values for each feature
        for feature in feature_names:
            mean_val = cluster_data[feature].mean()
            print(f'  {feature:20s}: {mean_val:8.2f}')
        
        # Most common rating
        rating_counts = cluster_data['Credit_Rating_Label'].value_counts()
        dominant_rating = rating_counts.index[0] if len(rating_counts) > 0 else 'N/A'
        print(f'  Dominant Rating     : {dominant_rating}')
        
        # Top countries (most recent year)
        latest_year = cluster_data['Year'].max()
        recent_countries = cluster_data[cluster_data['Year'] == latest_year]['Country'].tolist()[:5]
        print(f'  Sample Countries    : {", ".join(recent_countries)}')
    
    print('\n' + '='*80 + '\n')
    
    # Save detailed results
    os.makedirs('results/clustering', exist_ok=True)
    
    # Save country-level results
    output_path = 'results/clustering/country_clusters.csv'
    results_df.to_csv(output_path, index=False)
    print(f'✓ Country cluster assignments saved to: {output_path}')
    
    # Save cluster profiles
    cluster_profiles = []
    for cluster_id in range(n_clusters):
        cluster_data = results_df[results_df['Cluster'] == cluster_id]
        profile = {'Cluster': cluster_id, 'Count': len(cluster_data)}
        
        for feature in feature_names:
            profile[f'Mean_{feature}'] = cluster_data[feature].mean()
        
        rating_counts = cluster_data['Credit_Rating_Label'].value_counts()
        profile['Dominant_Rating'] = rating_counts.index[0] if len(rating_counts) > 0 else 'N/A'
        
        cluster_profiles.append(profile)
    
    profiles_df = pd.DataFrame(cluster_profiles)
    profiles_path = 'results/clustering/cluster_profiles.csv'
    profiles_df.to_csv(profiles_path, index=False)
    print(f'✓ Cluster profiles saved to: {profiles_path}\n')
    
    print(f'=== K-MEANS CLUSTERING (K={n_clusters}) COMPLETE ===\n')
    
    return kmeans, results_df


def compare_clusters_with_ratings(n_clusters=4):
    """
    Compare K-Means clusters with actual credit ratings
    
    Parameters:
    - n_clusters: Number of clusters used
    
    Returns:
    - comparison_metrics: Dictionary with ARI and NMI scores
    """
    
    print('=== COMPARING CLUSTERS WITH CREDIT RATINGS ===\n')
    
    # Load clustering results
    results_df = pd.read_csv('results/clustering/country_clusters.csv')
    
    # Group ratings into categories
    def rating_to_category(rating):
        if rating in ['AAA', 'AA+', 'AA', 'AA-']:
            return 'High Grade'
        elif rating in ['A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-']:
            return 'Medium Grade'
        elif rating in ['BB+', 'BB', 'BB-', 'B+', 'B', 'B-']:
            return 'Speculative'
        else:  # CCC+, CCC, CCC-, CC, etc.
            return 'High Risk'
    
    results_df['Rating_Category'] = results_df['Credit_Rating_Label'].apply(rating_to_category)
    
    # Create cross-tabulation
    crosstab = pd.crosstab(results_df['Cluster'], results_df['Rating_Category'])
    
    print('Cross-Tabulation (Cluster vs Rating Category):')
    print('='*80)
    print(crosstab)
    print('\n')
    
    # Calculate similarity metrics
    # Convert categories to numeric for metrics
    from sklearn.preprocessing import LabelEncoder
    le_cluster = LabelEncoder()
    le_rating = LabelEncoder()
    
    clusters_encoded = le_cluster.fit_transform(results_df['Cluster'])
    ratings_encoded = le_rating.fit_transform(results_df['Rating_Category'])
    
    ari = adjusted_rand_score(ratings_encoded, clusters_encoded)
    nmi = normalized_mutual_info_score(ratings_encoded, clusters_encoded)
    
    print('Similarity Metrics:')
    print(f'  Adjusted Rand Index (ARI): {ari:.4f}')
    print(f'  Normalized Mutual Information (NMI): {nmi:.4f}\n')
    
    # Interpretation
    if ari > 0.6:
        interpretation = 'Strong correspondence'
    elif ari > 0.3:
        interpretation = 'Moderate correspondence'
    else:
        interpretation = 'Weak correspondence'
    
    print(f'Interpretation: {interpretation} ({ari:.1%})\n')
    
    # Identify outliers (countries in unexpected clusters)
    print('Potential Outliers:')
    print('='*80)
    
    outliers = []
    for cluster_id in range(n_clusters):
        cluster_data = results_df[results_df['Cluster'] == cluster_id]
        rating_counts = cluster_data['Rating_Category'].value_counts()
        
        if len(rating_counts) > 0:
            dominant_category = rating_counts.index[0]
            
            # Find countries in this cluster but different category
            mismatched = cluster_data[cluster_data['Rating_Category'] != dominant_category]
            
            for _, row in mismatched.iterrows():
                outliers.append({
                    'Country': row['Country'],
                    'Year': row['Year'],
                    'Rating': row['Credit_Rating_Label'],
                    'Rating_Category': row['Rating_Category'],
                    'Cluster': cluster_id,
                    'Expected_Category': dominant_category,
                    'Silhouette_Score': row['Silhouette_Score']
                })
    
    if outliers:
        outliers_df = pd.DataFrame(outliers)
        outliers_df = outliers_df.sort_values('Silhouette_Score')
        print(outliers_df.head(10).to_string(index=False))
        print(f'\nTotal outliers detected: {len(outliers)}\n')
    else:
        print('No significant outliers detected.\n')
    
    # Visualize cross-tabulation as heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(crosstab, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Count'})
    plt.title(f'K-Means Clusters vs Credit Rating Categories (K={n_clusters})', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Rating Category', fontsize=12, fontweight='bold')
    plt.ylabel('Cluster', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    output_path = 'results/clustering/cluster_rating_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'✓ Comparison visualization saved to: {output_path}\n')
    print('=== CLUSTER-RATING COMPARISON COMPLETE ===\n')
    
    return {'ARI': ari, 'NMI': nmi, 'Outliers': len(outliers)}


def visualize_clusters_2d(n_clusters=4):
    """
    Visualize clusters in 2D using PCA
    
    Parameters:
    - n_clusters: Number of clusters
    """
    
    print('=== VISUALIZING CLUSTERS IN 2D (PCA) ===\n')
    
    # Load data
    df = pd.read_csv('data/processed/merged_dataset_labels.csv')
    results_df = pd.read_csv('results/clustering/country_clusters.csv')
    
    # Extract features
    X = df.drop(['Country', 'Year', 'Credit_Rating_Label'], axis=1)
    
    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA (8D → 2D)
    print('Applying PCA (8D → 2D)...')
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    variance_explained = pca.explained_variance_ratio_
    print(f'  PC1 explains: {variance_explained[0]:.1%} of variance')
    print(f'  PC2 explains: {variance_explained[1]:.1%} of variance')
    print(f'  Total: {variance_explained.sum():.1%} of variance\n')
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Colored by Cluster
    clusters = results_df['Cluster'].values
    scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, 
                               cmap='viridis', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Add cluster centers
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    centers_pca = pca.transform(kmeans.cluster_centers_)
    axes[0].scatter(centers_pca[:, 0], centers_pca[:, 1], 
                   c='red', marker='*', s=500, edgecolors='black', linewidth=2,
                   label='Cluster Centers')
    
    axes[0].set_xlabel(f'PC1 ({variance_explained[0]:.1%} variance)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel(f'PC2 ({variance_explained[1]:.1%} variance)', fontsize=12, fontweight='bold')
    axes[0].set_title('K-Means Clusters (PCA 2D)', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0], label='Cluster')
    
    # Plot 2: Colored by Rating Category
    def rating_to_numeric(rating):
        rating_order = ['CCC-', 'CCC', 'CCC+', 'CC', 'B-', 'B', 'B+', 
                       'BB-', 'BB', 'BB+', 'BBB-', 'BBB', 'BBB+',
                       'A-', 'A', 'A+', 'AA-', 'AA', 'AA+', 'AAA']
        try:
            return rating_order.index(rating)
        except:
            return 0
    
    ratings_numeric = results_df['Credit_Rating_Label'].apply(rating_to_numeric).values
    scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=ratings_numeric,
                              cmap='RdYlGn', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    axes[1].set_xlabel(f'PC1 ({variance_explained[0]:.1%} variance)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel(f'PC2 ({variance_explained[1]:.1%} variance)', fontsize=12, fontweight='bold')
    axes[1].set_title('Credit Ratings (PCA 2D)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter2, ax=axes[1], label='Rating')
    cbar.set_label('Rating (Low → High)', fontsize=10)
    
    plt.tight_layout()
    
    output_path = 'results/clustering/clusters_2d_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'✓ 2D visualization saved to: {output_path}\n')
    print('=== 2D VISUALIZATION COMPLETE ===\n')
