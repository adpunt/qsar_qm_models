"""
Diagnostic tools for hybrid representation analysis

Use these functions to analyze the quality and characteristics of your hybrid representation.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns


def analyze_hybrid_features(hybrid_data, feature_info, save_path=None):
    """
    Comprehensive analysis of hybrid representation.
    
    Args:
        hybrid_data: Dict with 'x_train', 'y_train' etc.
        feature_info: Feature information from create_hybrid_representation
        save_path: Optional path to save analysis plots
    """
    x_train = hybrid_data['x_train']
    y_train = hybrid_data['y_train']
    
    print("\n" + "="*70)
    print("HYBRID REPRESENTATION ANALYSIS")
    print("="*70)
    
    # 1. Basic statistics
    print("\n1. BASIC STATISTICS")
    print("-" * 70)
    print(f"Total samples (train): {x_train.shape[0]}")
    print(f"Total features: {x_train.shape[1]}")
    print(f"Feature dtype: {x_train.dtype}")
    print(f"Memory usage: {x_train.nbytes / 1024**2:.2f} MB")
    
    # 2. Per-representation statistics
    print("\n2. FEATURES PER REPRESENTATION")
    print("-" * 70)
    start_idx = 0
    for rep_name, info in feature_info.items():
        n_features = info['n_features']
        end_idx = start_idx + n_features
        
        rep_data = x_train[:, start_idx:end_idx]
        
        print(f"\n{rep_name}:")
        print(f"  Features: {n_features}")
        print(f"  Mean importance: {np.mean(info['importance_scores']):.6f}")
        print(f"  Max importance: {np.max(info['importance_scores']):.6f}")
        print(f"  Min importance: {np.min(info['importance_scores']):.6f}")
        print(f"  Feature range: [{np.min(rep_data):.4f}, {np.max(rep_data):.4f}]")
        print(f"  Feature mean: {np.mean(rep_data):.4f}")
        print(f"  Feature std: {np.std(rep_data):.4f}")
        
        # Check for constant features
        variances = np.var(rep_data, axis=0)
        n_constant = np.sum(variances < 1e-6)
        if n_constant > 0:
            print(f"  WARNING: {n_constant} nearly constant features detected!")
        
        start_idx = end_idx
    
    # 3. Correlation analysis
    print("\n3. FEATURE CORRELATION ANALYSIS")
    print("-" * 70)
    
    # Sample for speed if dataset is large
    n_samples = min(1000, x_train.shape[0])
    sample_indices = np.random.choice(x_train.shape[0], n_samples, replace=False)
    x_sample = x_train[sample_indices]
    
    corr_matrix = np.corrcoef(x_sample.T)
    
    # Find highly correlated pairs
    high_corr_pairs = []
    n_features = x_train.shape[1]
    for i in range(n_features):
        for j in range(i+1, n_features):
            if abs(corr_matrix[i, j]) > 0.9:
                high_corr_pairs.append((i, j, corr_matrix[i, j]))
    
    print(f"Highly correlated pairs (|r| > 0.9): {len(high_corr_pairs)}")
    if len(high_corr_pairs) > 0:
        print(f"Top 5 correlations:")
        sorted_pairs = sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:5]
        for i, j, corr in sorted_pairs:
            print(f"  Feature {i} <-> Feature {j}: r = {corr:.3f}")
    
    # 4. Target correlation
    print("\n4. TARGET CORRELATION")
    print("-" * 70)
    target_corrs = [np.corrcoef(x_train[:, i], y_train)[0, 1] for i in range(n_features)]
    target_corrs = np.array(target_corrs)
    target_corrs = np.nan_to_num(target_corrs, nan=0.0)
    
    print(f"Mean absolute correlation with target: {np.mean(np.abs(target_corrs)):.4f}")
    print(f"Max absolute correlation: {np.max(np.abs(target_corrs)):.4f}")
    print(f"Features with |r| > 0.5: {np.sum(np.abs(target_corrs) > 0.5)}")
    
    # Top correlated features per representation
    start_idx = 0
    for rep_name, info in feature_info.items():
        n_features_rep = info['n_features']
        end_idx = start_idx + n_features_rep
        rep_corrs = target_corrs[start_idx:end_idx]
        best_idx = start_idx + np.argmax(np.abs(rep_corrs))
        print(f"{rep_name} best feature: idx={best_idx}, |r|={np.abs(target_corrs[best_idx]):.4f}")
        start_idx = end_idx
    
    # 5. Generate plots if requested
    if save_path:
        print("\n5. GENERATING DIAGNOSTIC PLOTS")
        print("-" * 70)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Feature importance distribution
        ax = axes[0, 0]
        start_idx = 0
        for rep_name, info in feature_info.items():
            n_features_rep = info['n_features']
            end_idx = start_idx + n_features_rep
            ax.bar(range(start_idx, end_idx), info['importance_scores'], 
                   label=rep_name, alpha=0.7)
            start_idx = end_idx
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Importance Score')
        ax.set_title('Feature Importance Distribution')
        ax.legend()
        
        # Plot 2: Target correlation
        ax = axes[0, 1]
        ax.bar(range(len(target_corrs)), np.abs(target_corrs), alpha=0.7)
        ax.axhline(y=0.5, color='r', linestyle='--', label='|r| = 0.5')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('|Correlation with Target|')
        ax.set_title('Absolute Correlation with Target')
        ax.legend()
        
        # Plot 3: Correlation heatmap (sample)
        ax = axes[1, 0]
        # Show only first 50 features for readability
        n_show = min(50, corr_matrix.shape[0])
        sns.heatmap(corr_matrix[:n_show, :n_show], 
                   cmap='coolwarm', center=0, vmin=-1, vmax=1,
                   square=True, ax=ax, cbar_kws={'label': 'Correlation'})
        ax.set_title(f'Feature Correlation (first {n_show} features)')
        
        # Plot 4: PCA visualization
        ax = axes[1, 1]
        if x_train.shape[0] > 100:  # Only if we have enough samples
            pca = PCA(n_components=2)
            x_pca = pca.fit_transform(x_sample)
            scatter = ax.scatter(x_pca[:, 0], x_pca[:, 1], 
                               c=y_train[sample_indices], 
                               cmap='viridis', alpha=0.6)
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
            ax.set_title('PCA Visualization (colored by target)')
            plt.colorbar(scatter, ax=ax, label='Target Value')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plots saved to: {save_path}")
        plt.close()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70 + "\n")


def compare_representations(rep_data_dict, y_train, metric='target_correlation'):
    """
    Compare different representations including hybrid.
    
    Args:
        rep_data_dict: Dict with representation names as keys and x_train arrays as values
        y_train: Training targets
        metric: 'target_correlation', 'variance', or 'dimensionality'
    """
    print("\n" + "="*70)
    print(f"REPRESENTATION COMPARISON ({metric})")
    print("="*70)
    
    results = {}
    
    for rep_name, x_train in rep_data_dict.items():
        if metric == 'target_correlation':
            # Calculate mean absolute correlation with target
            n_features = x_train.shape[1]
            corrs = [np.corrcoef(x_train[:, i], y_train)[0, 1] for i in range(n_features)]
            corrs = np.array(corrs)
            corrs = np.nan_to_num(corrs, nan=0.0)
            score = np.mean(np.abs(corrs))
            
        elif metric == 'variance':
            # Calculate mean variance
            score = np.mean(np.var(x_train, axis=0))
            
        elif metric == 'dimensionality':
            # Number of features
            score = x_train.shape[1]
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        results[rep_name] = score
        print(f"{rep_name:20s}: {score:.6f}")
    
    print("="*70 + "\n")
    return results


def save_feature_rankings(feature_info, save_path='feature_rankings.csv'):
    """
    Save detailed feature rankings to CSV for later analysis.
    
    Args:
        feature_info: Feature information dict from create_hybrid_representation
        save_path: Path to save CSV file
    """
    import pandas as pd
    
    rows = []
    for rep_name, info in feature_info.items():
        for rank, (idx, score) in enumerate(zip(info['selected_indices'], 
                                                 info['importance_scores'])):
            rows.append({
                'representation': rep_name,
                'rank': rank + 1,
                'original_index': idx,
                'importance_score': score
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"Feature rankings saved to: {save_path}")
    return df


# Example usage:
if __name__ == "__main__":
    print("This module provides diagnostic tools for hybrid representations.")
    print("\nExample usage:")
    print("""
    from hybrid_representation import create_hybrid_representation
    from hybrid_diagnostics import analyze_hybrid_features, compare_representations
    
    # After creating hybrid:
    hybrid_train, hybrid_test, hybrid_val, feature_info = create_hybrid_representation(...)
    
    # Analyze the hybrid:
    hybrid_data = {
        'x_train': hybrid_train,
        'y_train': y_train,
        'x_test': hybrid_test,
        'y_test': y_test,
        'x_val': hybrid_val,
        'y_val': y_val
    }
    
    analyze_hybrid_features(hybrid_data, feature_info, save_path='hybrid_analysis.png')
    
    # Compare with original representations:
    rep_comparison = {
        'continuous_pdv': x_train_pdv,
        'ecfp4': x_train_ecfp4,
        'mol2vec': x_train_mol2vec,
        'hybrid': hybrid_train
    }
    
    compare_representations(rep_comparison, y_train, metric='target_correlation')
    """)