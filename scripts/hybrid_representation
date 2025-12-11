"""
Hybrid Molecular Representation Creation

This module provides functionality to create hybrid molecular representations
by combining the top N features from different representation types based on
feature importance scores.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Lasso
import warnings

warnings.filterwarnings('ignore')

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("WARNING: SHAP not available. Install with: pip install shap")
    print("Falling back to other importance methods.")


def calculate_feature_importance(x_train, y_train, method='random_forest', **kwargs):
    """
    Calculate feature importance scores for features.
    
    Args:
        x_train: Training features (n_samples, n_features)
        y_train: Training targets (n_samples,)
        method: Method for importance calculation
            - 'shap': SHAP values using TreeExplainer (RECOMMENDED)
            - 'shap_kernel': SHAP KernelExplainer (slower, model-agnostic)
            - 'random_forest': Use RF feature importances
            - 'mutual_info': Use mutual information
            - 'correlation': Use absolute correlation with target
            - 'lasso': Use LASSO coefficient magnitudes
        **kwargs: Additional arguments for specific methods
        
    Returns:
        importance_scores: Array of importance scores (n_features,)
    """
    n_features = x_train.shape[1]
    
    if method == 'shap':
        if not SHAP_AVAILABLE:
            print("WARNING: SHAP not available, falling back to random_forest")
            method = 'random_forest'
        else:
            # Train a model for SHAP
            n_estimators = kwargs.get('n_estimators', 100)
            max_depth = kwargs.get('max_depth', 10)
            random_state = kwargs.get('random_state', 42)
            
            print("  Training model for SHAP analysis...")
            rf = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1
            )
            rf.fit(x_train, y_train)
            
            # Use TreeExplainer for speed
            print("  Computing SHAP values...")
            explainer = shap.TreeExplainer(rf)
            
            # Sample if dataset is large (SHAP can be slow)
            max_samples = kwargs.get('shap_max_samples', 1000)
            if x_train.shape[0] > max_samples:
                sample_indices = np.random.RandomState(random_state).choice(
                    x_train.shape[0], max_samples, replace=False
                )
                x_sample = x_train[sample_indices]
            else:
                x_sample = x_train
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(x_sample)
            
            # Use mean absolute SHAP value as importance
            importance_scores = np.mean(np.abs(shap_values), axis=0)
            
            print("  SHAP computation complete")
            return importance_scores
    
    if method == 'shap_kernel':
        if not SHAP_AVAILABLE:
            print("WARNING: SHAP not available, falling back to random_forest")
            method = 'random_forest'
        else:
            # Train a model
            n_estimators = kwargs.get('n_estimators', 100)
            max_depth = kwargs.get('max_depth', 10)
            random_state = kwargs.get('random_state', 42)
            
            print("  Training model for SHAP Kernel analysis...")
            rf = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1
            )
            rf.fit(x_train, y_train)
            
            # Use KernelExplainer (model-agnostic but slower)
            print("  Computing SHAP values with KernelExplainer (this may take a while)...")
            
            # Sample background data
            n_background = kwargs.get('shap_background_samples', 100)
            background_indices = np.random.RandomState(random_state).choice(
                x_train.shape[0], min(n_background, x_train.shape[0]), replace=False
            )
            background = x_train[background_indices]
            
            # Sample evaluation data
            n_eval = kwargs.get('shap_eval_samples', 200)
            eval_indices = np.random.RandomState(random_state + 1).choice(
                x_train.shape[0], min(n_eval, x_train.shape[0]), replace=False
            )
            x_eval = x_train[eval_indices]
            
            explainer = shap.KernelExplainer(rf.predict, background)
            shap_values = explainer.shap_values(x_eval)
            
            # Use mean absolute SHAP value as importance
            importance_scores = np.mean(np.abs(shap_values), axis=0)
            
            print("  SHAP Kernel computation complete")
            return importance_scores
    
    if method == 'random_forest':
        n_estimators = kwargs.get('n_estimators', 100)
        max_depth = kwargs.get('max_depth', 10)
        random_state = kwargs.get('random_state', 42)
        
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        rf.fit(x_train, y_train)
        importance_scores = rf.feature_importances_
        
    elif method == 'mutual_info':
        random_state = kwargs.get('random_state', 42)
        importance_scores = mutual_info_regression(
            x_train, y_train, 
            random_state=random_state,
            n_jobs=-1
        )
        
    elif method == 'correlation':
        # Calculate absolute correlation with target
        importance_scores = np.abs([
            np.corrcoef(x_train[:, i], y_train)[0, 1] 
            for i in range(n_features)
        ])
        # Replace NaN with 0 (for constant features)
        importance_scores = np.nan_to_num(importance_scores, nan=0.0)
        
    elif method == 'lasso':
        alpha = kwargs.get('alpha', 0.01)
        random_state = kwargs.get('random_state', 42)
        
        lasso = Lasso(alpha=alpha, random_state=random_state, max_iter=1000)
        lasso.fit(x_train, y_train)
        importance_scores = np.abs(lasso.coef_)
        
    else:
        raise ValueError(f"Unknown importance method: {method}")
    
    return importance_scores


def select_top_features(importance_scores, n_features):
    """
    Select indices of top N features by importance.
    
    Args:
        importance_scores: Array of importance scores
        n_features: Number of top features to select
        
    Returns:
        selected_indices: Indices of selected features (sorted by importance)
    """
    n_features = min(n_features, len(importance_scores))
    selected_indices = np.argsort(importance_scores)[-n_features:][::-1]
    return selected_indices


def normalize_representation(x_data, method='standard'):
    """
    Normalize features before combining.
    
    Args:
        x_data: Feature matrix to normalize
        method: 'standard' (z-score), 'minmax', or 'none'
        
    Returns:
        normalized_data: Normalized features
        params: Normalization parameters (mean, std) or (min, max)
    """
    if method == 'standard':
        mean = np.mean(x_data, axis=0)
        std = np.std(x_data, axis=0)
        std[std == 0] = 1.0  # Avoid division by zero
        normalized = (x_data - mean) / std
        return normalized, {'mean': mean, 'std': std}
        
    elif method == 'minmax':
        min_val = np.min(x_data, axis=0)
        max_val = np.max(x_data, axis=0)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1.0
        normalized = (x_data - min_val) / range_val
        return normalized, {'min': min_val, 'max': max_val}
        
    elif method == 'none':
        return x_data, {}
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def apply_normalization(x_data, params, method='standard'):
    """Apply normalization using pre-computed parameters."""
    if method == 'standard':
        return (x_data - params['mean']) / params['std']
    elif method == 'minmax':
        range_val = params['max'] - params['min']
        range_val[range_val == 0] = 1.0
        return (x_data - params['min']) / range_val
    elif method == 'none':
        return x_data
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def create_hybrid_representation(
    representations_dict,
    n_per_rep=100,
    importance_method='random_forest',
    normalize_method='standard',
    verbose=True,
    **importance_kwargs
):
    """
    Create hybrid representation by combining top features from multiple representations.
    
    Args:
        representations_dict: Dict with representation names as keys, each containing:
            {
                'continuous_pdv': {'x_train': ..., 'y_train': ..., 'x_test': ..., 'x_val': ...},
                'ecfp4': {'x_train': ..., 'y_train': ..., 'x_test': ..., 'x_val': ...},
                'mol2vec': {'x_train': ..., 'y_train': ..., 'x_test': ..., 'x_val': ...}
            }
        n_per_rep: Number of features to select from each representation
        importance_method: Method for calculating feature importance
        normalize_method: How to normalize features before combining ('standard', 'minmax', 'none')
        verbose: Print progress information
        **importance_kwargs: Additional arguments for importance calculation
        
    Returns:
        hybrid_train, hybrid_test, hybrid_val: Combined feature matrices
        feature_info: Dict with information about selected features
    """
    hybrid_train_parts = []
    hybrid_test_parts = []
    hybrid_val_parts = []
    feature_info = {}
    
    for rep_name, rep_data in representations_dict.items():
        if verbose:
            print(f"\nProcessing {rep_name}...")
        
        x_train = rep_data['x_train']
        y_train = rep_data['y_train']
        x_test = rep_data['x_test']
        x_val = rep_data['x_val']
        
        # Convert to float for processing
        if x_train.dtype == np.uint8:
            x_train = x_train.astype(np.float32)
            x_test = x_test.astype(np.float32)
            x_val = x_val.astype(np.float32)
        
        # Calculate feature importance
        if verbose:
            print(f"  Calculating feature importance using {importance_method}...")
        importance_scores = calculate_feature_importance(
            x_train, y_train, 
            method=importance_method,
            **importance_kwargs
        )
        
        # Select top features
        n_select = min(n_per_rep, x_train.shape[1])
        selected_indices = select_top_features(importance_scores, n_select)
        
        if verbose:
            print(f"  Selected {len(selected_indices)} features")
            print(f"  Top 5 importance scores: {importance_scores[selected_indices[:5]]}")
        
        # Extract selected features
        x_train_selected = x_train[:, selected_indices]
        x_test_selected = x_test[:, selected_indices]
        x_val_selected = x_val[:, selected_indices]
        
        # Normalize if requested
        if normalize_method != 'none':
            if verbose:
                print(f"  Normalizing using {normalize_method}...")
            x_train_selected, norm_params = normalize_representation(
                x_train_selected, method=normalize_method
            )
            x_test_selected = apply_normalization(
                x_test_selected, norm_params, method=normalize_method
            )
            x_val_selected = apply_normalization(
                x_val_selected, norm_params, method=normalize_method
            )
        
        # Store parts
        hybrid_train_parts.append(x_train_selected)
        hybrid_test_parts.append(x_test_selected)
        hybrid_val_parts.append(x_val_selected)
        
        # Store feature information
        feature_info[rep_name] = {
            'selected_indices': selected_indices,
            'importance_scores': importance_scores[selected_indices],
            'n_features': len(selected_indices)
        }
    
    # Concatenate all parts
    if verbose:
        print("\nCombining representations...")
    
    hybrid_train = np.hstack(hybrid_train_parts).astype(np.float32)
    hybrid_test = np.hstack(hybrid_test_parts).astype(np.float32)
    hybrid_val = np.hstack(hybrid_val_parts).astype(np.float32)
    
    if verbose:
        print(f"Hybrid representation shape: {hybrid_train.shape}")
        total_features = sum(info['n_features'] for info in feature_info.values())
        print(f"Total features: {total_features}")
    
    return hybrid_train, hybrid_test, hybrid_val, feature_info


def check_multicollinearity(x_data, threshold=0.95, sample_size=1000):
    """
    Check for highly correlated features (basic check).
    
    Args:
        x_data: Feature matrix
        threshold: Correlation threshold above which to flag
        sample_size: Number of samples to use (for speed)
        
    Returns:
        correlated_pairs: List of (idx1, idx2, correlation) tuples
    """
    # Sample if dataset is large
    if x_data.shape[0] > sample_size:
        indices = np.random.choice(x_data.shape[0], sample_size, replace=False)
        x_sample = x_data[indices]
    else:
        x_sample = x_data
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(x_sample.T)
    
    # Find highly correlated pairs
    correlated_pairs = []
    n_features = x_data.shape[1]
    
    for i in range(n_features):
        for j in range(i+1, n_features):
            if abs(corr_matrix[i, j]) > threshold:
                correlated_pairs.append((i, j, corr_matrix[i, j]))
    
    return correlated_pairs


def check_feature_uniqueness(x_data, threshold=0.01):
    """
    Check if features provide unique information.
    
    Args:
        x_data: Feature matrix
        threshold: Variance threshold below which to flag
        
    Returns:
        low_variance_indices: Indices of features with low variance
    """
    variances = np.var(x_data, axis=0)
    low_variance_indices = np.where(variances < threshold)[0]
    return low_variance_indices