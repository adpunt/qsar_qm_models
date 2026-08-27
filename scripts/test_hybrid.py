"""
Test script for hybrid representation functionality

Run this to verify your hybrid representation system works before integrating
into main.py. This creates synthetic data similar to your molecular descriptors.
"""

import numpy as np
import sys
sys.path.append('.')

from hybrid_representation import (
    create_hybrid_representation,
    calculate_feature_importance,
    select_top_features,
    check_multicollinearity,
    check_feature_uniqueness
)
from hybrid_diagnostics import analyze_hybrid_features, compare_representations


def generate_synthetic_molecular_data(n_samples=1000, noise_level=0.1, random_state=42):
    """
    Generate synthetic data mimicking molecular representations.
    
    Returns data similar to:
    - continuous_pdv: 200 continuous features
    - ecfp4: 2048 binary features  
    - mol2vec: 300 continuous features
    """
    np.random.seed(random_state)
    
    # True underlying function: y = f(latent features) + noise
    n_latent = 10
    latent_features = np.random.randn(n_samples, n_latent)
    
    # True coefficients for target
    true_coef = np.random.randn(n_latent) * 2
    y = latent_features @ true_coef + np.random.randn(n_samples) * noise_level
    
    # Continuous PDV (200 features) - descriptors with some signal
    continuous_pdv = np.random.randn(n_samples, 200).astype(np.float32)
    # Inject signal into first 50 features
    continuous_pdv[:, :50] = latent_features[:, :5] @ np.random.randn(5, 50) + continuous_pdv[:, :50] * 0.5
    
    # ECFP4 (2048 binary features) - sparse binary fingerprint
    ecfp4 = (np.random.rand(n_samples, 2048) > 0.95).astype(np.uint8)
    # Inject some signal into first 100 bits
    signal_bits = (latent_features[:, :3] @ np.random.randn(3, 100) > 0).astype(np.uint8)
    ecfp4[:, :100] = signal_bits
    
    # mol2vec (300 features) - continuous embeddings
    mol2vec = np.random.randn(n_samples, 300).astype(np.float32)
    # Inject signal into first 30 features
    mol2vec[:, :30] = latent_features[:, :3] @ np.random.randn(3, 30) + mol2vec[:, :30] * 0.5
    
    # Split into train/test/val
    n_train = int(0.8 * n_samples)
    n_test = int(0.1 * n_samples)
    
    data = {
        'continuous_pdv': {
            'x_train': continuous_pdv[:n_train],
            'y_train': y[:n_train],
            'x_test': continuous_pdv[n_train:n_train+n_test],
            'x_val': continuous_pdv[n_train+n_test:]
        },
        'ecfp4': {
            'x_train': ecfp4[:n_train],
            'y_train': y[:n_train],
            'x_test': ecfp4[n_train:n_train+n_test],
            'x_val': ecfp4[n_train+n_test:]
        },
        'mol2vec': {
            'x_train': mol2vec[:n_train],
            'y_train': y[:n_train],
            'x_test': mol2vec[n_train:n_train+n_test],
            'x_val': mol2vec[n_train+n_test:]
        }
    }
    
    y_splits = {
        'y_train': y[:n_train],
        'y_test': y[n_train:n_train+n_test],
        'y_val': y[n_train+n_test:]
    }
    
    return data, y_splits


def test_feature_importance():
    """Test feature importance calculation methods."""
    print("\n" + "="*70)
    print("TEST 1: Feature Importance Calculation")
    print("="*70)
    
    # Generate simple data
    X = np.random.randn(100, 20)
    # Make first 5 features correlated with target
    y = X[:, :5].sum(axis=1) + np.random.randn(100) * 0.1
    
    methods = ['random_forest', 'mutual_info', 'correlation', 'lasso']
    
    # Check if SHAP is available
    try:
        import shap
        methods.insert(0, 'shap')  # Add SHAP as first method to test
        print("\n✓ SHAP is available and will be tested")
    except ImportError:
        print("\n⚠ SHAP not available - install with: pip install shap")
        print("  Proceeding with other methods...")
    
    for method in methods:
        print(f"\n{method}:")
        try:
            importance = calculate_feature_importance(X, y, method=method, random_state=42)
            top_5 = np.argsort(importance)[-5:][::-1]
            print(f"  Top 5 features: {top_5}")
            print(f"  Their scores: {importance[top_5]}")
            
            # Check if informative features are ranked highly
            informative_in_top5 = sum(1 for i in top_5 if i < 5)
            print(f"  Informative features in top 5: {informative_in_top5}/5")
            
            if informative_in_top5 >= 3:
                print(f"  ✓ PASS: Found most informative features")
            else:
                print(f"  ⚠ WARNING: May not be finding informative features well")
                
        except Exception as e:
            print(f"  ✗ FAIL: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)


def test_hybrid_creation():
    """Test hybrid representation creation."""
    print("\n" + "="*70)
    print("TEST 2: Hybrid Representation Creation")
    print("="*70)
    
    # Generate synthetic data
    data, y_splits = generate_synthetic_molecular_data(n_samples=500)
    
    print("\nInput representations:")
    for name, rep_data in data.items():
        print(f"  {name}: {rep_data['x_train'].shape[1]} features")
    
    # Determine which importance method to use
    try:
        import shap
        importance_method = 'shap'
        print("\n✓ Using SHAP for importance calculation (recommended)")
    except ImportError:
        importance_method = 'random_forest'
        print("\n⚠ SHAP not available, using random_forest")
    
    # Test hybrid creation
    print(f"\nCreating hybrid (50 features per representation, {importance_method} importance)...")
    try:
        hybrid_train, hybrid_test, hybrid_val, feature_info = create_hybrid_representation(
            representations_dict=data,
            n_per_rep=50,
            importance_method=importance_method,
            normalize_method='standard',
            verbose=False,
            random_state=42
        )
        
        print(f"✓ Hybrid created successfully!")
        print(f"  Shape: {hybrid_train.shape}")
        print(f"  Expected: ({data['continuous_pdv']['x_train'].shape[0]}, 150)")
        
        # Verify shape
        expected_features = 50 * 3  # 50 from each of 3 representations
        if hybrid_train.shape[1] == expected_features:
            print(f"  ✓ PASS: Correct number of features")
        else:
            print(f"  ✗ FAIL: Expected {expected_features}, got {hybrid_train.shape[1]}")
        
        # Verify data quality
        if not np.any(np.isnan(hybrid_train)):
            print(f"  ✓ PASS: No NaN values")
        else:
            print(f"  ✗ FAIL: Contains NaN values")
        
        if not np.any(np.isinf(hybrid_train)):
            print(f"  ✓ PASS: No infinite values")
        else:
            print(f"  ✗ FAIL: Contains infinite values")
        
        # Check feature info
        print(f"\nFeature info:")
        for rep_name, info in feature_info.items():
            print(f"  {rep_name}: {info['n_features']} features selected")
            print(f"    Mean importance: {np.mean(info['importance_scores']):.4f}")
        
        return hybrid_train, hybrid_test, hybrid_val, feature_info, y_splits
        
    except Exception as e:
        print(f"✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None


def test_diagnostics(hybrid_train, feature_info, y_train):
    """Test diagnostic functions."""
    print("\n" + "="*70)
    print("TEST 3: Diagnostic Functions")
    print("="*70)
    
    hybrid_data = {
        'x_train': hybrid_train,
        'y_train': y_train
    }
    
    try:
        # Test multicollinearity check
        print("\nChecking multicollinearity...")
        corr_pairs = check_multicollinearity(hybrid_train, threshold=0.9)
        print(f"  Highly correlated pairs (|r| > 0.9): {len(corr_pairs)}")
        print(f"  ✓ PASS: Multicollinearity check works")
        
        # Test uniqueness check
        print("\nChecking feature uniqueness...")
        low_var = check_feature_uniqueness(hybrid_train, threshold=0.01)
        print(f"  Low variance features: {len(low_var)}")
        print(f"  ✓ PASS: Uniqueness check works")
        
        # Test analysis (without plotting)
        print("\nRunning full analysis...")
        analyze_hybrid_features(hybrid_data, feature_info, save_path=None)
        print(f"  ✓ PASS: Analysis completed")
        
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        import traceback
        traceback.print_exc()


def test_normalization_methods():
    """Test different normalization methods."""
    print("\n" + "="*70)
    print("TEST 4: Normalization Methods")
    print("="*70)
    
    data, y_splits = generate_synthetic_molecular_data(n_samples=200)
    
    methods = ['standard', 'minmax', 'none']
    
    for method in methods:
        print(f"\nTesting {method} normalization...")
        try:
            hybrid_train, hybrid_test, hybrid_val, feature_info = create_hybrid_representation(
                representations_dict=data,
                n_per_rep=30,
                importance_method='correlation',  # Fast method
                normalize_method=method,
                verbose=False
            )
            
            print(f"  ✓ PASS: {method} normalization works")
            print(f"    Train range: [{hybrid_train.min():.2f}, {hybrid_train.max():.2f}]")
            print(f"    Train mean: {hybrid_train.mean():.4f}")
            print(f"    Train std: {hybrid_train.std():.4f}")
            
        except Exception as e:
            print(f"  ✗ FAIL: {e}")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("HYBRID REPRESENTATION TEST SUITE")
    print("="*70)
    
    # Test 1: Feature importance
    test_feature_importance()
    
    # Test 2: Hybrid creation
    hybrid_train, hybrid_test, hybrid_val, feature_info, y_splits = test_hybrid_creation()
    
    # Test 3: Diagnostics (if hybrid was created successfully)
    if hybrid_train is not None:
        test_diagnostics(hybrid_train, feature_info, y_splits['y_train'])
    
    # Test 4: Normalization
    test_normalization_methods()
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETE")
    print("="*70)
    print("\nIf all tests passed, you're ready to integrate into main.py!")
    print("See INTEGRATION_GUIDE.py for step-by-step instructions.")


if __name__ == "__main__":
    run_all_tests()