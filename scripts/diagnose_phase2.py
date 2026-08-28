"""
PROPER diagnostic - checks ALL data in files
"""

import pandas as pd
from pathlib import Path
import numpy as np

results_dir = Path("../results")
uncertainty_files = list(results_dir.glob("phase2*_uncertainty_values.csv"))

print(f"\n{'='*80}")
print(f"FULL DATA CHECK - ALL ROWS")
print(f"{'='*80}\n")

for filepath in sorted(uncertainty_files):
    print(f"\n{'='*80}")
    print(f"File: {filepath.name}")
    print('='*80)
    
    try:
        # Load FULL file
        df = pd.read_csv(filepath)
        
        print(f"Total rows: {len(df):,}")
        
        # Check columns
        if 'model' in df.columns and 'representation' in df.columns:
            print(f"✓ Has model & representation columns")
            print(f"  Models: {sorted(df['model'].unique())}")
            print(f"  Representations: {sorted(df['representation'].unique())}")
        
        # CHECK SIGMA VALUES - THE KEY ISSUE
        if 'sigma' in df.columns:
            sigmas = sorted(df['sigma'].unique())
            print(f"\n✓ Sigma levels found: {sigmas}")
            print(f"  Number of sigma levels: {len(sigmas)}")
            
            # Count rows per sigma
            print(f"\n  Rows per sigma:")
            for s in sigmas:
                count = len(df[df['sigma'] == s])
                print(f"    σ={s}: {count:,} rows")
        else:
            print(f"\n✗ NO sigma column!")
        
        # Check for decomposition
        if 'epistemic_uncertainty' in df.columns:
            print(f"\n✓ Has epistemic/aleatoric decomposition")
        else:
            print(f"\n✗ No epistemic/aleatoric decomposition")
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}\n")

# Load all and get overall summary
all_data = []
for filepath in uncertainty_files:
    try:
        df = pd.read_csv(filepath)
        all_data.append(df)
    except:
        pass

if all_data:
    combined = pd.concat(all_data, ignore_index=True)
    
    print(f"TOTAL DATA ACROSS ALL FILES:")
    print(f"  Total rows: {len(combined):,}")
    print(f"  Models: {sorted(combined['model'].unique())}")
    print(f"  Representations: {sorted(combined['representation'].unique())}")
    
    if 'sigma' in combined.columns:
        sigmas = sorted(combined['sigma'].unique())
        print(f"\n  Sigma levels: {sigmas}")
        print(f"\n  Rows per sigma (across all files):")
        for s in sigmas:
            count = len(combined[combined['sigma'] == s])
            pct = 100 * count / len(combined)
            print(f"    σ={s}: {count:,} rows ({pct:.1f}%)")