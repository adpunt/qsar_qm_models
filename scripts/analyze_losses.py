"""
Simple analysis script for loss function comparison experiments
Usage: python analyze_losses.py /path/to/results/
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

def load_results(results_dir):
    """Load all CSV files from results directory"""
    path = Path(results_dir)
    csv_files = [f for f in path.glob("**/loss_*.csv") if 'summary' not in f.name and 'landscape' not in f.name]
    
    if not csv_files:
        print(f"No loss CSV files found in {results_dir}")
        sys.exit(1)
    
    print(f"Loading {len(csv_files)} files...")
    
    # Load files one by one and handle errors
    dfs = []
    for f in csv_files:
        try:
            df_temp = pd.read_csv(f)
            # Check if it has the expected columns
            if 'loss_function' in df_temp.columns and 'mae' in df_temp.columns:
                dfs.append(df_temp)
                print(f"  ✓ {f.name}: {len(df_temp)} rows")
            else:
                print(f"  ⊘ {f.name}: Wrong format (missing loss_function or mae)")
        except Exception as e:
            print(f"  ✗ {f.name}: Error - {e}")
    
    if not dfs:
        print("No valid CSV files could be loaded")
        sys.exit(1)
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} total rows")
    return df

def plot_performance_vs_noise(df, output_dir):
    """Plot MAE vs noise for each loss function"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # MAE by loss
    ax = axes[0, 0]
    for loss in df['loss_function'].unique():
        data = df[df['loss_function'] == loss].groupby('sigma')['mae'].agg(['mean', 'std']).reset_index()
        ax.plot(data['sigma'], data['mean'], marker='o', label=loss)
        ax.fill_between(data['sigma'], data['mean']-data['std'], data['mean']+data['std'], alpha=0.2)
    ax.set_xlabel('Noise σ')
    ax.set_ylabel('MAE')
    ax.set_title('MAE vs Noise')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # R² by loss
    ax = axes[0, 1]
    for loss in df['loss_function'].unique():
        data = df[df['loss_function'] == loss].groupby('sigma')['r2'].mean().reset_index()
        ax.plot(data['sigma'], data['r2'], marker='o', label=loss)
    ax.set_xlabel('Noise σ')
    ax.set_ylabel('R²')
    ax.set_title('R² vs Noise')
    ax.grid(True, alpha=0.3)
    
    # Heatmap at high noise
    ax = axes[1, 0]
    high_noise = df[df['sigma'] >= 0.3].groupby(['loss_function', 'model'])['mae'].mean().unstack()
    sns.heatmap(high_noise, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax)
    ax.set_title('MAE at High Noise (σ≥0.3)')
    
    # Performance drop from clean
    ax = axes[1, 1]
    for loss in df['loss_function'].unique():
        loss_df = df[df['loss_function'] == loss]
        clean = loss_df[loss_df['sigma'] == 0]['mae'].mean()
        if clean > 0:
            drops = []
            sigmas = []
            for s in sorted(loss_df['sigma'].unique()):
                if s > 0:
                    noisy = loss_df[loss_df['sigma'] == s]['mae'].mean()
                    drops.append((noisy - clean) / clean)
                    sigmas.append(s)
            ax.plot(sigmas, drops, marker='o', label=loss)
    ax.set_xlabel('Noise σ')
    ax.set_ylabel('Relative MAE increase')
    ax.set_title('Performance Degradation')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'loss_comparison.png'}")
    plt.close()

def rank_losses(df):
    """Rank losses by performance"""
    rankings = []
    
    for loss in df['loss_function'].unique():
        loss_df = df[df['loss_function'] == loss]
        
        # Overall MAE
        overall_mae = loss_df['mae'].mean()
        
        # High noise performance
        high_noise = loss_df[loss_df['sigma'] >= 0.3]
        high_noise_mae = high_noise['mae'].mean() if len(high_noise) > 0 else overall_mae
        
        # Clean performance
        clean = loss_df[loss_df['sigma'] == 0]
        clean_mae = clean['mae'].mean() if len(clean) > 0 else overall_mae
        
        # Stability (std across iterations)
        stability = loss_df['mae'].std()
        
        rankings.append({
            'loss': loss,
            'overall_mae': overall_mae,
            'high_noise_mae': high_noise_mae,
            'clean_mae': clean_mae,
            'stability_std': stability,
            'n_experiments': len(loss_df)
        })
    
    rankings_df = pd.DataFrame(rankings).sort_values('overall_mae')
    return rankings_df

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_losses.py /path/to/results/")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    output_dir = Path('analysis_output')
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    df = load_results(results_dir)
    
    # Summary stats
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Loss functions: {df['loss_function'].nunique()}")
    print(f"  {list(df['loss_function'].unique())}")
    print(f"Models: {df['model'].nunique()}")
    print(f"Noise levels: {sorted(df['sigma'].unique())}")
    
    # Rankings
    print("\n" + "="*60)
    print("LOSS RANKINGS (by overall MAE)")
    print("="*60)
    rankings = rank_losses(df)
    print(rankings.to_string(index=False))
    rankings.to_csv(output_dir / 'rankings.csv', index=False)
    
    # Best at high noise
    print("\n" + "="*60)
    print("BEST AT HIGH NOISE (σ≥0.3)")
    print("="*60)
    high_noise_best = rankings.nsmallest(5, 'high_noise_mae')[['loss', 'high_noise_mae']]
    print(high_noise_best.to_string(index=False))
    
    # Plot
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)
    plot_performance_vs_noise(df, output_dir)
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    print(f"Results saved to: {output_dir}/")

if __name__ == "__main__":
    main()