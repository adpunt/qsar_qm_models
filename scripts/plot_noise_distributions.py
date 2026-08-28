import pandas as pd
import altair as alt
import numpy as np
from scipy import stats

# TODO: Run this beforehand
# scp scat9264@gateway.arc.ox.ac.uk:/data/stat-cadd/scat9264/qsar_qm_models/results/distributionGaussian.csv ../results/
# scp scat9264@gateway.arc.ox.ac.uk:/data/stat-cadd/scat9264/qsar_qm_models/results/distributionLeft.csv ../results/
# scp scat9264@gateway.arc.ox.ac.uk:/data/stat-cadd/scat9264/qsar_qm_models/results/distributionRight.csv ../results/
# scp scat9264@gateway.arc.ox.ac.uk:/data/stat-cadd/scat9264/qsar_qm_models/results/distributionU.csv ../results/
# scp scat9264@gateway.arc.ox.ac.uk:/data/stat-cadd/scat9264/qsar_qm_models/results/distributionUniform.csv ../results/

def plot_r2_vs_sigma_with_distributions(df: pd.DataFrame, representation: str):
    """
    For a given representation, generate a chart with one line per (model, distribution),
    showing R² vs sigma. This avoids making one plot per distribution.
    """
    def prepare_data(df):
        df = df[df['rep'] == representation].copy()
        grouped = df.groupby(['sigma', 'model', 'distribution'], as_index=False).agg(list).copy()
        grouped['mean_r2'] = grouped['r2_score'].apply(np.mean)
        # Optional: remove or comment these lines to skip CI
        # grouped['ci'] = grouped['r2_score'].apply(
        #     lambda x: stats.sem(x) * stats.t.ppf((1 + 0.95)/2., len(x)-1) if len(x) > 1 else 0
        # )
        # grouped['lower'] = grouped['mean_r2'] - grouped['ci']
        # grouped['upper'] = grouped['mean_r2'] + grouped['ci']
        grouped.columns.name = None
        grouped = grouped.reset_index(drop=True)
        return grouped
    
    df_prepped = prepare_data(df)
    for (model, dist) in df_prepped[['model', 'distribution']].drop_duplicates().values:
        sub = df_prepped[(df_prepped['model'] == model) & (df_prepped['distribution'] == dist)]
        x = sub['sigma'].values
        y = sub['mean_r2'].values
        if len(x) >= 2:
            order = np.argsort(x)
            x_sorted = x[order]
            y_sorted = y[order]
            auc_score = np.trapz(y_sorted, x_sorted)
            auc_score /= (x_sorted[-1] - x_sorted[0])
        else:
            auc_score = float('nan')
        print(f"{representation.upper()} — {model} — {dist}: AUC = {auc_score:.4f}")
    
    chart = alt.Chart(df_prepped).encode(
        x=alt.X('sigma:Q', title='Sigma'),
        y=alt.Y('mean_r2:Q', title='R²', scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('distribution:N', title='Noise Distribution'),
        strokeDash=alt.StrokeDash('model:N', title='Model'),
        tooltip=['sigma', 'mean_r2', 'model', 'distribution']
    ).mark_line().properties(
        width=600,
        height=400,
        title=f"{representation.upper()} — All Distributions"
    )
    return chart

# Load and combine individual CSV files
distribution_files = {
    'Gaussian': '../results/distributionGaussian.csv',
    'Left': '../results/distributionLeft.csv', 
    'Right': '../results/distributionRight.csv',
    'U': '../results/distributionU.csv',
    'Uniform': '../results/distributionUniform.csv'
}

# Load each file and add distribution column
dataframes = []
for dist_name, file_path in distribution_files.items():
    try:
        df = pd.read_csv(file_path)
        df['distribution'] = dist_name
        dataframes.append(df)
        print(f"Loaded {len(df)} rows from {file_path}")
    except FileNotFoundError:
        print(f"Warning: Could not find {file_path}")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

# Combine all dataframes
if dataframes:
    df_all = pd.concat(dataframes, ignore_index=True)
    print(f"Combined dataset has {len(df_all)} total rows")
    
    # Convert r2_score to numeric, filter invalid values
    df_all['r2_score'] = pd.to_numeric(df_all['r2_score'], errors='coerce')
    df_all = df_all[df_all['r2_score'].notna()]
    df_all = df_all[(df_all['r2_score'] >= -1.0) & (df_all['r2_score'] <= 1.0)]
    
    print(f"After filtering, dataset has {len(df_all)} rows")
    
    # Plot for each representation
    for rep in ['ecfp4', 'sns', 'smiles', 'pdv', 'graph']:
        chart = plot_r2_vs_sigma_with_distributions(df_all, rep)
        chart.save(f"lineplots_{rep}_distributions.html")
else:
    print("No data files could be loaded!")