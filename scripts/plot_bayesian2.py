
import pandas as pd
import altair as alt
import numpy as np
from scipy import stats

def prepare_summary(df, label, model_name):
    """
    Prepare summary statistics for plotting, with outlier handling
    """
    print(f"Processing {label} with model {model_name}")
    
    if model_name and 'model' in df.columns:
        df = df[df['model'] == model_name]
    
    if df.empty:
        print(f"Warning: No data after filtering for {label}")
        return pd.DataFrame()
    
    # Handle outliers in R² values
    print(f"R² range before outlier handling: {df['r2_score'].min():.3f} to {df['r2_score'].max():.3f}")
    
    # Cap extremely negative R² values (anything below -10 is likely a numerical issue)
    df = df.copy()
    df.loc[df['r2_score'] < -10, 'r2_score'] = -10
    
    # Also cap extremely high positive values (R² shouldn't be much above 1)
    df.loc[df['r2_score'] > 2, 'r2_score'] = 2
    
    print(f"R² range after outlier handling: {df['r2_score'].min():.3f} to {df['r2_score'].max():.3f}")
    
    # Group by sigma and calculate statistics
    grouped = df.groupby(['sigma']).agg({'r2_score': ['mean', 'std', 'count']}).reset_index()
    grouped.columns = ['sigma', 'mean_r2', 'std_r2', 'count']
    grouped['type'] = label
    grouped['model'] = model_name
    
    print(f"Final mean R² range: {grouped['mean_r2'].min():.3f} to {grouped['mean_r2'].max():.3f}")
    
    return grouped[['sigma', 'mean_r2', 'std_r2', 'count', 'model', 'type']]

def plot_bayesian_variants(df_list, labels, model_name):
    """
    Plots R² vs Sigma for multiple Bayesian model variants.
    Simplified to avoid Altair's rendering issues with complex charts.
    """
    print(f"\n=== Plotting Bayesian variants for model: {model_name} ===")
    
    summaries = []
    for df, label in zip(df_list, labels):
        if df is not None and not df.empty:
            summary = prepare_summary(df, label, model_name)
            if not summary.empty:
                summaries.append(summary)
    
    if not summaries:
        print("Error: No valid data to plot!")
        return None
    
    df_plot = pd.concat(summaries, ignore_index=True)
    print(f"\nFinal plotting data:")
    print(df_plot[['sigma', 'mean_r2', 'type']].round(3))
    
    # Simple, robust Altair chart
    chart = alt.Chart(df_plot).mark_line(
        point=True,
        strokeWidth=3
    ).encode(
        x=alt.X('sigma:Q', 
               title='Sigma (Noise Level)'),
        y=alt.Y('mean_r2:Q', 
               title='R² Score'),
        color=alt.Color('type:N', 
                      title='Model Variant'),
        tooltip=['sigma:Q', 'mean_r2:Q', 'type:N']
    ).properties(
        width=600,
        height=400,
        title=f"Bayesian Model Variants - {model_name.upper()}"
    ).resolve_scale(
        y='independent'  # This helps with rendering
    )
    
    return chart

# Load datasets
df_baseline = pd.read_csv("../results/bayesianBaselineGraph.csv")
df_full = pd.read_csv("../results/bayesianFullGraph.csv")
df_last_layer = pd.read_csv("../results/bayesianLastLayerGraph.csv")
df_variational = pd.read_csv("../results/bayesianVariationalGraph.csv")

dfs = [df_baseline, df_full, df_last_layer, df_variational]
labels = ['Baseline', 'Full Bayesian', 'Bayesian Last Layer', 'Variational']

# Create chart
chart = plot_bayesian_variants(dfs, labels, model_name="gin")

if chart is not None:
    # Save as JSON first (more reliable than HTML with Altair)
    chart.save('../results/bayesian_variants_gin.json')
    print("Chart saved as JSON")
    
    # Also try HTML
    try:
        chart.save('../results/bayesian_variants_gin.html')
        print("Chart saved as HTML")
    except Exception as e:
        print(f"HTML save failed: {e}")
        
    # Display the chart data for manual verification
    print("\n=== Chart should show these trends ===")
    df_display = chart.data
    for variant in df_display['type'].unique():
        variant_data = df_display[df_display['type'] == variant]
        print(f"\n{variant}:")
        for _, row in variant_data.iterrows():
            print(f"  Sigma {row['sigma']}: R² = {row['mean_r2']:.3f}")

# Show the chart (Altair only)
try:
    chart.show()
    print("Chart displayed successfully")
except Exception as e:
    print(f"Chart display failed: {e}")
    print("But chart was saved to files")