import pandas as pd
import altair as alt
import os

# scp scat9264@gateway.arc.ox.ac.uk:/data/stat-cadd/scat9264/qsar_qm_models/results/uncertainty_full_uncertainty.csv ../results
# scp scat9264@gateway.arc.ox.ac.uk:/data/stat-cadd/scat9264/qsar_qm_models/results/uncertainty_last_layer_uncertainty.csv ../results
# scp scat9264@gateway.arc.ox.ac.uk:/data/stat-cadd/scat9264/qsar_qm_models/results/uncertainty_variational_uncertainty.csv ../results
# scp scat9264@gateway.arc.ox.ac.uk:/data/stat-cadd/scat9264/qsar_qm_models/results/uncertainty_ngboost_uncertainty.csv ../results
# scp scat9264@gateway.arc.ox.ac.uk:/data/stat-cadd/scat9264/qsar_qm_models/results/uncertainty_qrf_uncertainty.csv ../results
# scp scat9264@gateway.arc.ox.ac.uk:/data/stat-cadd/scat9264/qsar_qm_models/results/uncertainty_gauche_uncertainty.csv ../results
# scp scat9264@gateway.arc.ox.ac.uk:/data/stat-cadd/scat9264/qsar_qm_models/results/uncertainty_graph_gp_uncertainty.csv ../results

def plot_uncertainty_analysis(input_filename):
    # Step 1: Load CSV
    df = pd.read_csv(input_filename)

    # Step 2: Compute Squared Error
    df['squared_error'] = (df['y_pred_mean'] - df['y_true'])**2
    
    # Create proper model and representation names
    model_names = {
        'dnn': 'DNN',
        'gauche': 'Gaussian Process',
        'graph_gp': 'Gaussian Process (Graph)',
        'qrf': 'QRF',
        'full': 'DNN with Full Bayesian Transformation',
        'last_layer': 'DNN with Last Layer Bayesian Transformation',
        'variational': 'DNN with Variational Inference',
        'ngboost': 'NGBoost'
    }
    
    rep_names = {
        'smiles': 'SMILES',
        'ecfp4': 'ECFP4',
        'sns': 'SNS',
        'pdv': 'PDV'
    }
    
    # Only apply mappings to models and reps that actually exist in this dataset
    unique_models = df['Model'].unique()
    unique_reps = df['Rep'].unique()
    
    df['Model_Clean'] = df['Model'].apply(lambda x: model_names.get(x, x) if x in unique_models else x)
    df['Rep_Clean'] = df['Rep'].apply(lambda x: rep_names.get(x, x) if x in unique_reps else x)
    df['Model_Rep'] = df['Model_Clean'] + ' - ' + df['Rep_Clean']

    # Chart 1: Mean Predictive Uncertainty vs. Noise Level
    mean_uncertainty = df.groupby(['Model', 'Rep', 'Sigma', 'Model_Rep', 'Model_Clean', 'Rep_Clean'])['y_pred_std'].mean().reset_index()

    chart1 = alt.Chart(mean_uncertainty).mark_line(point=True).encode(
        x=alt.X('Sigma:Q', title='Noise Level (σ)'),
        y=alt.Y('y_pred_std:Q', title='Mean Predictive Uncertainty (σŷ)'),
        color=alt.Color('Model_Rep:N', scale=alt.Scale(scheme='category20'), title='Model - Representation')
    ).properties(
        title='Mean Predictive Uncertainty vs. Noise Level',
        width=300,
        height=300
    )

    # Chart 2: Mean Predictive Uncertainty vs. Squared Error
    chart2 = alt.Chart(df).mark_circle(size=60, opacity=0.5).encode(
        x=alt.X('y_pred_std:Q', title='Predictive Uncertainty (σŷ)'),
        y=alt.Y('squared_error:Q', title='Squared Error'),
        color=alt.Color('Model_Rep:N', scale=alt.Scale(scheme='category20'), title='Model - Representation')
    ).properties(
        title='Predictive Uncertainty vs. Squared Error',
        width=300,
    )

    # Chart 3: Calibration Curve
    num_bins = 10
    df['uncertainty_bin'] = pd.qcut(df['y_pred_std'], q=num_bins, duplicates='drop').astype(str)
    df = df.dropna(subset=['uncertainty_bin'])

    calib = df.groupby(['Model', 'Rep', 'uncertainty_bin', 'Model_Rep', 'Model_Clean', 'Rep_Clean']).agg({
        'y_pred_std': 'mean',
        'squared_error': 'mean'
    }).reset_index()

    calib['predicted_variance'] = calib['y_pred_std']**2

    chart3 = alt.Chart(calib).mark_line(point=True).encode(
        x=alt.X('predicted_variance:Q', title='Mean Predicted Variance (σŷ²)'),
        y=alt.Y('squared_error:Q', title='Mean Squared Error (MSE)'),
        color=alt.Color('Model_Rep:N', scale=alt.Scale(scheme='category20'), title='Model - Representation')
    ).properties(
        title='Calibration Curve: Predicted Variance vs. MSE',
        width=300,
        height=300
    )

    # Chart 4: Slope of σŷ vs. σ
    slopes = []
    for (model, rep), group in mean_uncertainty.groupby(['Model', 'Rep']):
        if len(group) > 1:
            slope = (
                (group['y_pred_std'].iloc[-1] - group['y_pred_std'].iloc[0]) /
                (group['Sigma'].iloc[-1] - group['Sigma'].iloc[0])
            )
            slopes.append({
                'Model': model,
                'Rep': rep,
                'Slope': slope,
                'Model_Clean': model_names.get(model, model) if model in unique_models else model,
                'Rep_Clean': rep_names.get(rep, rep) if rep in unique_reps else rep
            })


    slopes_df = pd.DataFrame(slopes)

    chart4 = alt.Chart(slopes_df).mark_bar().encode(
        x=alt.X('Rep_Clean:N', title='Representation'),
        y=alt.Y('Slope:Q', title='Slope of σŷ vs. σ'),
        color=alt.Color('Model_Clean:N', title='Model'),
        column=alt.Column('Model_Clean:N', title='Model')
    ).properties(
        title='Slope of Uncertainty vs. Noise',
        width=150,
        height=300
    )

    # Combine
    final_chart = (chart1 | chart2) & (chart3 | chart4)
    final_chart = final_chart.resolve_scale(color='independent').properties(
        title=alt.TitleParams(
            text=[f"Uncertainty Analysis: {os.path.basename(input_filename).replace('uncertainty_', '').replace('_uncertainty.csv', '').replace('_', ' ').title()}"],
            fontSize=16,
            anchor='start'
        )
    )
    basename = os.path.basename(input_filename).replace('.csv', '')
    output_path = os.path.join(os.path.dirname(input_filename), basename + '_analysis.html')
    print(f"Saving to: {output_path}")
    final_chart.save(output_path)
    final_chart.save(output_path)
    print(f"Saved: {output_path}")

plot_uncertainty_analysis("../results/uncertainty_full_uncertainty.csv")
plot_uncertainty_analysis("../results/uncertainty_last_layer_uncertainty.csv")
plot_uncertainty_analysis("../results/uncertainty_variational_uncertainty.csv")
plot_uncertainty_analysis("../results/uncertainty_ngboost_uncertainty.csv")
plot_uncertainty_analysis("../results/uncertainty_qrf_uncertainty.csv")
plot_uncertainty_analysis("../results/uncertainty_gauche_uncertainty.csv")
plot_uncertainty_analysis("../results/uncertainty_graph_gp_uncertainty.csv")