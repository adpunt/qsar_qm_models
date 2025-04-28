from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)
from scipy.stats import pearsonr
import numpy as np
import os
import csv
import shap
import pandas as pd

# TOOD: this doesn't work with non-gaussian distributions at the moment, nor do a lot of things that rely solely on sigma
def save_results(filepath, s, iteration, model, rep, n, r2, mae, corr):
    """
    Save results to a CSV file specified by args.filepath.
    """
    if filepath:
        file_exists = os.path.isfile(filepath)

        with open(filepath, mode='a', newline='') as f:
            writer = csv.writer(f)
            
            # Write header if the file is new
            if not file_exists:
                writer.writerow(["sigma", "iteration", "model", "rep", "sample_size", "r2_score", "mae", "pearson_corr"])
            
            # Save the results
            writer.writerow([s, iteration, model, rep, n, r2, mae, corr])

def calculate_classification_metrics(y_test, prediction, logging=False):
    accuracy = accuracy_score(y_test, y_test_preds)
    roc_auc = roc_auc_score(y_test, y_test_probs[:, 1])  # Assuming binary classification
    precision = precision_score(y_test, y_test_preds, average="weighted")
    recall = recall_score(y_test, y_test_preds, average="weighted")
    f1 = f1_score(y_test, y_test_preds, average="weighted")
    pr_auc = average_precision_score(
        y_test, y_test_probs[:, 1], average="weighted"
    )
    # TODO: pearson?

    # Optionally log the metrics
    if logging:
        print("Accuracy:", accuracy)
        print("ROC AUC:", roc_auc)
        print("Precision", precision)
        print("Recall:", recall)
        print("F1:", f1)
        print("PR AUC:", pr_auc)

    return [accuracy, roc_auc, precision, recall, f1, pr_auc]

# TODO: add kendall's tau
def calculate_regression_metrics(y_test, prediction, logging=False):
    mae = mean_absolute_error(y_test, prediction)
    mse = mean_squared_error(y_test, prediction)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, prediction)
    pearson_corr, _ = pearsonr(y_test, prediction)

    # Optionally log the metrics
    if logging:
        print("Mean Absolute Error:", mae)
        print("Mean Squared Error:", mse)
        print("RMSE:", rmse)
        print("R-squared:", r2)
        print("Pearson Correlation:", pearson_corr)

    return mae, mse, rmse, r2, pearson_corr

def save_uncertainty_values(y_pred_mean, y_pred_std, y_true, filepath, model_name, rep, sigma_noise, iteration):
    """
    Save uncertainty values (predictive mean, predictive std, true y) to disk for later analysis.

    Parameters
    ----------
    y_pred_mean : np.ndarray
        Predicted mean outputs.
    y_pred_std : np.ndarray
        Predicted standard deviations.
    y_true : np.ndarray
        True labels.
    filepath : str
        Base path where the results should be saved (typically ends with '.csv').
    model_name : str
        Name of the model (e.g., 'gauche', 'bnn').
    rep : str
        Representation (e.g., 'ecfp', 'smiles').
    sigma_noise : float
        Noise level used for training.
    iteration : int
        Iteration number (e.g., bootstrap or trial iteration).
    """
    uncertainty_filepath = filepath.replace('.csv', '_uncertainty.csv')

    # Build the DataFrame
    df = pd.DataFrame({
        "Sample_Index": np.arange(len(y_pred_mean)),
        "Model": model_name,
        "Rep": rep,
        "Sigma": sigma_noise,
        "Iteration": iteration,
        "y_pred_mean": y_pred_mean,
        "y_pred_std": y_pred_std,
        "y_true": y_true
    })

    # Save as CSV (append if exists)
    if os.path.exists(uncertainty_filepath):
        df.to_csv(uncertainty_filepath, mode='a', header=False, index=False)
    else:
        df.to_csv(uncertainty_filepath, index=False)

    # Save raw arrays separately as .npy if you want
    np.save(f"y_pred_mean_{model_name}_{rep}_sigma{sigma_noise}_iter{iteration}.npy", y_pred_mean)
    np.save(f"y_pred_std_{model_name}_{rep}_sigma{sigma_noise}_iter{iteration}.npy", y_pred_std)
    np.save(f"y_true_{model_name}_{rep}_sigma{sigma_noise}_iter{iteration}.npy", y_true)
