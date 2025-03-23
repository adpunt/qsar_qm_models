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

def save_shap_values(shap_values, feature_names, x_test, filepath, model, iteration, rep):
    """
    Save SHAP values to a CSV file or NumPy file for large datasets.
    """
    shap_filepath = filepath.replace('.csv', '_shap.csv')  # Store SHAP values separately

    if shap_values is not None:
        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        shap_df.insert(0, "Sample_Index", np.arange(len(shap_values)))  # Track sample index
        shap_df.insert(1, "Model", model)
        shap_df.insert(2, "Iteration", iteration)
        shap_df.insert(3, "Rep", rep)

        # Save to CSV (appending if file exists)
        if os.path.exists(shap_filepath):
            shap_df.to_csv(shap_filepath, mode='a', header=False, index=False)
        else:
            shap_df.to_csv(shap_filepath, index=False)

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

    # Save as NumPy file for efficiency
    npy_filepath = filepath.replace('.csv', '_shap.npy')
    np.save(npy_filepath, shap_values)
