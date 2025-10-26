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

def save_results(filepath, s, iteration, model, rep, n, metrics, params_source='default'):
    """
    Save results to a CSV file with hyperparameter source tracking
    """
    if filepath:
        file_exists = os.path.isfile(filepath)

        with open(filepath, mode='a', newline='') as f:
            writer = csv.writer(f)
            
            # Write header if the file is new
            if not file_exists:
                writer.writerow(["sigma", "iteration", "model", "rep", "sample_size", "mae", "mse", "rmse", "r2", "pearson_corr", "params_source"])
            
            # Save the results
            writer.writerow([s, iteration, model, rep, n, metrics[0], metrics[1], metrics[2], metrics[3], metrics[4], params_source])

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

def save_uncertainty_values(y_pred_mean, y_pred_std, y_true_original, y_true_noisy, filepath, model_name, rep, sigma_noise, iteration, file_no):
    # Create uncertainty DataFrame
    uncertainty_df = pd.DataFrame({
        "Sample_Index": np.arange(len(y_pred_mean)),
        "Model": model_name,
        "Rep": rep,
        "Sigma": sigma_noise,
        "Iteration": iteration,
        "File_No": file_no,
        "y_true_original": y_true_original, 
        "y_true_noisy": y_true_noisy,
        "y_pred_mean": y_pred_mean,
        "y_pred_std": y_pred_std,
    })
    
    # Create uncertainty directory and save
    if filepath:
        uncertainty_dir = os.path.dirname(filepath.replace('.csv', '_uncertainty/'))
        os.makedirs(uncertainty_dir, exist_ok=True)
        
        uncertainty_file = os.path.join(
            uncertainty_dir, 
            f"uncertainty_{model_name}_{rep}_sigma{sigma_noise}_iter{iteration}_file{file_no}.csv"
        )
        uncertainty_df.to_csv(uncertainty_file, index=False)
        print(f"Saved uncertainty values to {uncertainty_file}")

def save_conformal_intervals(y_pred, y_lower, y_upper, y_true, filepath, model_name, rep, sigma_noise, iteration, file_no, alpha):
    # Create directory if it doesn't exist
    intervals_dir = os.path.join(os.path.dirname(filepath), "conformal_intervals")
    os.makedirs(intervals_dir, exist_ok=True)
    
    # Create dataframe with results
    results_df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'y_lower': y_lower,
        'y_upper': y_upper,
        'interval_width': y_upper - y_lower,
        'coverage': ((y_true >= y_lower) & (y_true <= y_upper)).astype(int),
        'alpha': alpha,
        'model_name': model_name,
        'rep': rep,
        'sigma_noise': sigma_noise,
        'iteration': iteration,
        'file_no': file_no
    })
    
    # Calculate empirical coverage
    empirical_coverage = results_df['coverage'].mean()
    results_df['empirical_coverage'] = empirical_coverage
    
    # Save to file following your naming convention
    filename = f"conformal_intervals_{model_name}_{rep}_sigma{sigma_noise}_iter{iteration}_file{file_no}.csv"
    filepath_full = os.path.join(intervals_dir, filename)
    results_df.to_csv(filepath_full, index=False)
    
    print(f"Conformal intervals saved. Target coverage: {1-alpha:.1%}, Empirical coverage: {empirical_coverage:.1%}")

def save_per_epoch_metrics(train_losses, val_losses, filepath, model_name, rep, sigma_noise, iteration, file_no):
   """
   Save per-epoch training and validation metrics to a CSV file.
   
   Parameters
   ----------
   train_losses : list or np.array
       Training losses for each epoch
   val_losses : list or np.array  
       Validation losses for each epoch
   filepath : str
       Path to save the CSV file (should end with _per_epoch.csv)
   model_name : str
       Name of the model
   rep : str
       Representation type (e.g., 'graph', 'fingerprint', etc.)
   sigma_noise : float
       Noise level applied to the data
   iteration : int
       Current iteration/repetition number
   file_no : int
       File number identifier
   """
   if filepath:
       # Create per-epoch specific filepath
       base_path = filepath.replace('.csv', '_per_epoch.csv')
       file_exists = os.path.isfile(base_path)
       
       with open(base_path, mode='a', newline='') as f:
           writer = csv.writer(f)
           
           # Write header if the file is new
           if not file_exists:
               writer.writerow(["sigma", "iteration", "model", "rep", "file_no", "epoch", "train_loss", "val_loss"])
           
           # Save metrics for each epoch
           for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
               writer.writerow([sigma_noise, iteration, model_name, rep, file_no, epoch, train_loss, val_loss])

