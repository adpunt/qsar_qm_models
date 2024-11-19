# QSAR QM Models Testing Framework

This repository contains a testing framework for evaluating the accuracy and robustness of QSAR (Quantitative Structure-Activity Relationship) models. Our framework uses pre-calculated quantum mechanical (QM) properties and experimental data to benchmark various molecular representations and machine learning models under different noise conditions.

## Features

- **Molecular Representations**: Supports bit vector-based representations like ECFP4, OHE SMILES, Sort & Slice fingerprints, and molecular graphs for GCN, GIN, and GATv2.
- **Machine Learning Models**: Includes Random Forest (RF), Support Vector Machine (SVM), XGBoost, Gaussian Process (GP), and various graph-based neural networks.
- **Noise Simulation**: Introduces controlled artificial noise (Gaussian, left-tailed, right-tailed, U-shaped, and uniform) to mimic real-world conditions.
- **Efficiency**: Memory-safe processing using Rust, threading, and memory-mapped files.
- **Advanced Testing Capabilities**: Hyperparameter tuning with Bayesian optimization, bootstrapping for error bars, and plotting with Vega-Altair.
- **Primary QM Property**: Focuses on predicting HOMO and LUMO gaps.

## Installation

1. **Rust**: Ensure Rust is installed. Instructions can be found [here](https://www.rust-lang.org/tools/install).
2. **Clone the Repository**:
    ```sh
    git clone https://github.com/adpunt/qsar_qm_models
    cd qsar_qm_models
    ```
3. Install the required Python packages:
	```sh
	pip install numpy torch torch-geometric rdkit bayesian-optimization altair pandas scikit-learn xgboost catboost
	```

## Usage 

### Arguments

The framework uses command-line arguments for configuration. Below are the details of each argument:

### Required Arguments
- `-q`, `--qm_property` (str): The QM property to predict (e.g., HOMO, LUMO).
- `-m`, `--models` (list of str): Models to use for prediction (e.g., rf, svm, xgboost).
- `-r`, `--molecular_representations` (list of str): Molecular representations to use (e.g., smiles, ecfp4).
- `-n`, `--sample-size` (int): Number of samples to use.

### Optional Arguments
- `--random-seed` (int): Random seed for reproducibility (default: 42).
- `-b`, `--bootstrapping` (int): Number of bootstrapping iterations for generating error bars (default: 1).
- `--use-extra` (bool): Use information from the entire dataset rather than just the training set (default: False).
- `--sampling-proportion` (list of float): Proportion of the dataset to which artificial noise will be added (default: None).
- `--noise` (bool): Flag to generate artificial Gaussian noise (default: False).
- `--sigma` (list of float): Standard deviation(s) of artificially added Gaussian noise (default: None).
- `--distribution` (str): Distribution of artificial noise (default: Gaussian).
- `-t`, `--hyperparameter-tuning` (bool): Flag for hyperparameter tuning (default: False).
- `--pair-comparison` (bool): Flag to create a plot comparing the noise response of different model/representation pairs (default: False).
- `--raw-filename` (str): Filename for saving raw data (default: ../results/raw_data.pkl).
- `--pred-tracking` (bool): Flag to plot the top misclassified samples and track predictions (default: False).
- `-a`, `--alternative-smiles` (int): Generate alternative SMILES for the same molecules in the fixed test set (default: 0, mixed train is 1, alternative train is 2, alternative test is 3).

### Example

Run the program with specific options:
```bash
python ./noise_detection/run_qm_models.py -e 30 -q homo_lumo_gap -m rf svm -r ecfp4 smiles --batch-size 32 --hyperparameter_tuning True --random-seed 123
```

This command predicts the `homo_lumo_gap` property using `rf` and `svm` models with `ecfp4` and `smiles` representations, with 30 epochs, a batch size of 32, hyperparameter tuning enabled, and a random seed of 123.

### Warnings to Ignore

- **Warnings**: On first run, you may encounter warnings such as:
	```sh
	Explicit valence for atom # 5 C, 5, is greater than permitted.
	```
	These are due to some invalid molecules in the QM9 dataset and can be ignored as they are relatively few.

### Testing

Run a unit test to make sure the baseline functionality is working:
```bash
python -m unittest ./noise_detection/test_run_qm_models.py
```

This command runs a unit testing, which runs all the baseline pairs (RF/ECFP4, GB/ECFP4, SVM/ECFP4, RF/SMILES, GB/SMILES, SVM/SMILES, GIN/Graph) with N=30,000 and default settings and ensures that for every experiment, the experiment successfully runs and the R-squared value is greater than 0.7.