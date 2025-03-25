# QSAR/QM Models Testing Framework

This repository contains a comprehensive framework for testing and evaluating QSAR (Quantitative Structure-Activity Relationship) models using pre-calculated quantum mechanical (QM) properties, experimental data, and molecular representations. It supports various machine learning architectures and noise conditions, enabling robust benchmarking of model performance and feature utility.

## Features

### Molecular Representations
- **Bit Vector-Based Representations**: Includes ECFP4, Sort & Slice fingerprints, and One-Hot Encoded (OHE) SMILES.
- **Graph-Based Representations**: Supports GIN, GCN, GATv2, and MPNN architectures with advanced molecular embedding methods.
- **Alternative SMILES**: Generates canonical and alternative SMILES representations with customizable randomization.

### Machine Learning Models
- **Standard Models**: Random Forest (RF), Support Vector Machine (SVM), Gradient Boosting (GB), and Gaussian Processes (GP).
- **Graph-Based Neural Networks**: Includes GIN, GCN, and Co-Teaching methods.
- **Custom Architectures**: Supports Gauche GP implementation and various deep learning models.

### Noise Simulation
- Introduces controlled artificial noise (Gaussian, uniform, and other distributions) to simulate real-world variability.
- Supports domain-specific noise injection with clustering-based sampling.

### Efficiency
- Memory-safe processing with Rust integration for pre-processing and data manipulation.
- Multi-threaded execution using `concurrent.futures`.
- Support for memory-mapped (mmap) file storage to handle large datasets.

### Advanced Testing Capabilities
- Hyperparameter optimization with Bayesian Optimization.
- Bootstrapping for error bars and confidence intervals.
- Robust metric calculation for regression and classification tasks.

### Datasets
- **QM9**: Pre-calculated QM properties for small organic molecules.
- **PolarisHub Datasets**: Supports external datasets for classification tasks, such as binding affinity and toxicity prediction.

## Installation

### 1. Clone the Repository
git clone https://github.com/adpunt/qsar_qm_models.git
cd qsar_qm_models

### 2. Python Requirements
Install the necessary Python packages:
pip install numpy torch torch-geometric rdkit bayesian-optimization altair pandas scikit-learn xgboost catboost deepchem polaris

### 3. Rust Requirements
Ensure Rust is installed. Instructions are available at https://www.rust-lang.org/tools/install. Once installed, build the Rust processor:
cd rust_processor
cargo build --release

## Usage

### Running the Framework
The framework uses command-line arguments for configuration. Below are the available arguments:

#### Required Arguments
- `-q`, `--qm_property`: The QM property to predict (e.g., `homo_lumo_gap`, `alpha`).
- `-m`, `--models`: Models to use for prediction (e.g., `rf`, `svm`, `gb`).
- `-r`, `--molecular_representations`: Molecular representations to use (e.g., `smiles`, `ecfp4`).
- `-n`, `--sample-size`: Number of samples to use.

#### Optional Arguments
- `--random-seed`: Random seed for reproducibility (default: `42`).
- `-b`, `--bootstrapping`: Number of bootstrapping iterations (default: `1`).
- `--sampling-proportion`: Proportion of the dataset to which artificial noise will be added.
- `--noise`: Flag to generate artificial Gaussian noise (default: `False`).
- `--sigma`: Standard deviation(s) of artificially added Gaussian noise.
- `--distribution`: Distribution of artificial noise (default: `gaussian`).
- `-t`, `--hyperparameter-tuning`: Enable hyperparameter tuning (default: `False`).
- `-d`, `--dataset`: Dataset to run experiments on (`QM9` or PolarisHub datasets).
- `-s`, `--split`: Method for splitting data (default: `random`).

#### Example
python scripts/run_qm_qsar_models.py -q homo_lumo_gap -m rf svm -r ecfp4 smiles -n 10000 --noise True --sigma 1.0 --distribution gaussian

This command predicts the `homo_lumo_gap` property using RF and SVM models with ECFP4 and SMILES representations, introducing Gaussian noise with a standard deviation of 1.0.

### Warnings to Ignore
Warnings such as:
Explicit valence for atom # 5 C, 5, is greater than permitted.
are caused by invalid molecules in the QM9 dataset. These can be ignored.

## Testing

Run unit tests to ensure the framework is functioning correctly:
python -m unittest scripts/test_run_qm_models.py

This runs baseline tests for RF/ECFP4, GB/ECFP4, SVM/ECFP4, RF/SMILES, GB/SMILES, SVM/SMILES, and GIN/Graph. Each experiment checks for R-squared values greater than 0.7.

---

## Using Custom .pt Models in the Framework

### 1. How to Save a Model (.pt Format)

#### Using a Full Model Save (Recommended for Ease of Use)
torch.save(model, "my_model.pt")
- Pros: Simple, architecture is included.
- Cons: Larger file size, not as version-friendly.

#### Using state_dict Save (Recommended for Stability & Version Control)
torch.save({"state_dict": model.state_dict()}, "my_model.pt")
- Pros: Smaller file, easier to update.
- Cons: Requires redefining model architecture before loading.

---

### 2. Required Model Architecture for state_dict Loads
If using state_dict, your architecture must be defined before loading.
Modify get_predefined_model_class() in run_model.py to match your architecture.

def get_predefined_model_class():
    """Define your model architecture to match the saved state_dict."""
    class CustomModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(100, 50)
            self.fc2 = torch.nn.Linear(50, 1)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    return CustomModel

---

### 3. JSON File Format for Hyperparameters
If tuning is enabled, provide a JSON file containing hyperparameter bounds.

Example (my_model_metadata.json):

{
    "learning_rate": [0.0001, 0.01],
    "batch_size": [8, 64],
    "dropout": [0.1, 0.5]
}

- If no metadata file is provided, hyperparameter tuning is disabled.

---

### 4. Running run_model() with a .pt Model

Basic Usage (No Hyperparameter Tuning):

run_model(x_train, y_train, x_test, y_test, 
          model_type="custom", 
          molecular_representation="ecfp4", 
          hyperparameter_tuning=False, 
          bootstrapping=1, 
          sigma=None, 
          current_seed=42, 
          distribution="normal", 
          dataset="my_dataset", 
          featuriser="morgan",
          model_path="my_model.pt",
          metadata_path=None)

With Hyperparameter Tuning:

run_model(x_train, y_train, x_test, y_test, 
          model_type="custom", 
          molecular_representation="ecfp4", 
          hyperparameter_tuning=True, 
          bootstrapping=1, 
          sigma=None, 
          current_seed=42, 
          distribution="normal", 
          dataset="my_dataset", 
          featuriser="morgan",
          model_path="my_model.pt",
          metadata_path="my_model_metadata.json")

- If metadata_path is None, no tuning occurs.
- If metadata_path is provided, the specified hyperparameter ranges are used.


---

### Contact
For questions or issues, please open a GitHub issue or reach out to the repository owner.



### Installing PyTorch Geometric (macOS / Linux, CPU-only)

After activating your micromamba environment (e.g., `micromamba activate py_rust_env`), install the PyTorch Geometric dependencies using the official wheel index:

```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric \
  -f https://data.pyg.org/whl/torch-2.5.1+cpu.html
```

### Installing torch-geometric (CPU-only server, PyTorch 2.5.1)

After setting up your environment and installing PyTorch 2.5.1 (CPU-only), install the compatible PyTorch Geometric packages with:

```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric \
  -f https://data.pyg.org/whl/torch-2.5.1+cpu.html
```

If you're using a different version of PyTorch or need CUDA support, update the torch-2.5.1+cpu.html portion of the URL to match your version and CUDA setup. See: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
