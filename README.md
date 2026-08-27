# QSAR/QM Models Testing Framework

This repository contains a comprehensive framework for testing and evaluating QSAR (Quantitative Structure-Activity Relationship) models using pre-calculated quantum mechanical (QM) properties, experimental data, and molecular representations. It supports various machine learning architectures and noise conditions, enabling robust benchmarking of model performance and feature utility.

## Features

### Molecular Representations

The study's set, settled 2026-08-26, is six: **ECFP4** (Morgan radius 2, 2,048
bits, computed in Python and carried through the record), **PDV**
(`continuous_pdv`, 200 RDKit descriptors as float32), **Sort & Slice**
(`sns`, 1,024 substructure counts as uint16), **MHG-GNN**, **Avalon** and
**ChemBERTa** (`DeepChem/ChemBERTa-77M-MTR`, 384 wide).

Anything else is refused by name: one-hot SMILES, randomized SMILES and the
binary `pdv` still build but are not part of the study, and mol2vec is deleted.
Graph representations (GIN, GCN, GATv2, MPNN) exist for QM9 and are not in the
job generator's roster.

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

Every check below RUNS the code. None matches source text: a string match passes
whether or not the line it matched ever executes, and that is how a dead block
survived in this repo for two days.

Run them all:

```
for t in scripts/test_*.py; do python "$t" || echo "FAILED $t"; done
cd rust && cargo test --release
```

| check | what it guards |
|---|---|
| `scripts/test_qm9_split_alignment.py` | the graph models' molecules, labels and split composition |
| `scripts/test_ecfp4_identity.py` | `ecfp4` is a Morgan radius-2 fingerprint, on both pipelines |
| `scripts/test_figure_conditions.py` | the noise condition on a row, the level axis, the uncertainty column, sibling files |
| `scripts/test_result_row_condition.py` | the condition reaches the row; the manifest header and join |
| `scripts/test_no_shadowed_definitions.py` | no top-level definition in either pipeline is shadowed by a later one |
| `scripts/test_bnn_kl_term.py` | the Bayesian networks are fitted on the ELBO, not plain MSE |
| `scripts/test_spec_is_live.py` | changing a value in `models/model_defaults.py` changes what is built |
| `scripts/test_generated_job_flags.py` | every flag the job generator emits is one the program has |
| `scripts/test_uncertainty_writer.py` | the per-molecule uncertainty writer and both its call sites |
| `scripts/test_record_alignment.py` | the packed record cannot be silently misaligned |
| `scripts/check_fixes_fail_when_removed.py` | that each check above fails when its fix is removed |
| `rust/tests/noise_gates.rs` (28) | the injector: dose, shape, targeting, provenance |
| `rust/tests/writer_guards.rs` (5) | the record writer: length, alignment, featurisation failures |

The experimental pipeline's checks live in the KIRBy checkout, under
`tests/smoke/`: `smoke_kirby_uncertainty.py` (80 checks),
`smoke_kirby_splits.py`, `smoke_kirby_target_scaling.py`,
`smoke_kirby_merge.py`.

Several checks carry a CONTROL: the rule they replaced, run on the same data, so
the number the check would have taken under the old behaviour is printed beside
the one it takes now.

And the checks themselves are checked. `scripts/check_fixes_fail_when_removed.py`
breaks each fix in the real file, runs its check, and puts the file back. A check
that stays green with its fix removed guards nothing. Two did, the first time it
was run.

`scripts/test_run_qm_models.py` is the older baseline suite (RF/ECFP4, GB/ECFP4,
SVM/ECFP4 and so on, each asserting R-squared above 0.7). It predates the noise
redesign.

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

> **The four companion packages are optional, and installing the wrong build of them
> breaks `import torch_geometric` outright.**
>
> `torch-scatter`, `torch-sparse`, `torch-cluster` and `torch-spline-conv` ship compiled
> extensions linked against one specific libtorch. If they do not match the installed
> PyTorch, loading them raises `OSError: Symbol not found`. `torch_geometric.typing`
> catches that and disables them — but `nn/conv/gravnet_conv.py` catches only
> `ImportError`, so the `OSError` escapes and the whole package fails to import, taking
> `scripts/process_and_train.py` with it.
>
> Nothing in this project uses the operators they provide (`GCNConv`, `GINConv`,
> `GATv2Conv` and the pooling functions are all pure PyTorch), so `torch_geometric` alone
> is enough. If you hit that error:
>
> ```bash
> python -m pip uninstall -y torch-scatter torch-sparse torch-cluster torch-spline-conv
> python -c "import torch_geometric; print(torch_geometric.__version__)"
> ```
>
> `python scripts/check_environment.py` detects this exact condition and names the fix.
> Note that the wheel index below is pinned to **torch 2.5.1** — match it to your actual
> torch version, or skip the four packages.


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
