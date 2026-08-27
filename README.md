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
- **Standard Models**: Random Forest (`rf`), quantile forest (`qrf`), Support
  Vector Machine (`svm`, RBF kernel), XGBoost (`xgboost`), LightGBM (`lgb`),
  NGBoost (`ngboost`) and Gaussian processes (`gauche`, `gauche_rbf`).
- **Graph-Based Neural Networks**: Includes GIN, GCN, and Co-Teaching methods.
- **Custom Architectures**: Supports Gauche GP implementation and various deep learning models.

### Noise Simulation

Label noise is injected to simulate real assay error. The scheme was redesigned
on 2026-08-26 (`NOISE_DESIGN.md`) and the old one is gone, not deprecated.

A noise condition is a **shape** (how one draw is distributed) crossed with a
**targeting rule** (who gets hit and how hard):

- shapes: Gaussian, Student-t, Laplace
- targeting: uniform, grouped-wider, grouped-shifted, outlier, censoring

The **level** is the amount of noise actually **delivered** -- by default a
fraction of the clean training labels' standard deviation, or the label's own
units with `--dose-units label`. Every condition solves for whatever internal
scale hits the level it was asked for, so the same number means the same amount
of corruption under every condition. That was not true before: the six old
strategies were one strategy at six doses, delivering between 0.49x and 2.00x
the same amount at a common setting, and their whole apparent severity ordering
was that.

The settled condition set lives in `noise_conditions.json` and is read, never
restated, by both injectors and by every job generator.

⚠️ Held-out labels are never noised. Validation carries its own independently
drawn noise, dosed against the **clean training** spread rather than its own.

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
Ensure Rust is installed. Instructions are available at https://www.rust-lang.org/tools/install.
Once installed, build the processor (the crate lives in `rust/`, and the binary the pipeline
looks for is `rust/target/release/rust_processor`):

```bash
cd rust
cargo build --release
cargo test --release          # 28 noise gates + 5 writer guards
```

## Usage

### Running the Framework
The framework uses command-line arguments for configuration. Below are the available arguments:

The entry point is **`scripts/process_and_train.py`**.
`scripts/run_qm_qsar_models.py` is superseded and no longer works against the
current Rust binary.

#### Required Arguments
- `-m`, `--models`: model(s) to run, e.g. `rf svm lgb`.
- `-r`, `--molecular_representations`: representation(s), e.g. `ecfp4 continuous_pdv`.

#### Commonly used
- `-d`, `--dataset`: dataset (default `QM9`).
- `-t`, `--target`: property to predict (default `homo_lumo_gap`).
- `-n`, `--sample-size`: molecules per repetition (default `10000`).
- `-b`, `--repetitions`: repetitions, i.e. replicates (default `1`).
- `-s`, `--split`: split method (default **`scaffold`**; every experiment in the
  study uses scaffold splits).
- `--random-seed`: random seed (default `42`).
- `-f`, `--filepath`: where to write the results CSV.

#### Noise
- `--noise-level`: one or more levels. The amount **delivered**, not a knob
  (default `0.0`). For censoring it is the fraction of labels clipped instead.
- `--dose-units`: `spread` (a fraction of the clean training label standard
  deviation, the default) or `label` (the label's own units, e.g. log units on
  the experimental datasets).
- `--noise-shape`: `gaussian`, `student_t` or `laplace`.
- `--noise-targeting`: `uniform`, `grouped_wide`, `grouped_shift`, `outlier` or
  `censoring`. The two grouped ones also take the longer spellings
  `grouped_wider` and `grouped_shifted`, which is what every results row,
  manifest and figure calls them.
- Condition parameters: `--nu`, `--noise-lambda`, `--group-fraction`,
  `--group-variance-share`, `--outlier-p`, `--censor-side`. Their defaults are
  the settled values and are sourced in `NOISE_DESIGN.md`.

> **Retired flags are refused by name, not ignored.** `--sigma`,
> `--distribution`, `--noise-strategy` and `--strategy-params` all exit with a
> message naming the replacement. A job script written against the old scheme
> would otherwise run silently under the new one, where the level means
> something different.

#### Example

Run it **from `scripts/`** — that is what the job scripts do (`cd scripts` first),
and the default output path is relative to it.

```bash
cd scripts
python process_and_train.py \
    -d QM9 -t homo_lumo_gap \
    -m rf lgb -r ecfp4 continuous_pdv \
    -n 10000 -b 10 -s scaffold \
    --noise-level 0.0 0.5 1.0 --noise-shape gaussian --noise-targeting uniform \
    -f ../results/example.csv
```

Random forest and LightGBM on ECFP4 and PDV, ten replicates of 10,000 molecules
on a scaffold split, at three noise levels: none, half the clean training label
spread, and one whole spread.

#### Before submitting anything to a cluster
```bash
python scripts/check_environment.py --deep --validation
```
It names the interpreter, **constructs** each model rather than importing its
package, and runs the two fits that fail on contact in a bad environment. The
job scripts run a cheap version of it themselves -- `--models <label>` for the
QM9 family, `--validation-models <label>` for the validation and uncertainty
families, which use a different set of model names.

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
| `scripts/test_uncertainty_stats.py` | the uncertainty statistics, against data whose answer is known by construction |
| `scripts/test_record_alignment.py` | the packed record cannot be silently misaligned |
| `scripts/test_config_isolation.py` | two tasks running at once cannot corrupt each other's data |
| `scripts/test_failure_propagation.py` | a failure inside the noise injector stops the run |
| `scripts/test_avalon_failure.py` | Avalon refuses an unparseable molecule instead of returning zeros |
| `scripts/test_embedding_storage.py` | the learned embeddings are stored without damaging their geometry |
| `scripts/test_injector_wiring.py` | the Python noise wiring, on a machine with no training stack |
| `scripts/test_noise_conditions.py` | the settled condition set binds the Python injector too |
| `scripts/test_uncertainty_job_scripts.py` | every uncertainty job's command line, through the runner's own parser |
| `scripts/test_validation_job_scripts.py` | every validation job's command line, its conditions, its three guards |
| `scripts/test_condition_names.py` | one noise condition has one name on both injectors, against `condition_names.json` |
| `scripts/test_predictive_head.py` | the network's own predicted variance is read back as fitted, and reaches the file per molecule |
| `scripts/crosscheck_injectors.py` | the Rust and Python injectors deliver the same thing (342 checks) |
| `scripts/crosscheck_pipeline_reference.py` | the pipeline's injector against the reference implementation |
| `scripts/check_environment.py` | this interpreter can build every model the job asks for |
| `scripts/check_bib_and_docs.py` | the bibliography resolves; the two design documents have not re-drifted |
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

**If it is killed, it leaves a broken file behind.** Stopping the run from
outside skips the step that puts the file back, and nothing says so — on
2026-08-27 that left `utils.py` with a `save_results` that raises and `models.py`
with a fix undone, both live on disk and neither committed. It now copies the
original out before touching anything, and a copy still sitting in
`scripts/.harness_unrestored/` at start-up means the last run was killed: the
file is put back and the run refuses to start until you have looked at
`git diff`.

Two scripts in `scripts/` match `test_*.py` but are **not** part of this suite,
and the loop above will run them:

- `scripts/test_noise_arms.py` — design exploration from 2026-08-24, written
  against five *proposed* noise conditions before the set was settled, and using
  vocabulary the project has since dropped. It is evidence behind
  `NOISE_DESIGN.md`, not a guard. It passes, and it needs the raw QM9 CSV
  (`data/QM9/raw/gdb9.sdf.csv`), so it fails on a checkout without the data.
- `scripts/test_hybrid.py` — a manual check for the hybrid-representation
  feature, which is not part of the study's roster.

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
