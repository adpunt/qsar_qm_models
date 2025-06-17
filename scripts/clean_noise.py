import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import numpy as np
from torch_geometric.datasets import QM9
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, AllChem
from sklearn.neighbors import NearestNeighbors
from itertools import product
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import MLE
from quantile_forest import RandomForestQuantileRegressor
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import regex as re

sys.path.append('../models/')
sys.path.append('../results/')

# -- Reuse from your code --
from models import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

properties = {
    'homo_lumo_gap': 4, 'alpha': 1, 'G': 10, 'H': 9, 'U': 8,
    'G_a': 15, 'H_a': 14, 'U_a': 13, 'mu': 0, 'A': 16, 'B': 17, 'C': 18
}

model_types = ['dnn']
bayesian_transforms = ['full', 'last_layer', 'variational']
noise_identifications = ['std', 'residual', 'mc_percentile']
label_smoothings = ['hard_replacement_neighbor', 'hard_replacement_model', 'soft_smoothing_neighbor', 'soft_smoothing_model', 'removal']

# Create the grid
experiment_grid = []

class QRFWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Draw from quantiles with some noise to mimic stochasticity
        q16, q50, q84 = self.model.predict(x, quantiles=[0.16, 0.5, 0.84]).T
        noise = np.random.normal(loc=0.0, scale=0.5 * (q84 - q16))
        return torch.tensor(q50 + noise, dtype=torch.float32).unsqueeze(-1)


class NGBoostWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.dist = None

    def forward(self, x):
        if self.dist is None:
            self.dist = self.model.pred_dist(x)
        mean = self.dist.loc
        std = self.dist.scale
        noise = np.random.normal(loc=0.0, scale=std)
        return torch.tensor(mean + noise, dtype=torch.float32).unsqueeze(-1)


class SmilesTokenizer(object):
    """
    A simple regex-based tokenizer adapted from the deepchem smiles_tokenizer package.
    SMILES regex pattern for the tokenization is designed by Schwaller et. al., ACS Cent. Sci 5 (2019)
    """

    def __init__(self):
        self.regex_pattern = (
            r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\."
            r"|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
        )
        self.regex = re.compile(self.regex_pattern)

    def tokenize(self, smiles):
        """
        Tokenizes SMILES string.

        Parameters
        ----------
        smiles : str
            Input SMILES string.

        Returns
        -------
        List[str]
            A list of tokens.
        """
        tokens = [token for token in self.regex.findall(smiles)]
        return tokens

def build_vocab(smiles_list, tokenizer, max_vocab_size):
    """
    Builds a vocabulary of N=max_vocab_size most common tokens from list of SMILES strings.

    Parameters
    ----------
    smiles_list : List[str]
        List of SMILES strings.
    tokenizer : SmilesTokenizer
    max_vocab_size : int
        Maximum size of vocabulary.

    Returns
    -------
    Dict[str, int]
        A dictionary that defines mapping of a token to its index in the vocabulary.
    """
    tokenized_smiles = [tokenizer.tokenize(s) for s in smiles_list]
    token_counter = Counter(c for s in tokenized_smiles for c in s)
    tokens = [token for token, _ in token_counter.most_common(max_vocab_size)]
    vocab = {token: idx for idx, token in enumerate(tokens)}
    return vocab


def smiles_to_ohe(smiles, tokenizer, vocab):
    """
    Transforms SMILES string to one-hot encoding representation.

    Parameters
    ----------
    smiles : str
        Input SMILES string.
    tokenizer : SmilesTokenizer
    vocab : Dict[str, int]
        A dictionary that defines mapping of a token to its index in the vocabulary.

    Returns
    -------
    Tensor
        A pytorch Tensor with shape (n_tokens, vocab_size), where n_tokens is the
        length of tokenized input string, vocab_size is the number of tokens in
        the vocabulary
    """
    unknown_token_id = len(vocab) - 1
    token_ids = [vocab.get(token, unknown_token_id) for token in tokenizer.tokenize(smiles)]
    ohe = torch.eye(len(vocab))[token_ids]
    return ohe

for model_type, bayesian_transform, noise_identification, label_smoothing in product(
    model_types, bayesian_transforms, noise_identifications, label_smoothings
):
    config = {
        "model_type": model_type,
        "bayesian_transform": bayesian_transform,
        "noise_identification": noise_identification,
        "label_smoothing": label_smoothing,
    }
    experiment_grid.append(config)

def load_qm9_data(target="homo_lumo_gap", sample_size=500):
    qm9 = QM9(root=os.path.join("..", "data", "QM9"))
    valid_indices = torch.load(os.path.join("..", "data", "valid_qm9_indices.pth"))
    qm9 = qm9.index_select(valid_indices)

    # Isolate the target property
    y_target = pd.DataFrame(qm9.data.y.numpy())
    property_idx = properties[target]
    y = torch.Tensor(y_target[property_idx])

    # Now sample
    if sample_size < len(qm9):
        perm = torch.randperm(len(qm9))[:sample_size]
        qm9 = qm9.index_select(perm)
        y = y[perm]  # <<<<< SUBSET y ACCORDING TO perm ALSO

    smiles_list = [data.smiles for data in qm9]
    return smiles_list, y

def smiles_to_ecfp4(smiles_list, y, n_bits=2048):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    fps = []
    valid_indices = []

    for idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue  # skip invalid SMILES
        fp = generator.GetFingerprint(mol)
        arr = np.zeros((n_bits,), dtype=int)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
        valid_indices.append(idx)

    # Subset y to match valid smiles
    y = y[valid_indices]

    return np.array(fps), y

def add_regression_noise(y, sigma=0.1):
    noise = torch.normal(mean=0, std=sigma, size=y.shape)
    y_noisy = y + noise
    return y_noisy

def train_and_evaluate_rf(x_train, y_train, x_test, y_test):
    rf = RandomForestRegressor()
    rf.fit(x_train, y_train)
    preds = rf.predict(x_test)
    return r2_score(y_test, preds)

def run_experiment(config, x_train, y_train_noisy, x_test, y_test, sigma):
    model_type = config["model_type"]
    bayesian_transform = config["bayesian_transform"]
    noise_identification = config["noise_identification"]
    label_smoothing = config["label_smoothing"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Build model
    if model_type == "dnn":
        model = DNNRegressionModel(input_size=x_train.shape[1], hidden_size1=128, hidden_size2=64)
        
        # Apply Bayesian transform
        if bayesian_transform == "last_layer":
            model = apply_bayesian_transformation_last_layer(model)
        elif bayesian_transform == "full":
            model = apply_bayesian_transformation(model)
        elif bayesian_transform == "variational":
            model = apply_bayesian_transformation_last_layer_variational(model)
        # else no transformation if bayesian_transform == "none"
        
        model.to(device)

        # Train model
        model.train()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        train_loader = TorchDataLoader(
            TensorDataset(torch.tensor(x_train, dtype=torch.float32).to(device), 
                          torch.tensor(y_train_noisy, dtype=torch.float32).view(-1, 1).to(device)),
            batch_size=64,
            shuffle=True
        )
        
        for epoch in range(50):  # adjustable epochs
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
        
    # elif model_type == "graph_gp":
    #     from polaris.kernel import WeisfeilerLehmanKernel, VertexHistogramKernel, EdgeHistogramKernel, NeighborhoodHashKernel
    #     from polaris.model import GraphGP
    #     from polaris.data import NonTensorialInputs
    #     import gpytorch
    #     import networkx as nx

    #     # Step 1: Convert x_train (bit vectors) into trivial graphs
    #     def bitvector_to_graph(bitvec):
    #         G = nx.Graph()
    #         for idx, bit in enumerate(bitvec):
    #             if bit == 1:
    #                 G.add_node(idx, label=1)
    #         return G

    #     train_graphs = [bitvector_to_graph(vec) for vec in x_train]
    #     test_graphs = [bitvector_to_graph(vec) for vec in x_test]

    #     X_train = NonTensorialInputs(train_graphs)
    #     X_test = NonTensorialInputs(test_graphs)

    #     y_train_flat = y_train_noisy.flatten().float()
    #     y_test_flat = y_test.flatten().float()

    #     likelihood = gpytorch.likelihoods.GaussianLikelihood(noise=1e-3)
    #     kernel = WeisfeilerLehmanKernel(node_label='label')

    #     model = GraphGP(X_train, y_train_flat, likelihood, kernel)

    #     mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    #     model.train()
    #     likelihood.train()
    #     optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    #     training_iter = 50
    #     for i in range(training_iter):
    #         optimizer.zero_grad()
    #         output = model(X_train)
    #         loss = -mll(output, y_train_flat)
    #         loss.backward()
    #         optimizer.step()

    #     model.eval()
    #     likelihood.eval()
        
    #     with torch.no_grad():
    #         preds = model(X_train)
    #         pred_mean = preds.mean.numpy()
    #         pred_std = np.sqrt(preds.variance.numpy())

    else:
        raise ValueError(f"Unknown model_type {model_type}")
        
    # -------------------------
    # Predict multiple times (Bayesian sampling)
    # -------------------------
    model.eval()
    preds = []
    with torch.no_grad():
        for _ in range(30):  # n_samples
            preds.append(model(torch.tensor(x_train, dtype=torch.float32).to(device)).cpu().numpy())
    preds = np.stack(preds, axis=0)  # (30, batch_size, 1)
    pred_mean = preds.mean(axis=0).flatten()
    pred_std = preds.std(axis=0).flatten()

    # -------------------------
    # Noise identification
    # -------------------------
    if noise_identification == "residual":
        noisy_mask = (np.abs(pred_mean - y_train_noisy.numpy()) > 0.2)  # adjustable threshold
    elif noise_identification == "std":
        noisy_mask = (pred_std > 0.2)  # adjustable threshold
    elif noise_identification == "mc_percentile":
        lower_bound = pred_mean - 2 * pred_std
        upper_bound = pred_mean + 2 * pred_std
        noisy_mask = ~((y_train_noisy.numpy() >= lower_bound) & (y_train_noisy.numpy() <= upper_bound))
    else:
        raise NotImplementedError(f"Noise ID method {noise_identification} not implemented yet.")

    # -------------------------
    # Save original training set (before cleaning)
    # -------------------------
    x_train_original = x_train.copy()
    y_train_noisy_original = y_train_noisy.clone()

    # -------------------------
    # Label smoothing / correction
    # -------------------------
    if label_smoothing == "hard_replacement_neighbor":
        # Find neighbors and replace
        knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
        knn.fit(x_train)
        y_train_cleaned = y_train_noisy.clone()

        for idx in np.where(noisy_mask)[0]:
            distances, neighbor_indices = knn.kneighbors(x_train[idx].reshape(1, -1))
            y_neighbors = y_train_noisy[neighbor_indices[0]]
            y_train_cleaned[idx] = y_neighbors.mean()

        x_train_cleaned = x_train

    elif label_smoothing == "hard_replacement_model":
        y_train_cleaned = y_train_noisy.clone()
        y_train_cleaned[noisy_mask] = torch.tensor(pred_mean[noisy_mask])

        x_train_cleaned = x_train

    elif label_smoothing == "soft_smoothing_neighbor":
        knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
        knn.fit(x_train)
        y_train_cleaned = y_train_noisy.clone()

        for idx in np.where(noisy_mask)[0]:
            distances, neighbor_indices = knn.kneighbors(x_train[idx].reshape(1, -1))
            y_neighbors = y_train_noisy[neighbor_indices[0]]
            neighbor_mean = y_neighbors.mean()
            y_train_cleaned[idx] = 0.5 * y_train_noisy[idx] + 0.5 * neighbor_mean

        x_train_cleaned = x_train

    elif label_smoothing == "soft_smoothing_model":
        y_train_cleaned = y_train_noisy.clone()
        y_train_cleaned[noisy_mask] = (
            0.5 * y_train_noisy[noisy_mask] + 0.5 * torch.tensor(pred_mean[noisy_mask])
        )

        x_train_cleaned = x_train

    elif label_smoothing == "removal":
        keep_mask = ~noisy_mask
        x_train_cleaned = x_train[keep_mask]
        y_train_cleaned = y_train_noisy[keep_mask]

    else:
        raise NotImplementedError(f"Label smoothing method {label_smoothing} not implemented yet.")

    # -------------------------
    # Train RF on noisy and cleaned labels
    # -------------------------

    # RF on noisy (ALWAYS use original x_train, y_train_noisy)
    rf = RandomForestRegressor()
    rf.fit(x_train_original, y_train_noisy_original.numpy())
    preds_noisy = rf.predict(x_test)
    r2_noisy = r2_score(y_test.numpy(), preds_noisy)

    # RF on cleaned (use cleaned x_train, y_train_cleaned)
    rf = RandomForestRegressor()
    rf.fit(x_train_cleaned, y_train_cleaned.numpy())
    preds_cleaned = rf.predict(x_test)
    r2_cleaned = r2_score(y_test.numpy(), preds_cleaned)

    return r2_noisy, r2_cleaned

def variational_em_label_denoising(x_train, y_train_noisy, model, num_em_steps=5, num_samples=30, batch_size=64):
    """
    Perform Variational EM for label denoising.

    Parameters:
        x_train (np.ndarray): Training features.
        y_train_noisy (torch.Tensor): Noisy labels (shape: [N]).
        model (nn.Module): A Bayesian model supporting stochastic forward passes.
        num_em_steps (int): Number of EM iterations.
        num_samples (int): Number of Monte Carlo samples per E-step.
        batch_size (int): Batch size for training in M-step.

    Returns:
        y_denoised (torch.Tensor): Denoised label estimates (posterior means).
        y_var (torch.Tensor): Posterior variances (uncertainty).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    x_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_noisy = y_train_noisy.to(device)

    # Initialize y_denoised with noisy labels
    y_denoised = y_noisy.clone().detach()

    for em_step in range(num_em_steps):
        model.eval()
        preds_samples = []

        with torch.no_grad():
            for _ in range(num_samples):
                preds = model(x_tensor).squeeze(-1)
                preds_samples.append(preds)

        preds_samples = torch.stack(preds_samples)  # (num_samples, N)
        posterior_mean = preds_samples.mean(dim=0)
        posterior_var = preds_samples.var(dim=0)
        y_denoised = posterior_mean.detach()

        # Check if model is trainable (i.e., has parameters)
        if not any(p.requires_grad for p in model.parameters()):
            continue  # Skip M-step entirely for non-trainable models

        # M-step
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        dataset = TensorDataset(x_tensor, y_denoised.unsqueeze(1))
        train_loader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(5):
            for xb, yb in train_loader:
                optimizer.zero_grad()
                output = model(xb)
                loss = criterion(output, yb)
                loss.backward()
                optimizer.step()

    return y_denoised.cpu(), posterior_var.cpu()

if __name__ == "__main__":
    representation_types = ["pdv", "smiles"]
    tokenizer = SmilesTokenizer()
    max_vocab_size = 30

    # Step 1: Load QM9
    smiles_list, y = load_qm9_data(target="homo_lumo_gap", sample_size=500)

    for rep in representation_types:
        if rep == "ecfp4":
            x, y_used = smiles_to_ecfp4(smiles_list, y)

        elif rep == "pdv":
            x = []
            valid_indices = []
            for idx, smi in enumerate(smiles_list):
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                canonical = Chem.MolToSmiles(mol, isomericSmiles=False)
                try:
                    desc = rdkit_mol_descriptors_from_smiles(canonical)
                    x.append(desc)
                    valid_indices.append(idx)
                except:
                    continue
            x = np.array(x)
            y_used = y[valid_indices]

        elif rep == "smiles":
            vocab = build_vocab(smiles_list, tokenizer, max_vocab_size)
            vocab_size = len(vocab)

            ohe_list = []
            valid_indices = []
            for idx, smi in enumerate(smiles_list):
                try:
                    ohe = smiles_to_ohe(smi, tokenizer, vocab)
                    ohe_list.append(ohe)
                    valid_indices.append(idx)
                except:
                    continue

            x = pad_sequence(ohe_list, batch_first=True, padding_value=0)
            x = x.view(x.size(0), -1)  # Flatten for DNN compatibility
            y_used = y[valid_indices]

        else:
            raise ValueError(f"Unknown representation: {rep}")


    # Step 3: Split
    x_train, x_test, y_train, y_test = train_test_split(x, y_used, test_size=0.2, random_state=42)

    # Step 4: Define noise levels
    sigma_values = [0.1, 0.3, 0.5]

    # Step 5: Run experiments
    # for sigma in sigma_values:
    #     print(f"\n=== Sigma = {sigma} ===")
    #     y_train_noisy = add_regression_noise(y_train, sigma=sigma)

    #     for config in experiment_grid:
    #         print(f"Running config: {config}")
    #         r2_noisy, r2_cleaned = run_experiment(config, x_train, y_train_noisy, x_test, y_test, sigma)
    #         print(f"   R² with noisy labels: {r2_noisy:.4f}")
    #         print(f"   R² with cleaned labels: {r2_cleaned:.4f}")

    # Run EM experiments
    for sigma in sigma_values:
        print(f"\n=== Sigma = {sigma} ===")

        # Add noise
        y_train_noisy = add_regression_noise(y_train, sigma=sigma)

        # Normalize using clean training stats
        y_mean = y_train.mean()
        y_std = y_train.std()

        y_train_normalized = (y_train - y_mean) / y_std
        y_train_noisy_normalized = (y_train_noisy - y_mean) / y_std
        y_test_normalized = (y_test - y_mean) / y_std

        print("Clean (normalized):", y_train_normalized[:5])
        print("Noisy (normalized):", y_train_noisy_normalized[:5])

        for model_type in ["qrf", "ngboost"]:
            print(f"--- Using model: {model_type} ---")

            if model_type in ["full", "last_layer", "variational"]:
                model = DNNRegressionModel(input_size=x_train.shape[1], hidden_size1=128, hidden_size2=64)
                if model_type == "full":
                    model = apply_bayesian_transformation(model)
                elif model_type == "last_layer":
                    model = apply_bayesian_transformation_last_layer(model)
                elif model_type == "variational":
                    model = apply_bayesian_transformation_last_layer_variational(model)

                # Run EM denoising
                y_denoised, _ = variational_em_label_denoising(
                    x_train=x_train,
                    y_train_noisy=y_train_noisy,
                    model=model,
                    num_em_steps=5,
                    num_samples=30
                )

            elif model_type == "qrf":
                qrf = RandomForestQuantileRegressor(n_estimators=500)
                qrf.fit(x_train, y_train_noisy.numpy())
                model = QRFWrapper(qrf)
                y_denoised, _ = variational_em_label_denoising(
                    x_train=x_train,
                    y_train_noisy=y_train_noisy,
                    model=model,
                    num_em_steps=5,
                    num_samples=30
                )

            elif model_type == "ngboost":
                ngb = NGBRegressor(
                    Dist=Normal,
                    Score=MLE,
                    n_estimators=500,
                    learning_rate=0.01,
                    natural_gradient=True,
                    verbose=False
                )
                ngb.fit(x_train, y_train_noisy.numpy())
                model = NGBoostWrapper(ngb)
                y_denoised, _ = variational_em_label_denoising(
                    x_train=x_train,
                    y_train_noisy=y_train_noisy,
                    model=model,
                    num_em_steps=5,
                    num_samples=30
                )

            # Compare RF on noisy vs cleaned labels
            rf = RandomForestRegressor()
            rf.fit(x_train, y_train_noisy.numpy())
            r2_noisy = r2_score(y_test.numpy(), rf.predict(x_test))

            rf.fit(x_train, y_denoised.numpy())
            r2_cleaned = r2_score(y_test.numpy(), rf.predict(x_test))

            print(f"   R² with noisy labels:   {r2_noisy:.4f}")
            print(f"   R² with cleaned labels: {r2_cleaned:.4f}")



# Next up:
# - Different bayesian transformations/models
# - Plots - need to save results
# - Different noise cleaning methods
