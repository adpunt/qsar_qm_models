# json and os are used at module level by load_best_hyperparameters, which is the
# ONLY way a tuned hyperparameter reaches a model. Neither was imported here:
# `os` happened to arrive through one of the star-imports below, and `json` did
# not arrive at all, so the tuned branch raised NameError the moment both files
# existed. It had never been reached, because results/hyperparameter_decisions.json
# has never existed -- the function returned early every time (RERUN_PLAN.md 5.7g).
# Every other use of json in this file is a function-local `import json`.
import json
import os

import torch
import torch.nn as nn
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
from torch_geometric.nn import GCNConv, GINConv, GATv2Conv, global_mean_pool, global_add_pool
import numpy as np
import math
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torch.utils.data import DataLoader as TorchDataLoader
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
from torch_geometric.nn import GCNConv, GINConv, MessagePassing
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch_geometric.typing import Adj, OptTensor, PairTensor, Size
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
# from torch_sparse import SparseTensor
from torch_geometric.nn.inits import glorot, zeros
import gpytorch
from typing import Union
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
# lightgbm, xgboost, quantile_forest and grakel are imported INSIDE the
# functions that use them, not here.
#
# At module scope every job loaded every one of them whatever model it was
# asked to run, so a Gaussian-process task sat in a process with both boosting
# libraries resident -- and that is the precondition of the segfault in
# RERUN_PLAN.md 2.8e. Measured after this change: importing this module no
# longer loads lightgbm, xgboost or quantile_forest at all.
#
# What CANNOT move, and why the environment fix is the real one: torch,
# torch_geometric, gpytorch and gauche are needed to DEFINE this module --
# classes below inherit from nn.Module, MessagePassing, gpytorch.models.ExactGP
# and SIGP. A lightgbm job therefore still loads the whole Gaussian-process
# stack, and no rearrangement of imports can change that. grakel is the same
# story from the other side: gauche.kernels.graph_kernels imports it, so it
# arrives anyway. The local import below is honesty about the dependency, not
# a saving.
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.svm import SVR, SVC
from torch.nn.utils import parameters_to_vector as Params2Vec, vector_to_parameters as Vec2Params
import matplotlib.pyplot as plt
import torchbnn as bnn
from torchhk import transform_model, transform_layer
try:
    from botorch.fit import fit_gpytorch_mll as fit_gpytorch_model
except ImportError:
    from botorch import fit_gpytorch_model
import gauche
from gauche.kernels.fingerprint_kernels import *
from gauche.kernels.graph_kernels import *
from gauche import SIGP, NonTensorialInputs
from gauche.dataloader import MolPropLoader
from gauche.dataloader.data_utils import transform_data
from gauche.kernels.graph_kernels import WeisfeilerLehmanKernel, VertexHistogramKernel
try:
    import torchcp
    from torchcp.regression.predictor import SplitPredictor, ACIPredictor
    from torchcp.classification.score import APS
    from torchcp.regression.score import ABS
except ImportError:
    torchcp = None  # conformal prediction unavailable (torchsort/PyTorch version mismatch)
from sklearn.isotonic import IsotonicRegression
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool
from rdkit import Chem


from utils import * 
from loss_functions import *

# The shared parameter spec. models/ is on sys.path when process_and_train.py
# runs from scripts/, but not when this module is imported from the repo root
# (the parity audit does that), so resolve it from this file's own location.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from model_defaults import (
    SPEC_VERSION, BAYESIAN_DEFAULTS, GP_DEFAULTS, NEURAL_DEFAULTS,
    SKLEARN_DEFAULTS, UNCERTAINTY_DEFAULTS, gp_fit_threads, provenance_columns,
    bnn_kl_weight, sklearn_params, spec_hash,
)

def bnn_elbo_criterion(base_criterion, model, n_train):
    """Wrap a plain loss so a torchbnn network is fitted on the ELBO.

    BNN-alpha and BNN-beta were trained with plain MSE until 2026-08-27. There
    was no KL, ELBO or BKLLoss anywhere in either pipeline, so nothing pulled the
    variational posterior toward the prior and `prior_sigma` was only an
    initialisation. torchbnn samples weights in train AND eval mode, so the
    posterior width received MSE gradients, which drive it toward zero: what was
    reported as epistemic uncertainty was the residual weight noise the fit
    happened to leave. The VBLL variants carried a KL term all along, so the two
    families were never on equal footing (RERUN_PLAN.md 2.12).

    The weight comes from the shared spec: 'elbo' means 1 / n_train, because the
    objective is sum_i NLL_i + KL and the criterion here is the MEAN over a
    batch.
    """
    kl = bnn.BKLLoss(reduction='mean', last_layer_only=False)
    weight = bnn_kl_weight(n_train)

    def criterion(output, target, *rest):
        return base_criterion(output, target, *rest) + weight * kl(model)

    criterion.kl_weight = weight
    criterion.base = base_criterion
    return criterion


# TODO: reorder imports

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNNRegressionModel(nn.Module):
    """Vanilla RNN with one recurrent layer"""

    def __init__(self, input_size, hidden_size=32, num_layers=1):
        """
        Vanilla RNN

        Parameters
        ----------
        input_size : int
            The number of expected features in the input vector
        hidden_size : int
            The number of features in the hidden state

        """
        super(RNNRegressionModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, hn = self.rnn(x, h0)
        out = out[:, -1]
        out = self.dropout(out)
        out = self.fc(out)
        return out


class GRURegressionModel(nn.Module):
    """GRU network with one recurrent layer"""

    def __init__(self, input_size, hidden_size=32, num_layers=1):
        """
        GRU network

        Parameters
        ----------
        input_size : int
            The number of expected features in the input vector
        hidden_size : int
            The number of features in the hidden state

        """
        super(GRURegressionModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, hn = self.gru(x, h0)
        out = out[:, -1]
        out = self.dropout(out)
        out = self.fc(out)
        return out

class ModelTrainer(object):
    """A class that provides training and validation infrastructure for the model and keeps track of training and validation metrics."""

    def __init__(self, model, lr, name=None, clip_gradients=False):
        """
        Initialization.

        Parameters
        ----------
        model : nn.Module
            a model
        lr : float
            learning rate for one training step

        """
        self.model = model
        self.lr = lr
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        self.clip_gradients = clip_gradients
        self.model.to(device)

        self.train_loss = []
        self.batch_loss = []
        self.val_loss = []

    def _train_epoch(self, loader):
        self.model.train()
        epoch_loss = 0
        batch_losses = []
        for i, (X_batch, y_batch) in enumerate(loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            self.optimizer.zero_grad()
            y_pred = self.model(X_batch)
            loss = self.criterion(y_pred, y_batch.unsqueeze(1))
            loss.backward()

            if self.clip_gradients:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1, norm_type=2)

            self.optimizer.step()
            epoch_loss += loss.item()
            batch_losses.append(loss.item())

        return epoch_loss / len(loader), batch_losses

    def _eval_epoch(self, loader):
        self.model.eval()
        val_loss = 0
        predictions = []
        targets = []
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch.unsqueeze(1))
                val_loss += loss.item()
                predictions.append(y_pred.detach().numpy())
                targets.append(y_batch.unsqueeze(1).detach().numpy())

        predictions = np.concatenate(predictions).flatten()
        targets = np.concatenate(targets).flatten()
        return val_loss / len(loader), predictions, targets

    def train(self, train_loader, val_loader, args, s, iteration, file_no, model_name, rep, print_every=10):
        for e in range(args.epochs):
            train_loss, train_loss_batches = self._train_epoch(train_loader)
            val_loss, _, _ = self._eval_epoch(val_loader)
            self.batch_loss += train_loss_batches
            self.train_loss.append(train_loss)
            self.val_loss.append(val_loss)
            if e % print_every == 0:
                print(f"Epoch {e+0:03} | train_loss: {train_loss:.5f} | val_loss: {val_loss:.5f}")
        
        # Save per-epoch metrics if requested
        if args and hasattr(args, 'save_per_epoch_metrics') and args.save_per_epoch_metrics:
            save_per_epoch_metrics(
                train_losses=self.train_loss,
                val_losses=self.val_loss,
                filepath=args.filepath,
                model_name=model_name,
                rep=rep,
                sigma_noise=s,
                iteration=iteration,
                file_no=file_no
            )

    def validate(self, val_loader):
        """
        Validate the model

        Parameters
        ----------
        val_loader :
            a dataloader with training data

        Returns
        -------
        Tuple[list, list, list]
            Loss, y_predicted, y_target for each datapoint in val_loader.
        """
        loss, y_pred, y_targ = self._eval_epoch(val_loader)
        return loss, y_pred, y_targ

# Removed 2026-08-27: a duplicate top-level definition that a later one in
# this file shadowed, so it never ran -- class GCN (shadowed by the definition at 286).
# scripts/test_no_shadowed_definitions.py fails if another one appears.

"""
Graph Neural Network Architectures for Molecular Property Prediction

Implements GCN, GAT, and GIN with optional Bayesian support via dropout.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool, global_add_pool


class GCN(nn.Module):
    """
    Graph Convolutional Network (Kipf & Welling, 2017).
    """
    def __init__(self, num_node_features, hidden_dim=128, num_layers=3, 
                 dropout=0.0, use_edge_features=False):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(num_node_features, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.regression_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            if i < len(self.convs) - 1:  # Don't dropout last layer
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        graph_embedding = global_mean_pool(x, batch)
        
        # Regression
        out = self.regression_head(graph_embedding)
        
        return out, graph_embedding


class GAT(nn.Module):
    """
    Graph Attention Network (Veličković et al., 2018).
    """
    def __init__(self, num_node_features, hidden_dim=128, num_layers=3,
                 dropout=0.0, num_heads=4, use_edge_features=False):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        
        # First layer: multi-head attention with concat
        self.convs.append(GATConv(num_node_features, hidden_dim, heads=num_heads, concat=True))
        current_dim = hidden_dim * num_heads
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(current_dim, hidden_dim, heads=num_heads, concat=True))
        
        # Last layer: single head, no concat
        if num_layers > 1:
            self.convs.append(GATConv(current_dim, hidden_dim, heads=1, concat=False))
            final_dim = hidden_dim
        else:
            final_dim = current_dim
        
        self.regression_head = nn.Linear(final_dim, 1)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.elu(x)
            if i < len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        graph_embedding = global_mean_pool(x, batch)
        out = self.regression_head(graph_embedding)
        
        return out, graph_embedding


class GIN(nn.Module):
    """
    Graph Isomorphism Network (Xu et al., 2019).
    
    More expressive than GCN - can distinguish different graph structures.
    """
    def __init__(self, num_node_features, hidden_dim=128, num_layers=3,
                 dropout=0.0, use_edge_features=False):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First GIN layer
        nn1 = nn.Sequential(
            nn.Linear(num_node_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs.append(GINConv(nn1, train_eps=True))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Subsequent GIN layers
        for _ in range(num_layers - 1):
            nn_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(nn_layer, train_eps=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        self.regression_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            if i < len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Sum pooling (better for GIN)
        graph_embedding = global_add_pool(x, batch)
        out = self.regression_head(graph_embedding)
        
        return out, graph_embedding


# Removed 2026-08-27: a duplicate top-level definition that a later one in
# this file shadowed, so it never ran -- create_gnn_model (shadowed by the copy at 4106).
# scripts/test_no_shadowed_definitions.py fails if another one appears.

class GATv2(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64, heads=1, dropout=0.5):
        super(GATv2, self).__init__()
        # Initialize the first GATv2 convolutional layer
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        # Initialize the second GATv2 convolutional layer, taking the concatenated output of the first layer
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
        # Initialize the third GATv2 convolutional layer, with output not concatenated
        self.conv3 = GATv2Conv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.dropout(x)
        x = torch.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = torch.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        
        # Pool the node features across the graph to create graph-level features
        x = global_mean_pool(x, batch)
        
        return x

class GATv2a(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64, heads=1, dropout=0.5):
        super(GATv2a, self).__init__()
        # Initialize the first GATv2 convolutional layer
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        # Initialize the second GATv2 convolutional layer, taking the concatenated output of the first layer
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
        # Initialize the third GATv2 convolutional layer, with output not concatenated
        self.conv3 = GATv2Conv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.dropout(x)
        x = torch.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = torch.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        
        # Pool the node features across the graph to create graph-level features
        x = global_add_pool(x, batch)
        
        return x

class MLPRegressor(nn.Module):
    """Multi-Layer Perceptron for regression on non-sequential data."""

    def __init__(self, input_size, hidden_size=32, num_hidden_layers=2, dropout_rate=0.2):
        super(MLPRegressor, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        self.output_layer = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        x = self.dropout(x)
        x = self.output_layer(x)
        return x

class Gauche(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel_class):
        super(Gauche, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel_class())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class MTLRegressionModel(nn.Module):
    """Multi-task learning model with shared hidden layers and multiple output heads."""
    def __init__(self, input_size, hidden_size=128, num_tasks=2):
        super(MTLRegressionModel, self).__init__()
        self.shared_fc1 = nn.Linear(input_size, hidden_size)
        self.shared_fc2 = nn.Linear(hidden_size, hidden_size)

        # Separate output layers for different tasks
        self.task_heads = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(num_tasks)])

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.activation(self.shared_fc1(x))
        x = self.dropout(x)
        x = self.activation(self.shared_fc2(x))
        x = self.dropout(x)

        return torch.cat([head(x) for head in self.task_heads], dim=1)

class ResidualMLP(nn.Module):
    """Fully connected model with residual connections."""
    def __init__(self, input_size, hidden_size=128, num_layers=3):
        super(ResidualMLP, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.output_layer = nn.Linear(hidden_size, 1)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            residual = x
            x = self.activation(layer(x))
            x = x + residual  # Residual connection
        x = self.dropout(x)
        return self.output_layer(x)
class FactorizationMLP(nn.Module):
    """Factorization Machine with MLP for bit vector data."""
    def __init__(self, input_size, hidden_size=128, factor_size=16):
        super(FactorizationMLP, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.factor_matrix = nn.Parameter(torch.randn(input_size, factor_size))

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        linear_term = self.linear(x)

        # Factorization term: element-wise interaction
        interaction_term = 0.5 * torch.sum(torch.pow(torch.matmul(x, self.factor_matrix), 2) - 
                                           torch.matmul(torch.pow(x, 2), torch.pow(self.factor_matrix, 2)), dim=1, keepdim=True)

        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        mlp_out = self.fc2(x)

        return linear_term + interaction_term + mlp_out

# Subclass the SIGP call that allows us to use kernels over
# discrete inputs with GPyTorch and BoTorch machinery
class GraphGP(SIGP):
    def __init__(
        self,
        train_x: NonTensorialInputs,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
        kernel: gpytorch.kernels.Kernel,
        **kernel_kwargs,
    ):
        """
        A subclass of the SIGP class that allows us to use kernels over
        discrete inputs with GPyTorch and BoTorch machinery.

        Parameters:
        -----------
        train_x: NonTensorialInputs
            The training inputs for the model. These are graph objects.
        train_y: torch.Tensor
            The training labels for the model.
        likelihood: gpytorch.likelihoods.Likelihood
            The likelihood function for the model.
        kernel: gpytorch.kernels.Kernel
            The kernel function for the model.
        **kernel_kwargs:
            The keyword arguments for the kernel function.
        """

        super().__init__(train_x, train_y, likelihood)
        self.mean = gpytorch.means.ConstantMean()
        self.covariance = kernel

    def forward(self, x):
        """
        A forward pass through the model.
        """
        mean = self.mean(torch.zeros(len(x), 1)).float()
        covariance = self.covariance(x)

        # because graph kernels operate over discrete inputs it is beneficial
        # to add some jitter for numerical stability
        jitter = max(covariance.diag().mean().detach().item() * 1e-4, 1e-4)
        covariance += torch.eye(len(x)) * jitter
        return gpytorch.distributions.MultivariateNormal(mean, covariance)


def training(loader, model, loss, optimizer, y_train_noisy=None):
    """Training one epoch

    Args:
        loader (DataLoader): loader (DataLoader): training data divided into batches
        model (nn.Module): GNN model to train on
        loss (nn.functional): loss function to use during training
        optimizer (torch.optim): optimizer during training
        y_train_noisy (Tensor, optional): noisy training targets from Rust. If None, uses d.y from loader.

    Returns:
        float: training loss
    """
    model.train()

    current_loss = 0
    current_idx = 0
    
    for d in loader:
        optimizer.zero_grad()
        d.x = d.x.float()

        out = model(d)

        # Use noisy targets if provided, otherwise use d.y
        if y_train_noisy is not None:
            batch_size = len(d.y)
            y_batch = y_train_noisy[current_idx:current_idx + batch_size].to(d.y.device)
            current_idx += batch_size
        else:
            y_batch = d.y

        l = loss(out, torch.reshape(y_batch, (len(y_batch), 1)))
        current_loss += l / len(loader)
        l.backward()
        optimizer.step()
    
    return current_loss, model


def validation(loader, model, loss, y_val_noisy=None):
    """Validation

    Args:
        loader (DataLoader): validation set in batches
        model (nn.Module): current trained model
        loss (nn.functional): loss function
        y_val_noisy (Tensor, optional): noisy validation targets from Rust. If None, uses d.y from loader.

    Returns:
        float: validation loss
    """
    model.eval()
    val_loss = 0
    current_idx = 0
    
    with torch.no_grad():
        for d in loader:
            out = model(d)
            
            # Use noisy targets if provided, otherwise use d.y
            if y_val_noisy is not None:
                batch_size = len(d.y)
                y_batch = y_val_noisy[current_idx:current_idx + batch_size].to(d.y.device)
                current_idx += batch_size
            else:
                y_batch = d.y
            
            l = loss(out, torch.reshape(y_batch, (len(y_batch), 1)))
            val_loss += l / len(loader)
    
    return val_loss


@torch.no_grad()
def testing(loader, model, y_test_noisy=None):
    """Testing

    Args:
        loader (DataLoader): test dataset
        model (nn.Module): trained model
        y_test_noisy (Tensor, optional): noisy test targets from Rust. If None, uses d.y from loader.

    Returns:
        float: test loss
    """
    loss = torch.nn.MSELoss()
    test_loss = 0
    test_target = np.empty((0))
    test_y_target = np.empty((0))
    current_idx = 0
    
    for d in loader:
        out = model(d)
        
        # Use noisy targets if provided, otherwise use d.y
        if y_test_noisy is not None:
            batch_size = len(d.y)
            y_batch = y_test_noisy[current_idx:current_idx + batch_size].to(d.y.device)
            current_idx += batch_size
        else:
            y_batch = d.y
        
        l = loss(out, torch.reshape(y_batch, (len(y_batch), 1)))
        test_loss += l / len(loader)

        # save prediction vs ground truth values for plotting
        test_target = np.concatenate((test_target, out.detach().cpu().numpy()[:, 0]))
        test_y_target = np.concatenate((test_y_target, y_batch.detach().cpu().numpy()))

    return test_loss, test_target, test_y_target

def train_epochs(epochs, model, train_loader, val_loader, args, s, iteration, file_no, model_name,
                 y_train_noisy=None, y_val_noisy=None, learning_rate=0.001):
    """Training over all epochs
    Args:
        epochs (int): number of epochs to train for
        model (nn.Module): the current model
        train_loader (DataLoader): training data in batches
        val_loader (DataLoader): validation data in batches
        args: arguments object containing save_per_epoch_metrics flag and filepath
        s: sigma noise level
        iteration: current iteration number
        file_no: file number identifier
        model_name: name of the model
        y_train_noisy (Tensor, optional): noisy training targets from Rust
        y_val_noisy (Tensor, optional): noisy validation targets from Rust
        learning_rate (float, optional): learning rate for optimizer. Default: 0.001
    Returns:
        array: returning train and validation losses over all epochs, prediction and ground truth values for training data in the last epoch
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    loss = torch.nn.MSELoss()
    train_target = np.empty((0))
    train_y_target = np.empty((0))
    train_loss = np.empty(epochs)
    val_loss = np.empty(epochs)
    for epoch in range(epochs):
        epoch_loss, model = training(train_loader, model, loss, optimizer, y_train_noisy=y_train_noisy)
        v_loss = validation(val_loader, model, loss, y_val_noisy=y_val_noisy)
        
        # Record predictions vs actual values for training data from last epoch
        if epoch == epochs - 1:
            current_idx = 0
            for d in train_loader:
                out = model(d)
                
                # Use noisy targets if provided, otherwise use d.y
                if y_train_noisy is not None:
                    batch_size = len(d.y)
                    y_batch = y_train_noisy[current_idx:current_idx + batch_size]
                    current_idx += batch_size
                else:
                    y_batch = d.y
                
                train_target = np.concatenate((train_target, out.detach().cpu().numpy()[:, 0]))
                train_y_target = np.concatenate((train_y_target, y_batch.detach().cpu().numpy()))
        train_loss[epoch] = epoch_loss.detach().cpu().numpy()
        val_loss[epoch] = v_loss.detach().cpu().numpy()
        # print current train and val loss
        if epoch % 2 == 0:
            print(
                "Epoch: "
                + str(epoch)
                + ", Train loss: "
                + str(epoch_loss.item())
                + ", Val loss: "
                + str(v_loss.item())
            )
    
    # Save per-epoch metrics if requested
    if args.save_per_epoch_metrics:
        save_per_epoch_metrics(
            train_losses=train_loss,
            val_losses=val_loss,
            filepath=args.filepath,
            model_name=model_name,
            rep='graph',
            sigma_noise=s,
            iteration=iteration,
            file_no=file_no
        )
    
    return train_loss, val_loss, train_target, train_y_target, model

@torch.no_grad()
def testing_co_teaching(loader, model):
    loss = torch.nn.MSELoss()
    test_loss_f = 0
    test_loss_g = 0
    test_target = np.empty((0))
    test_y_target = np.empty((0))

    for data in loader:
        data = data.to(model.device)
        output_f = model.model_f(data)
        output_g = model.model_g(data)

        loss_f = loss(output_f, data.y.view(-1, 1))
        loss_g = loss(output_g, data.y.view(-1, 1))

        test_loss_f += loss_f.item() / len(loader)
        test_loss_g += loss_g.item() / len(loader)

        # Save prediction vs ground truth values for plotting
        test_target = np.concatenate((test_target, output_f.cpu().detach().numpy()[:, 0]))
        test_y_target = np.concatenate((test_y_target, data.y.cpu().detach().numpy()))

    return (test_loss_f + test_loss_g) / 2, test_target, test_y_target

def mutually_agreed_samples(loss_f, loss_g, tolerance=0.05):
    """
    Select mutually agreed samples where losses between the two models are within a tolerance range.
    """
    loss_diff = torch.abs(loss_f - loss_g)
    agreed_indices = torch.where(loss_diff < tolerance)[0]
    
    return agreed_indices

def training_co_teaching(loader, model, loss, ratio=0.5, tolerance=0.2):
    model.model_f.train()
    model.model_g.train()

    current_loss_f = 0
    current_loss_g = 0
    
    for data in loader:
        data = data.to(model.device)
        model.optimizer_f.zero_grad()
        model.optimizer_g.zero_grad()

        output_f = model.model_f(data)
        output_g = model.model_g(data)

        loss_f = loss(output_f, data.y.view(-1, 1)).squeeze()  # Remove extra dimension
        loss_g = loss(output_g, data.y.view(-1, 1)).squeeze()

        # Ranking losses to select small-loss samples
        num_small_samples = int(ratio * len(loss_f))

        # Ensure indices are 1D tensors
        sorted_indices_f = torch.argsort(loss_f)[:num_small_samples]
        sorted_indices_g = torch.argsort(loss_g)[:num_small_samples]

        # Flatten the indices
        sorted_indices_f = sorted_indices_f.view(-1)
        sorted_indices_g = sorted_indices_g.view(-1)

        # Get mutually agreed samples within a tolerance range
        agreed_indices = mutually_agreed_samples(loss_f, loss_g, tolerance)

        # Ensure agreed_indices is a 1D tensor
        if agreed_indices.ndim > 1:
            agreed_indices = agreed_indices.view(-1)

        # Combine small-loss and agreed-upon samples for training
        combined_indices = torch.cat([sorted_indices_f, sorted_indices_g, agreed_indices])

        selected_indices = combined_indices.unique()

        # Recompute the losses for the selected samples
        selected_loss_f = loss_f[selected_indices]
        selected_loss_g = loss_g[selected_indices]

        # Compute average losses and perform backpropagation
        current_loss_f += selected_loss_f.mean().item() / len(loader)
        current_loss_g += selected_loss_g.mean().item() / len(loader)

        selected_loss_f.mean().backward()
        model.optimizer_f.step()

        selected_loss_g.mean().backward()
        model.optimizer_g.step()

    return (current_loss_f + current_loss_g) / 2, model

def validation_co_teaching(loader, model, loss, ratio=0.5, tolerance=0.2):
    model.model_f.eval()
    model.model_g.eval()
    val_loss_f = 0
    val_loss_g = 0

    with torch.no_grad():
        all_losses_f = []
        all_losses_g = []
        all_indices = []

        for data in loader:
            data = data.to(model.device)
            output_f = model.model_f(data)
            output_g = model.model_g(data)

            loss_f = loss(output_f, data.y.view(-1, 1)).squeeze()  # Remove extra dimension
            loss_g = loss(output_g, data.y.view(-1, 1)).squeeze()

            all_losses_f.append(loss_f)
            all_losses_g.append(loss_g)
            all_indices.append(torch.arange(len(loss_f)))

        # Concatenate all losses and indices
        all_losses_f = torch.cat(all_losses_f)
        all_losses_g = torch.cat(all_losses_g)
        all_indices = torch.cat(all_indices)

        # Ranking losses to select small-loss samples
        num_small_samples = int(ratio * len(all_losses_f))
        sorted_indices_f = torch.argsort(all_losses_f)[:num_small_samples]
        sorted_indices_g = torch.argsort(all_losses_g)[:num_small_samples]

        # Get mutually agreed samples within a tolerance range
        agreed_indices = mutually_agreed_samples(all_losses_f, all_losses_g, tolerance)

        # Ensure agreed_indices is a 1D tensor
        if agreed_indices.ndim > 1:
            agreed_indices = agreed_indices.view(-1)

        # Combine small-loss and agreed-upon samples for training
        combined_indices = torch.cat([sorted_indices_f, sorted_indices_g, agreed_indices])
        selected_indices = combined_indices.unique()

        # Recompute the losses for the selected samples
        selected_loss_f = all_losses_f[selected_indices]
        selected_loss_g = all_losses_g[selected_indices]

        if selected_loss_f.numel() > 0:
            val_loss_f = selected_loss_f.mean().item()
            val_loss_g = selected_loss_g.mean().item()

    return (val_loss_f + val_loss_g) / 2

# Add forget_rate parameter to control the percentage of samples to drop at each epoch
def train_epochs_co_teaching(epochs, model, train_loader, val_loader, path, ratio=0.5, tolerance=0.2, forget_rate=0.2):
    loss = torch.nn.MSELoss(reduction='none')
    train_target = np.empty((0))
    train_y_target = np.empty((0))
    train_loss = np.empty(epochs)
    val_loss = np.empty(epochs)
    best_loss = math.inf

    for epoch in range(epochs):
        epoch_loss, model = training_co_teaching(train_loader, model, loss, tolerance=tolerance, ratio=ratio)
        v_loss = validation_co_teaching(val_loader, model, loss, tolerance=tolerance, ratio=ratio)

        if v_loss < best_loss:
            torch.save(model.model_f.state_dict(), f"{path}_f.pth")
            torch.save(model.model_g.state_dict(), f"{path}_g.pth")
            best_loss = v_loss

        if epoch == epochs - 1:
            for batch_idx, data in enumerate(train_loader):
                data = data.to(model.device)
                output_f = model.model_f(data)
                output_g = model.model_g(data)

                # Record truly vs predicted values for training data from last epoch
                train_target = np.concatenate((train_target, output_f.cpu().detach().numpy()[:, 0]))
                train_y_target = np.concatenate((train_y_target, noisy_y_values.cpu().numpy()))

        train_loss[epoch] = epoch_loss
        val_loss[epoch] = v_loss

        # Print current train and val loss
        if epoch % 2 == 0:
            print(
                f"Epoch: {epoch}, Train loss: {epoch_loss:.4f}, Val loss: {v_loss:.4f}"
            )

    return train_loss, val_loss, train_target, train_y_target, model

class DNNRegressionModel(nn.Module):
    """Densely-connected neural network for binding affinity prediction"""

    def __init__(self, input_size, hidden_size1=32, hidden_size2=32):
        """
        Fully-connected neural network

        Parameters
        ----------
        input_size : int
            Number of features in the input vector
        hidden_size1 : int
            Number of neurons in the first hidden layer
        hidden_size2 : int
            Number of neurons in the second hidden layer
        """
        super(DNNRegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, 1)
        self.activation = nn.ReLU()  # Default activation (will be tuned)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # No activation for regression output
        return x

class FlexibleDNNRegressionModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, activation_fn=nn.ReLU(), dropout=0.2):
        super(FlexibleDNNRegressionModel, self).__init__()

        layers = []
        in_size = input_size

        for h_size in hidden_sizes:
            layers.append(nn.Linear(in_size, h_size))
            layers.append(activation_fn)
            layers.append(nn.Dropout(p=dropout))
            in_size = h_size

        layers.append(nn.Linear(in_size, 1))  # Output layer (regression)
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def apply_bayesian_transformation(model):
    """
    Converts an existing PyTorch model's Linear layers to Bayesian Linear layers.
    
    Parameters
    ----------
    model : nn.Module
        The PyTorch model to be transformed.
        
    Returns
    -------
    model : nn.Module
        The transformed model with Bayesian layers.
    """
    # Convert Linear -> BayesLinear
    transform_model(
        model, 
        nn.Linear, 
        bnn.BayesLinear, 
        args={
            # From the shared spec, not restated here: a number typed into this
            # file cannot be changed from model_defaults.py, while every results
            # row carries a spec_hash asserting that it can (RERUN_PLAN.md 2.13).
            "prior_mu": BAYESIAN_DEFAULTS['bnn_prior_mu'],
            "prior_sigma": 0.1,
            "in_features": ".in_features",
            "out_features": ".out_features", 
            "bias": ".bias"
        }, 
        attrs={"weight_mu": ".weight"}
    )
    return model

def apply_bayesian_transformation_last_layer(model):
    """
    Replaces only the final nn.Linear layer in the model with a Bayesian Linear layer.
    Uses torchhk-style transform_layer to apply the conversion.

    Parameters
    ----------
    model : nn.Module
        Your PyTorch model with at least one nn.Linear layer.

    Returns
    -------
    model : nn.Module
        The modified model with the final nn.Linear replaced by bnn.BayesLinear.
    """
    last_linear_name = None
    last_linear_module = None

    # Find the last nn.Linear layer
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Linear):
            last_linear_name = name
            last_linear_module = module
            break

    if last_linear_module is None:
        raise ValueError("No nn.Linear layer found to replace.")

    # Build Bayesian version of the final layer
    bayesian_layer = transform_layer(
        last_linear_module,
        nn.Linear,
        bnn.BayesLinear,
        args={
            # From the shared spec (RERUN_PLAN.md 2.13).
            "prior_mu": BAYESIAN_DEFAULTS['bnn_prior_mu'],
            "prior_sigma": BAYESIAN_DEFAULTS['bnn_prior_sigma'],
            "in_features": ".in_features",
            "out_features": ".out_features",
            "bias": ".bias"
        },
        attrs={"weight_mu": ".weight"}
    )

    # Helper: assign new module to its place in the model
    def set_nested_attr(obj, attr_path, value):
        attrs = attr_path.split(".")
        for a in attrs[:-1]:
            obj = getattr(obj, a)
        setattr(obj, attrs[-1], value)

    # Replace the final linear layer
    set_nested_attr(model, last_linear_name, bayesian_layer)

    return model

class VBLLLayer(nn.Module):
    """
    Variational Bayesian Last Layer (Harrison 2024).

    Maintains a mean-field variational posterior q(W) = N(weight_mu, diag(exp(weight_log_sigma)^2))
    over the weight matrix, with a standard normal prior p(W) = N(0, I).
    Uses the reparameterization trick for gradient estimation.
    """
    def __init__(self, in_features, out_features, prior_mu=None, prior_sigma=None):
        super(VBLLLayer, self).__init__()
        # From the shared spec, not restated here (RERUN_PLAN.md 2.13).
        _init_log_sigma = BAYESIAN_DEFAULTS['vbll_init_log_sigma']
        _init_log_noise = BAYESIAN_DEFAULTS['vbll_init_log_noise_var']
        self.in_features = in_features
        self.out_features = out_features
        self.prior_mu = (BAYESIAN_DEFAULTS['vbll_prior_mu']
                         if prior_mu is None else prior_mu)
        self.prior_sigma = (BAYESIAN_DEFAULTS['vbll_prior_sigma']
                            if prior_sigma is None else prior_sigma)

        # Variational posterior parameters
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_log_sigma = nn.Parameter(
            torch.full((out_features, in_features), _init_log_sigma))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_log_sigma = nn.Parameter(
            torch.full((out_features,), _init_log_sigma))

        # Learned log observation noise (aleatoric uncertainty)
        self.log_noise_var = nn.Parameter(torch.tensor(_init_log_noise))

        # Initialize weight_mu with Kaiming uniform
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        fan_in = in_features
        bound = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias_mu, -bound, bound)

    @property
    def noise_var(self):
        return torch.exp(self.log_noise_var)

    def kl_divergence(self):
        """Closed-form KL(q(W)||p(W)) for diagonal Gaussians."""
        prior_var = self.prior_sigma ** 2

        # Weight KL
        w_var = torch.exp(2.0 * self.weight_log_sigma)
        kl_w = 0.5 * torch.sum(
            w_var / prior_var
            + ((self.prior_mu - self.weight_mu) ** 2) / prior_var
            - 1.0
            + math.log(prior_var) - 2.0 * self.weight_log_sigma
        )

        # Bias KL
        b_var = torch.exp(2.0 * self.bias_log_sigma)
        kl_b = 0.5 * torch.sum(
            b_var / prior_var
            + ((self.prior_mu - self.bias_mu) ** 2) / prior_var
            - 1.0
            + math.log(prior_var) - 2.0 * self.bias_log_sigma
        )

        return kl_w + kl_b

    def forward(self, x):
        # Weights are SAMPLED in eval mode too, deliberately: the Monte Carlo
        # uncertainty passes need it. So a VBLL point prediction is the mean of
        # 100 stochastic passes, while a plain DNN's is one deterministic pass.
        #
        # This used to be written as `if self.training: ... else: ...` with the
        # two branches textually identical, which reads as though eval does
        # something different -- the posterior mean, say -- and is a trap for
        # anyone reasoning about whether a VBLL prediction is deterministic
        # (RERUN_PLAN.md 2.13). One branch now, and the reason with it.
        weight_sigma = torch.exp(self.weight_log_sigma)
        weight = self.weight_mu + weight_sigma * torch.randn_like(self.weight_mu)
        bias_sigma = torch.exp(self.bias_log_sigma)
        bias = self.bias_mu + bias_sigma * torch.randn_like(self.bias_mu)
        return F.linear(x, weight, bias)


class VBLLLoss(nn.Module):
    """
    ELBO loss for VBLL: Gaussian NLL (with learned noise) + KL divergence / n_data.

    NLL = 0.5 * log(noise_var) + 0.5 * (pred - target)^2 / noise_var
    Loss = mean(NLL) + sum(KL_i(q||p)) / n_data

    Supports both last-layer and full-network VBLL: collects KL from ALL
    VBLLLayers in the model. Observation noise comes from the last VBLLLayer
    (the output layer).
    """
    def __init__(self, model, n_data):
        super(VBLLLoss, self).__init__()
        self.model = model
        self.n_data = n_data

        # Collect ALL VBLLLayers in the model
        self.vbll_layers = [m for m in model.modules() if isinstance(m, VBLLLayer)]
        # Observation noise comes from the last (output) VBLLLayer
        self.output_layer = self.vbll_layers[-1] if self.vbll_layers else None

    def forward(self, pred, target):
        if self.output_layer is not None:
            noise_var = self.output_layer.noise_var
            # Gaussian NLL with learned observation noise
            nll = 0.5 * torch.log(noise_var) + 0.5 * ((pred - target) ** 2) / noise_var
            nll = nll.mean()
            # Sum KL from all variational layers
            kl = sum(layer.kl_divergence() for layer in self.vbll_layers)
            return nll + kl / self.n_data
        # Fallback to MSE if no VBLLLayer found
        return nn.MSELoss()(pred, target)


def apply_bayesian_transformation_last_layer_variational(model):
    """
    Converts the last Linear layer of a PyTorch model to a VBLLLayer
    (Variational Bayesian Last Layer, Harrison 2024) while keeping the rest
    of the model deterministic.

    The VBLLLayer maintains a variational posterior over the last-layer weights
    and is trained with an ELBO loss (MSE + KL divergence).

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to be transformed.

    Returns
    -------
    model : nn.Module
        The transformed model with the last layer replaced by a VBLLLayer.
    """
    last_linear_name = None
    last_linear_module = None

    # Identify the last nn.Linear layer
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Linear):
            last_linear_name = name
            last_linear_module = module
            break

    if last_linear_module is None:
        raise ValueError("No nn.Linear layer found to replace.")

    # Create VBLLLayer with same dimensions
    vbll_layer = VBLLLayer(
        in_features=last_linear_module.in_features,
        out_features=last_linear_module.out_features
    )

    # Initialize weight_mu from pretrained weights
    with torch.no_grad():
        vbll_layer.weight_mu.copy_(last_linear_module.weight.data)
        if last_linear_module.bias is not None:
            vbll_layer.bias_mu.copy_(last_linear_module.bias.data)

    # Helper for recursive attribute setting
    def set_nested_attr(obj, attr_path, value):
        attrs = attr_path.split(".")
        for a in attrs[:-1]:
            obj = getattr(obj, a)
        setattr(obj, attrs[-1], value)

    # Replace in the model
    set_nested_attr(model, last_linear_name, vbll_layer)

    return model

def apply_bayesian_transformation_full_variational(model):
    """
    Converts ALL Linear layers in a PyTorch model to VBLLLayers.

    This is the full-network analogue of apply_bayesian_transformation_last_layer_variational:
    every nn.Linear gets a variational posterior over its weights, trained with ELBO.
    Only the final (output) VBLLLayer retains the learned observation noise parameter;
    hidden layers contribute epistemic uncertainty only.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to be transformed.

    Returns
    -------
    model : nn.Module
        The transformed model with all Linear layers replaced by VBLLLayers.
    """
    def set_nested_attr(obj, attr_path, value):
        attrs = attr_path.split(".")
        for a in attrs[:-1]:
            obj = getattr(obj, a)
        setattr(obj, attrs[-1], value)

    linear_layers = [(name, module) for name, module in model.named_modules()
                     if isinstance(module, nn.Linear)]

    if not linear_layers:
        raise ValueError("No nn.Linear layers found to replace.")

    for name, module in linear_layers:
        vbll_layer = VBLLLayer(
            in_features=module.in_features,
            out_features=module.out_features
        )
        with torch.no_grad():
            vbll_layer.weight_mu.copy_(module.weight.data)
            if module.bias is not None:
                vbll_layer.bias_mu.copy_(module.bias.data)
        set_nested_attr(model, name, vbll_layer)

    return model


# =============================================================================
# OUT-OF-FOLD SCORING OF THE TRAINING MOLECULES
# =============================================================================
#
# Corruption enters the TRAINING split. Until 2026-08-26 QM9 saved per-molecule
# uncertainty for test molecules only and `predict(x_train` appeared nowhere in
# this file, so no training molecule was ever predicted for and the question
# "does predicted uncertainty find the corrupted labels?" had no data behind it
# at all (RERUN_PLAN.md §2.6, §3.1).
#
# Scoring a training molecule with the model that fitted it does not answer the
# question either: it measures memorisation. A Gaussian process has zero
# posterior variance at its own training inputs and a forest has fitted those
# exact rows. Every training molecule therefore gets its prediction and its
# uncertainty from a model that never saw ITS label.
#
# ONE implementation, shared by every model family, mirroring `_oof_predict` in
# /Users/apunt/repos/KIRBy/tests/alternative_data_noise_robustness.py so the two
# producers cannot drift.


def oof_predict(fit_predict, X, y_noisy, n_folds, groups=None, seed=42, label=''):
    """Out-of-fold predictions and uncertainties over the molecules a model fits.

    Splits the fit rows into `n_folds` parts, fits on the rest and scores the
    held-out part, rotating.

    The noise is injected ONCE, by the injector, before any of this, so a molecule
    carries the same corruption in whichever fold it lands.

    `groups` are the Murcko-scaffold group ids of the fit rows. When supplied the
    inner split is scaffold-grouped, matching the outer split: without it the fit
    set contains close analogues of every held-out molecule, so out-of-fold
    uncertainties come from an interpolation regime while the test set is an
    extrapolation regime, and the two are not on the same scale.

    `fit_predict(X_fit, y_fit, X_score) -> (mean, std_or_None)`.

    Returns (oof_mean, oof_unc, n_folds_ok). The caller MUST check n_folds_ok: a
    silently truncated out-of-fold pass looks exactly like a complete one in the
    output file.
    """
    n = len(y_noisy)
    oof_mean = np.full(n, np.nan)
    oof_unc = np.full(n, np.nan)

    tag = f"[oof{(' ' + label) if label else ''}]"

    n_groups = len(np.unique(groups)) if groups is not None else 0
    if groups is not None and n_groups >= n_folds:
        print(f"      {tag} scaffold-grouped inner split: {n_folds} folds over "
              f"{n_groups} scaffold groups, {n} molecules", flush=True)
        splitter = GroupKFold(n_splits=n_folds)
        folds = [(tr, te) for tr, te in splitter.split(X, y_noisy, groups)]
    else:
        # Both fallback branches say so. A fallback nobody is told about is how a
        # grouped inner split silently becomes a random one.
        if groups is None:
            print(f"      {tag} FALLBACK no scaffold groups were supplied — using a "
                  f"deterministic random split of {n} molecules into {n_folds} folds. "
                  f"Out-of-fold uncertainty is then measured in an interpolation "
                  f"regime and is not on the same footing as the test set.",
                  flush=True)
        else:
            print(f"      {tag} FALLBACK only {n_groups} scaffold groups for {n_folds} "
                  f"folds — using a deterministic random split instead.", flush=True)
        order = np.random.RandomState(seed).permutation(n)
        parts = np.array_split(order, n_folds)
        folds = [(np.setdiff1d(order, held), held) for held in parts]

    n_ok = 0
    for keep, held in folds:
        try:
            m, u = fit_predict(X[keep], y_noisy[keep], X[held])
        except Exception as e:
            print(f"      {tag} fold failed: {type(e).__name__}: {e}", flush=True)
            continue
        oof_mean[held] = np.asarray(m, dtype=float).ravel()
        if u is not None:
            oof_unc[held] = np.asarray(u, dtype=float).ravel()
        n_ok += 1
    if n_ok < len(folds):
        print(f"      {tag} WARNING {n_ok}/{len(folds)} inner folds succeeded",
              flush=True)
    return oof_mean, oof_unc, n_ok


def _fill_non_finite(values):
    """Replace non-finite entries with the mean of the finite ones.

    A failed inner fold leaves NaN. The NaN stays in what is WRITTEN, so a reader
    can see which molecules were not scored; this fill exists only so the shape
    recomputed from the prediction is defined everywhere. Same policy as KIRBy.
    """
    v = np.asarray(values, dtype=float)
    finite = np.isfinite(v)
    fill = float(v[finite].mean()) if finite.any() else 0.0
    return np.where(finite, v, fill)


def score_training_molecules_out_of_fold(
        fit_predict, x_fit, y_fit, train_noise, args, s, rep, iteration,
        iteration_seed, file_no, model_name,
        train_slice=slice(None), val_slice=None,
        y_pred_std_calibrated=None, temperature=None,
        epistemic_from=None, restore_torch_rng=False):
    """Score the molecules a model fitted, out of fold, and write their rows.

    `train_slice` / `val_slice` say which provenance rows `x_fit` is made of, in
    `vstack((x_train, x_val))` order.

    `val_slice` defaults to None — no validation rows — because since 2026-08-27
    **no model stacks validation into its training set** (RERUN_PLAN.md §2.12).
    It used to differ by family, and three callers went on passing the slice that
    described the old regime after the merge was removed: the forest took the
    `slice(None)` default and NGBoost and the Gauche GP passed half of validation,
    so each asked the provenance for molecules its model no longer fitted. The
    guard below caught it rather than mis-attributing the noise, which is what it
    is for. The default is None so that forgetting is the safe case; a family that
    genuinely fits some of validation must say so explicitly.

    UNITS. Predictions and uncertainties are written in the model's own
    (standardised) units, exactly as on a test row, so no column changes meaning
    between splits. `injected_noise`, `noise_scale`, `noise_pattern` and
    `noise_pattern_pred` are written in RAW label units, bit-identical to the
    injector's provenance file. `y_true_original` is the raw clean label and
    `y_true_noisy` is the standardised noisy label -- the same two meanings those
    columns already carry on a test row.

    Returns the number of inner folds that succeeded, or None when nothing was
    written.
    """
    if train_noise is None:
        return None
    n_folds = int(getattr(args, 'oof_folds', 0) or 0)
    if n_folds <= 1:
        return None
    if not getattr(args, 'uncertainty', False):
        return None

    rows = train_noise.rows(train_slice, val_slice)
    n = len(y_fit)
    if len(rows['epsilon_raw']) != n:
        raise RuntimeError(
            f"out-of-fold scoring for {model_name}: the model fits {n} rows but the "
            f"recorded noise covers {len(rows['epsilon_raw'])}. The two must be the "
            f"same molecules in the same order — anything else attributes one "
            f"molecule's noise to another, which is the original QM9 defect.")

    groups = rows['group']
    if np.any(groups < 0):
        # -1 is 'this molecule was not in the scaffold group map'. Refuse rather
        # than group every one of them together under the same id.
        print(f"      [oof {model_name}] {int(np.sum(groups < 0))} of {n} molecules "
              f"are missing from the scaffold group map; falling back to an "
              f"ungrouped split.", flush=True)
        groups = None

    # Neural training consumes the GLOBAL torch generator (weight initialisation
    # and a shuffled DataLoader). Without this snapshot the extra out-of-fold fits
    # would advance the stream, so the MAIN model at every later noise level would
    # be initialised differently and this job's R2 would silently disagree with a
    # run made without --oof-folds. KIRBy does the same thing.
    if restore_torch_rng:
        _tstate = torch.get_rng_state()
        _cstate = (torch.cuda.get_rng_state_all()
                   if torch.cuda.is_available() else None)
    try:
        oof_mean, oof_unc, n_ok = oof_predict(
            fit_predict, np.asarray(x_fit), np.asarray(y_fit), n_folds,
            groups=groups, seed=iteration_seed, label=model_name)
    finally:
        if restore_torch_rng:
            torch.set_rng_state(_tstate)
            if _cstate is not None:
                torch.cuda.set_rng_state_all(_cstate)

    if n_ok == 0:
        print(f"      [oof {model_name}] every inner fold failed — writing no "
              f"train_oof rows rather than a block of blanks.", flush=True)
        return 0
    if not np.isfinite(oof_unc).any():
        print(f"      [oof {model_name}] the inner fits produced no per-molecule "
              f"uncertainty — writing no train_oof rows.", flush=True)
        return n_ok

    # The sham ceiling: the level-free shape recomputed from what the model
    # PREDICTED rather than from the true label. Computed HERE, after the
    # out-of-fold values exist. The experimental pipeline put the identical block
    # ABOVE the block that fills them in, so its guard was false on every pass of
    # every loop and the column came out empty on every training row it ever wrote.
    pattern_pred = train_noise.pattern_pred_from_standardised(
        _fill_non_finite(oof_mean), rows['noise_pattern_raw'])

    save_uncertainty_values(
        y_pred_mean=oof_mean,
        y_pred_std=oof_unc,
        y_true_original=rows['y_clean_raw'],
        y_true_noisy=rows['y_written'],
        filepath=args.filepath,
        model_name=model_name,
        rep=rep,
        sigma_noise=s,
        iteration=iteration,
        file_no=file_no,
        y_pred_std_calibrated=y_pred_std_calibrated,
        temperature=temperature,
        epistemic_uncertainty=epistemic_from,
        aleatoric_uncertainty=None,
        split='train_oof',
        injected_noise=rows['epsilon_raw'],
        canonical_smiles=rows['canonical_smiles'],
        noise_scale=rows['noise_scale_raw'],
        noise_pattern=rows['noise_pattern_raw'],
        noise_pattern_pred=pattern_pred,
        oof_folds_ok=n_ok,
        noise_type=getattr(train_noise, 'noise_type', None),
    )
    scored = int(np.isfinite(oof_mean).sum())
    print(f"      [oof {model_name}] wrote {n} train_oof rows "
          f"({scored} scored, {n - scored} left NaN by a failed fold), "
          f"{n_ok}/{n_folds} inner folds ok", flush=True)
    return n_ok


def _held_out_noise_columns(train_noise, n_rows, y_pred=None):
    """The recorded columns for held-out molecules, as keyword arguments.

    `noise_scale` and `injected_noise` are exactly zero on a held-out molecule --
    test labels are never corrupted. `noise_pattern` is NOT zero: it is the
    level-free shape that molecule's region would receive, computed against the
    TRAINING distribution's cut-points. Passing it is what lets the analysis ask
    whether a model becomes less certain where the data is unreliable, and lets the
    zero-level subtraction remove the label-magnitude confound.

    Returns {} when the run recorded no test rows, so every call site degrades to
    the previous behaviour rather than raising.
    """
    if train_noise is None:
        return {}
    rows = getattr(train_noise, 'test_rows', lambda: None)()
    if not rows:
        return {}
    if len(rows['canonical_smiles']) != n_rows:
        print(f"      [noise] {len(rows['canonical_smiles'])} recorded held-out "
              f"molecules against {n_rows} scored rows -- not writing the shape "
              f"columns rather than lining up the wrong molecules.", flush=True)
        return {}
    out = {
        'canonical_smiles': rows['canonical_smiles'],
        'noise_scale': rows['noise_scale_raw'],
        'noise_pattern': rows['noise_pattern_raw'],
        'noise_type': getattr(train_noise, 'noise_type', None),
    }
    # The sham ceiling on held-out molecules: the same shape recomputed from what
    # the model PREDICTED. If uncertainty tracks that as closely as it tracks the
    # real shape, the model is following its own prediction and has learned nothing
    # about where the data is unreliable. The experimental pipeline writes this
    # column for held-out rows and QM9 did not, so the two could not be compared.
    if y_pred is not None:
        try:
            out['noise_pattern_pred'] = train_noise.pattern_pred_from_standardised(
                np.asarray(y_pred, dtype=float).ravel(), rows['noise_pattern_raw'])
        except Exception as exc:
            print(f"      [noise] could not recompute the shape from the held-out "
                  f"predictions ({type(exc).__name__}: {exc}) -- leaving the column "
                  f"blank rather than writing something else under its name.",
                  flush=True)
    return out


def train_rf_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, model_type, file_no, y_test_original, trial=None, train_noise=None):
    from quantile_forest import RandomForestQuantileRegressor
    params = {}
    params_source = 'default'

    if hasattr(args, 'use_best_params') and args.use_best_params and not args.tuning:
        best_params = load_best_hyperparameters('rf', rep)
        if best_params is not None:
            params = best_params
            params_source = 'tuned'
            print(f"Using tuned hyperparameters for rf-{rep}")
    
    if not params:
        if args.tuning:
            use_default_max_depth = trial.suggest_categorical('use_default_max_depth', [True, False])
            if use_default_max_depth:
                params['max_depth'] = None
            else:
                params['max_depth'] = trial.suggest_int('max_depth', 10, 200)

            params['max_features'] = trial.suggest_categorical('max_features', ['sqrt', 1.0, None])
            params['min_samples_leaf'] = trial.suggest_int('min_samples_leaf', 1, 50)
            params['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 20)
            params['n_estimators'] = trial.suggest_int('n_estimators', 10, 2000)
            params['bootstrap'] = trial.suggest_categorical('bootstrap', [True, False])
            params_source = 'tuning_trial'
        else:
            # Shared with the experimental pipeline via models/model_defaults.py.
            # Do not restate the numbers here -- that is how the two drifted.
            params = sklearn_params('qrf' if model_type == 'qrf' else 'rf')
            params_source = 'default'

    if model_type == 'rf':
        model = RandomForestRegressor(random_state=iteration_seed, **params)
    elif model_type == 'qrf':
        quantile = trial.suggest_float('quantile', 0.1, 0.9) if args.tuning else 0.5
        model = RandomForestQuantileRegressor(random_state=iteration_seed, **params)
        if trial is not None:
            trial.set_user_attr("quantile", quantile)

    # Settled 2026-08-27 by the author: NO model stacks validation into its
    # training set. It used to differ by model family -- forest, SVM, XGBoost and
    # LightGBM took all of validation (90% of the data), NGBoost and the Gauche GP
    # took half (85%), the neural models took none (80%) -- so the ANOVA's model
    # factor confounded model family with training-set size, and nothing in the
    # results row recorded which regime produced it (RERUN_PLAN.md 2.12).
    # Validation stays held out, for early stopping and for calibration.

    model.fit(x_train, y_train)

    if model_type == 'qrf':
        q16, q50, q84 = model.predict(x_test, quantiles=[0.16, 0.5, 0.84]).T
        y_pred = q50
        y_pred_mean = q50
        std_est = (q84 - q16) / 2  # IQR-based std estimate
        
        if args.uncertainty:
            # Decompose distributional uncertainty
            epistemic, aleatoric, total = decompose_uncertainty_distributional(
                y_pred_mean, std_est, model_type='qrf', is_variance=False
            )
            
            # QRF doesn't get temperature calibration (quantiles are already calibrated)
            temperature = None
            y_pred_std_calibrated = std_est
            
            save_uncertainty_values(
                y_pred_mean=y_pred_mean,
                y_pred_std=std_est,
                y_true_original=y_test_original,
                y_true_noisy=y_test,
                filepath=args.filepath,
                model_name=model_type,
                rep=rep,
                sigma_noise=s,
                iteration=iteration,
                file_no=file_no,
                y_pred_std_calibrated=y_pred_std_calibrated,
                temperature=temperature,
                epistemic_uncertainty=epistemic,
                aleatoric_uncertainty=aleatoric,
                split='test',
                **_held_out_noise_columns(train_noise, len(y_pred_mean), y_pred_mean),
            )

            # The TRAINING molecules, scored by forests that never saw their
            # labels. `val_slice=None` because validation is NOT fitted: the
            # author settled on 2026-08-27 that no model stacks it into training
            # (the branch above, RERUN_PLAN.md 2.12). This call used to take the
            # default `slice(None)`, which was right while the forest fitted
            # train+validation and wrong the moment it stopped -- it asked the
            # provenance for every validation row the model no longer sees.
            def _fp(x_fit, y_fit, x_score):
                inner = RandomForestQuantileRegressor(
                    random_state=iteration_seed, **params)
                inner.fit(x_fit, y_fit)
                iq16, iq50, iq84 = inner.predict(
                    x_score, quantiles=[0.16, 0.5, 0.84]).T
                return iq50, (iq84 - iq16) / 2

            score_training_molecules_out_of_fold(
                _fp, x_train, y_train, train_noise, args, s, rep, iteration,
                iteration_seed, file_no, model_type, val_slice=None)
    else:
        y_pred = model.predict(x_test)

    metrics = calculate_regression_metrics(y_test, y_pred, logging=True)
    save_results(args.filepath, s, iteration, model_type, rep, args.sample_size, metrics, params_source)

    return metrics[3]

def train_svm_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, trial=None, train_noise=None):
    params = {}
    params_source = 'default'

    if hasattr(args, 'use_best_params') and args.use_best_params and not args.tuning:
        best_params = load_best_hyperparameters('svm', rep)
        if best_params is not None:
            params = best_params
            params_source = 'tuned'
            print(f"Using tuned hyperparameters for svm-{rep}")

    if not params:
        if args.tuning:
            params['C'] = trial.suggest_float('C', 0, 100)
            params['gamma'] = trial.suggest_categorical('gamma', ['scale', 'auto'])
            params['kernel'] = trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid'])

            if params['kernel'] == 'poly':
                params['degree'] = trial.suggest_int('degree', 2, 5)
                params['coef0'] = trial.suggest_float('coef0', 0.0, 10.0)

            if params['kernel'] == 'sigmoid':
                params['coef0'] = trial.suggest_float('coef0', 0.0, 10.0)
            params_source = 'tuning_trial'
        else:
            params = sklearn_params('svm')
            params_source = 'default'

    # Settled 2026-08-27 by the author: NO model stacks validation into its
    # training set. It used to differ by model family -- forest, SVM, XGBoost and
    # LightGBM took all of validation (90% of the data), NGBoost and the Gauche GP
    # took half (85%), the neural models took none (80%) -- so the ANOVA's model
    # factor confounded model family with training-set size, and nothing in the
    # results row recorded which regime produced it (RERUN_PLAN.md 2.12).
    # Validation stays held out, for early stopping and for calibration.

    model = SVR(**params)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    metrics = calculate_regression_metrics(y_test, y_pred, logging=True)

    save_results(args.filepath, s, iteration, 'svm', rep, args.sample_size, metrics,
                 params_source)

    return metrics[3]

def train_ngboost_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, file_no, y_test_original, trial=None, train_noise=None):
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
    from ngboost.scores import MLE
    
    params = {}
    params_source = 'default'
    
    if hasattr(args, 'use_best_params') and args.use_best_params and not args.tuning:
        best_params = load_best_hyperparameters('ngboost', rep)
        if best_params is not None:
            params = best_params
            params_source = 'tuned'
            print(f"Using tuned hyperparameters for ngboost-{rep}")
    
    if not params:
        if args.tuning:
            params['learning_rate'] = trial.suggest_float('learning_rate', 0.001, 0.2, log=True)
            params['n_estimators'] = trial.suggest_int('n_estimators', 10, 2000)
            params['natural_gradient'] = trial.suggest_categorical('natural_gradient', [True, False])
            params_source = 'tuning_trial'
        else:
            params = sklearn_params('ngboost')
            params_source = 'default'
    
    # STEP 1: Split validation for calibration
    if args.uncertainty:
        # Settled 2026-08-27 by the author: NO model stacks validation into its
        # training set. It used to differ by model family -- forest, SVM, XGBoost and
        # LightGBM took all of validation (90% of the data), NGBoost and the Gauche GP
        # took half (85%), the neural models took none (80%) -- so the ANOVA's model
        # factor confounded model family with training-set size, and nothing in the
        # results row recorded which regime produced it (RERUN_PLAN.md 2.12).
        # Validation stays held out, for early stopping and for calibration.
        x_val_train = x_train
        y_val_train = y_train
        x_val_cal = x_val
        y_val_cal = y_val
    else:
        # Same split with --uncertainty off: validation is never trained on.
        x_val_train = x_train
        y_val_train = y_train
    
    # Dist and Score come from the shared spec by NAME, so the experimental
    # pipeline resolves the same two classes. A tuned-parameter dict has neither
    # key, so fall back to the spec's own values.
    _ngb_dist = {'Normal': Normal}[params.get('dist', SKLEARN_DEFAULTS['ngboost']['dist'])]
    _ngb_score = {'MLE': MLE}[params.get('score', SKLEARN_DEFAULTS['ngboost']['score'])]
    model = NGBRegressor(
        Dist=_ngb_dist,
        Score=_ngb_score,
        natural_gradient=params['natural_gradient'],
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
        verbose=False,
        random_state=iteration_seed,
    )
    model.fit(x_val_train, y_val_train)
    
    # STEP 2: Get predictions and calibrate
    y_pred = model.predict(x_test)
    y_dist = model.pred_dist(x_test)
    y_pred_std_uncalibrated = y_dist.scale
    
    if args.uncertainty:
        # Get calibration predictions
        y_cal_pred = model.predict(x_val_cal)
        y_cal_dist = model.pred_dist(x_val_cal)
        y_cal_pred_std = y_cal_dist.scale
        
        # Find temperature
        temperature = calibrate_uncertainty_simple(y_cal_pred, y_cal_pred_std, y_val_cal)
        y_pred_std_calibrated = y_pred_std_uncalibrated * temperature
        
        # Decompose distributional uncertainty
        epistemic, aleatoric, total = decompose_uncertainty_distributional(
            y_pred, y_pred_std_uncalibrated, model_type='ngboost', is_variance=False
        )
        
        # Apply calibration to aleatoric
        if aleatoric is not None:
            aleatoric = aleatoric * temperature
        
    else:
        temperature = None
        y_pred_std_calibrated = None
        epistemic = None
        aleatoric = None
    
    metrics = calculate_regression_metrics(y_test, y_pred, logging=True)
    save_results(args.filepath, s, iteration, 'ngboost', rep, args.sample_size, metrics, params_source)
    
    # *** UPDATED: Save with decomposition ***
    if args.uncertainty:
        save_uncertainty_values(
            y_pred_mean=y_pred,
            y_pred_std=y_pred_std_uncalibrated,
            y_true_original=y_test_original,
            y_true_noisy=y_test,
            filepath=args.filepath,
            model_name="ngboost",
            rep=rep,
            sigma_noise=s,
            iteration=iteration,
            file_no=file_no,
            y_pred_std_calibrated=y_pred_std_calibrated,
            temperature=temperature,
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            split='test',
            **_held_out_noise_columns(train_noise, len(y_test), y_pred),
        )

        # The TRAINING molecules. `val_slice=None` because `x_val_train` IS
        # `x_train` -- validation is held out for calibration and never fitted
        # (the branch above, settled 2026-08-27, RERUN_PLAN.md 2.12). This used to
        # pass the first half of validation, which is what NGBoost fitted under
        # the old regime; when the merge went, the slice stayed, and the
        # provenance then covered 250 molecules the model never saw.

        def _fp(x_fit, y_fit, x_score):
            inner = NGBRegressor(
                Dist=_ngb_dist,
                Score=_ngb_score,
                natural_gradient=params['natural_gradient'],
                n_estimators=params['n_estimators'],
                learning_rate=params['learning_rate'],
                verbose=False,
                random_state=iteration_seed,
            )
            inner.fit(x_fit, y_fit)
            dist = inner.pred_dist(x_score)
            return dist.loc, dist.scale

        score_training_molecules_out_of_fold(
            _fp, x_val_train, y_val_train, train_noise, args, s, rep, iteration,
            iteration_seed, file_no, 'ngboost', val_slice=None)

    return metrics[3]

def train_xgboost_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, trial=None, train_noise=None):
    from xgboost import XGBRegressor
    params = {}
    params_source = 'default'

    if hasattr(args, 'use_best_params') and args.use_best_params and not args.tuning:
        best_params = load_best_hyperparameters('xgboost', rep)
        if best_params is not None:
            params = best_params
            params_source = 'tuned'
            print(f"Using tuned hyperparameters for xgboost-{rep}")

    if not params:
        if args.tuning:
            use_default_max_depth = trial.suggest_categorical('use_default_max_depth', [True, False])
            if use_default_max_depth:
                params['max_depth'] = None
            else:
                params['max_depth'] = trial.suggest_int('max_depth', 2, 20)

            use_default_learning_rate = trial.suggest_categorical('use_default_learning_rate', [True, False])
            if use_default_learning_rate:
                params['learning_rate'] = None
            else:
                params['learning_rate'] = trial.suggest_float('learning_rate', 0.001, 0.2, log=True)

            params['subsample'] = trial.suggest_float('subsample', 0.5, 1.0)
            params['n_estimators'] = trial.suggest_int('n_estimators', 10, 2000)
            params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.5, 1.0)
            params['colsample_bylevel'] = trial.suggest_float('colsample_bylevel', 0.5, 1.0)
            params['min_child_weight'] = trial.suggest_int('min_child_weight', 1, 10)
            params['gamma'] = trial.suggest_float('gamma', 0, 5.0)
            params['reg_alpha'] = trial.suggest_float('reg_alpha', 0.0, 1.0)
            params['reg_lambda'] = trial.suggest_float('reg_lambda', 0.0, 1.0)
            params_source = 'tuning_trial'
        else:
            params = sklearn_params('xgboost')
            params_source = 'default'

    # Settled 2026-08-27 by the author: NO model stacks validation into its
    # training set. It used to differ by model family -- forest, SVM, XGBoost and
    # LightGBM took all of validation (90% of the data), NGBoost and the Gauche GP
    # took half (85%), the neural models took none (80%) -- so the ANOVA's model
    # factor confounded model family with training-set size, and nothing in the
    # results row recorded which regime produced it (RERUN_PLAN.md 2.12).
    # Validation stays held out, for early stopping and for calibration.

    model = XGBRegressor(random_state=iteration_seed, **params)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    metrics = calculate_regression_metrics(y_test, y_pred, logging=True)

    save_results(args.filepath, s, iteration, 'xgboost', rep, args.sample_size, metrics,
                 params_source)

    return metrics[3]


def train_lgb_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, trial=None, train_noise=None):
    import lightgbm as lgb
    params = {}
    params_source = 'default'

    if hasattr(args, 'use_best_params') and args.use_best_params and not args.tuning:
        best_params = load_best_hyperparameters('lgb', rep)
        if best_params is not None:
            params = best_params
            params_source = 'tuned'
            print(f"Using tuned hyperparameters for lgb-{rep}")

    if not params:
        if args.tuning:
            params['n_estimators'] = trial.suggest_int('n_estimators', 50, 2000)
            params['learning_rate'] = trial.suggest_float('learning_rate', 0.005, 0.3, log=True)
            params['num_leaves'] = trial.suggest_int('num_leaves', 15, 127)
            params['max_depth'] = trial.suggest_int('max_depth', 3, 15)
            params['subsample'] = trial.suggest_float('subsample', 0.5, 1.0)
            params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.5, 1.0)
            params['min_child_samples'] = trial.suggest_int('min_child_samples', 5, 50)
            params['reg_alpha'] = trial.suggest_float('reg_alpha', 0.0, 1.0)
            params['reg_lambda'] = trial.suggest_float('reg_lambda', 0.0, 1.0)
            params_source = 'tuning_trial'
        else:
            params = sklearn_params('lightgbm')
            params_source = 'default'

    # Settled 2026-08-27 by the author: NO model stacks validation into its
    # training set. It used to differ by model family -- forest, SVM, XGBoost and
    # LightGBM took all of validation (90% of the data), NGBoost and the Gauche GP
    # took half (85%), the neural models took none (80%) -- so the ANOVA's model
    # factor confounded model family with training-set size, and nothing in the
    # results row recorded which regime produced it (RERUN_PLAN.md 2.12).
    # Validation stays held out, for early stopping and for calibration.

    model = lgb.LGBMRegressor(random_state=iteration_seed, verbose=-1, **params)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    metrics = calculate_regression_metrics(y_test, y_pred, logging=True)

    save_results(args.filepath, s, iteration, 'lgb', rep, args.sample_size, metrics,
                 params_source)

    return metrics[3]


def init_rbf_lengthscale(model, train_x):
    """Start the RBF lengthscale at the median distance between training molecules.

    gpytorch starts it at softplus(0), about 0.69. Measured 2026-08-26, typical
    distances between molecules run from 17 on PDV to 1,100 on the learned
    embeddings. At 0.69 every molecule looks infinitely far from every other, the
    kernel matrix is the identity, the marginal likelihood is flat, and neither
    botorch nor Adam has a gradient to follow. The fit returns a constant.

    That is what produced R2 = -0.0158 for MHG-GNN and +0.0087 for mol2vec in
    results/gp_kernel_harvest/qm9/, and it was read as the representations being
    unusable. Started here instead, the SAME features reach 0.68 to 0.76.

    Only the RBF kernel has a lengthscale. Tanimoto is a ratio and has none, so
    this is a no-op there.
    """
    if not GP_DEFAULTS['init_lengthscale_from_data']:
        return None

    base = getattr(model.covar_module, 'base_kernel', None)
    if base is None or not hasattr(base, 'lengthscale') or base.lengthscale is None:
        return None

    with torch.no_grad():
        n = min(GP_DEFAULTS['lengthscale_probe_n'], len(train_x))
        probe = train_x[torch.randperm(len(train_x))[:n]]
        median = torch.cdist(probe, probe).flatten().median().item()

    if not np.isfinite(median) or median <= 0:
        print("[gp] median distance is not usable; leaving the lengthscale at its default")
        return None

    base.lengthscale = median
    print(f"[gp] lengthscale started at the median distance between molecules: {median:.4g}")
    return median


def gp_fit_collapsed(y_pred, y_train):
    """Did the fit give up and predict one constant?

    A Gaussian process that cannot use its features returns the mean everywhere.
    That still produces a number, and the number looks like a bad result rather
    than a failed fit -- which is how the two learned embeddings were written off.
    Never let it pass silently.
    """
    spread = float(np.std(y_pred))
    threshold = GP_DEFAULTS['collapse_fraction'] * float(np.std(y_train))
    collapsed = spread < threshold
    if collapsed:
        print(f"[gp] WARNING: the fit COLLAPSED -- predictions vary by {spread:.4g} "
              f"against a training label spread of {float(np.std(y_train)):.4g}. "
              f"This is a failed fit, not a poor representation.")
    return collapsed, spread


def fit_gp_with_fallback(mll, model, likelihood, train_x, train_y):
    """Fit a GP marginal likelihood, and say which optimiser did it.

    botorch from about 0.12 requires its fitter's argument to be a botorch
    Model -- it calls transform_inputs -- and `Gauche` is a plain gpytorch
    ExactGP, so on a newer botorch this raises AttributeError. The experimental
    pipeline already fell back to a plain gpytorch Adam loop; QM9 did not, so
    the SAME environment that merely changed the experimental fit would kill
    every QM9 GP job outright.

    Both pipelines now use this same try/except and both record the answer, so a
    GP row always says which optimiser produced it.

    Returns 'botorch' or 'adam_fallback'.
    """
    init_rbf_lengthscale(model, train_x)
    try:
        with gp_fit_threads():
            fit_gpytorch_model(mll)
        return 'botorch'
    except Exception as exc:
        print(f"[gp] botorch fitter refused this model "
              f"({type(exc).__name__}: {exc}); using the Adam fallback. "
              f"Recorded as gp_fit_method=adam_fallback.")
        model.train()
        likelihood.train()
        opt = torch.optim.Adam(model.parameters(),
                               lr=GP_DEFAULTS['fallback_adam_lr'])
        with gp_fit_threads():
            for _ in range(GP_DEFAULTS['fallback_adam_iters']):
                opt.zero_grad()
                loss = -mll(model(train_x), train_y)
                loss.backward()
                opt.step()
        return 'adam_fallback'


def train_gauche_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, file_no, y_test_original, trial=None, train_noise=None):
    params = {}
    params_source = 'default'
    if hasattr(args, 'use_best_params') and args.use_best_params and not args.tuning:
        best_params = load_best_hyperparameters('gauche', rep)
        if best_params is not None:
            params = best_params
            params_source = 'tuned'
            print(f"Using tuned hyperparameters for gauche-{rep}")
    
    if not params:
        if args.tuning:
            params['kernel_name'] = trial.suggest_categorical('kernel', [
                'Tanimoto', 'BraunBlanquet', 'Dice', 'Faith', 'Forbes',
                'InnerProduct', 'Intersection', 'MinMax', 'Otsuka',
                'Rand',
            ])
            params['outputscale'] = trial.suggest_float('outputscale', 0.1, 10.0, log=True)
            params['likelihood_noise'] = trial.suggest_float('likelihood_noise', 1e-4, 0.1, log=True)
            params_source = 'tuning_trial'
        else:
            kernel_cli = getattr(args, 'kernel', 'tanimoto').capitalize()
            if kernel_cli == 'Rbf':
                kernel_cli = 'RBF'
            params['kernel_name'] = kernel_cli
            params['outputscale'] = GP_DEFAULTS['outputscale']
            params['likelihood_noise'] = GP_DEFAULTS['likelihood_noise']
            params_source = 'default'

    kernel_map = {
        'Tanimoto': gauche.kernels.fingerprint_kernels.tanimoto_kernel.TanimotoKernel,
        'BraunBlanquet': gauche.kernels.fingerprint_kernels.braun_blanquet_kernel.BraunBlanquetKernel,
        'Dice': gauche.kernels.fingerprint_kernels.dice_kernel.DiceKernel,
        'Faith': gauche.kernels.fingerprint_kernels.faith_kernel.FaithKernel,
        'Forbes': gauche.kernels.fingerprint_kernels.forbes_kernel.ForbesKernel,
        'InnerProduct': gauche.kernels.fingerprint_kernels.inner_product_kernel.InnerProductKernel,
        'Intersection': gauche.kernels.fingerprint_kernels.intersection_kernel.IntersectionKernel,
        'MinMax': gauche.kernels.fingerprint_kernels.minmax_kernel.MinMaxKernel,
        'Otsuka': gauche.kernels.fingerprint_kernels.otsuka_kernel.OtsukaKernel,
        'Rand': gauche.kernels.fingerprint_kernels.rand_kernel.RandKernel,
        'RBF': gpytorch.kernels.RBFKernel,
    }

    # STEP 1: Split validation for calibration
    if args.uncertainty:
        # Settled 2026-08-27 by the author: NO model stacks validation into its
        # training set. It used to differ by model family -- forest, SVM, XGBoost and
        # LightGBM took all of validation (90% of the data), NGBoost and the Gauche GP
        # took half (85%), the neural models took none (80%) -- so the ANOVA's model
        # factor confounded model family with training-set size, and nothing in the
        # results row recorded which regime produced it (RERUN_PLAN.md 2.12).
        # Validation stays held out, for early stopping and for calibration.
        x_train_full = x_train
        y_train_full = y_train
        x_val_cal = x_val
        y_val_cal = y_val
    else:
        # Same split with --uncertainty off: validation is never trained on.
        x_train_full = x_train
        y_train_full = y_train

    x_train_tensor = torch.from_numpy(x_train_full).double()
    x_test_tensor = torch.from_numpy(x_test).double()
    y_train_tensor = torch.from_numpy(y_train_full).double()

    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise=params['likelihood_noise'])
    kernel_class = kernel_map[params['kernel_name']]
    model = Gauche(x_train_tensor, y_train_tensor, likelihood, kernel_class)
    # params['outputscale'] used to be computed here and never applied, so the
    # ScaleKernel actually started at gpytorch's default of softplus(0) ~ 0.693
    # while the experimental pipeline set 1.0 "to match". Apply it on both sides.
    if GP_DEFAULTS['apply_outputscale']:
        model.covar_module.outputscale = params['outputscale']

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    gp_fit_method = fit_gp_with_fallback(mll, model, likelihood,
                                         x_train_tensor, y_train_tensor)

    model.eval()
    likelihood.eval()
    
    # STEP 2: Get predictions and calibrate
    with torch.no_grad():
        # Test predictions
        # `model(x)` is the LATENT f posterior; `likelihood(model(x))` is the
        # PREDICTIVE distribution of an observation, which is the latent variance
        # PLUS the likelihood noise. The reported std used to be the latent one,
        # so it was epistemic-only -- while the very next block computed
        # `total = sqrt(posterior_variance + likelihood_noise)` and threw it away.
        # A coverage number is a statement about observations, so it needs the
        # predictive variance (RERUN_PLAN.md 2.13).
        test_preds = model(x_test_tensor)
        observed_preds = likelihood(test_preds)
        y_pred = observed_preds.mean.numpy()
        pred_vars = test_preds.variance.numpy()          # latent: the epistemic part
        y_pred_std_uncalibrated = np.sqrt(observed_preds.variance.numpy())

        if args.uncertainty:
            # Calibration predictions
            x_val_cal_tensor = torch.from_numpy(x_val_cal).double()
            cal_preds = likelihood(model(x_val_cal_tensor))
            y_cal_pred_mean = cal_preds.mean.numpy()
            y_cal_pred_std = np.sqrt(cal_preds.variance.numpy())
            
            # Find temperature
            temperature = calibrate_uncertainty_simple(y_cal_pred_mean, y_cal_pred_std, y_val_cal)
            y_pred_std_calibrated = y_pred_std_uncalibrated * temperature
            
            # Decompose GP uncertainty
            likelihood_noise = likelihood.noise.item()
            epistemic, aleatoric, total = decompose_uncertainty_gp(pred_vars, likelihood_noise)
            
            # Apply calibration to epistemic
            epistemic = epistemic * temperature
            aleatoric = aleatoric * temperature
            
        else:
            temperature = None
            y_pred_std_calibrated = None
            epistemic = None
            aleatoric = None

    model_name = 'gauche_rbf' if params['kernel_name'] == 'RBF' else 'gauche'
    # A fit that gave up and predicted one constant still produces a number, and
    # that number reads as a bad representation rather than a failed fit. Record it.
    collapsed, _ = gp_fit_collapsed(y_pred, y_train_full)
    metrics = calculate_regression_metrics(y_test, y_pred, logging=True)
    save_results(args.filepath, s, iteration, model_name, rep, args.sample_size,
                 metrics, gp_fit_method=gp_fit_method, gp_collapsed=int(collapsed))

    # *** UPDATED: Save with decomposition ***
    if args.uncertainty:
        save_uncertainty_values(
            y_pred_mean=y_pred,
            y_pred_std=y_pred_std_uncalibrated,
            y_true_original=y_test_original,
            y_true_noisy=y_test,
            filepath=args.filepath,
            model_name=model_name,
            rep=rep,
            sigma_noise=s,
            iteration=iteration,
            file_no=file_no,
            y_pred_std_calibrated=y_pred_std_calibrated,
            temperature=temperature,
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            split='test',
            **_held_out_noise_columns(train_noise, len(y_test), y_pred),
        )

        # The TRAINING molecules. A GP has ZERO posterior variance at its own
        # training inputs, so scoring them with the fitted model would report
        # confidence that says nothing about the labels -- this family is the
        # clearest case for why the out-of-fold pass exists at all.
        # `val_slice=None`: `x_train_full` IS `x_train`, because validation is no
        # longer stacked into training (the branch above, settled 2026-08-27,
        # RERUN_PLAN.md 2.12). The half-of-validation slice this used to pass was
        # the old regime's, and it outlived it.

        def _fp(x_fit, y_fit, x_score):
            xt = torch.from_numpy(np.asarray(x_fit)).double()
            yt = torch.from_numpy(np.asarray(y_fit)).double()
            xs = torch.from_numpy(np.asarray(x_score)).double()
            lik = gpytorch.likelihoods.GaussianLikelihood(
                noise=params['likelihood_noise'])
            inner = Gauche(xt, yt, lik, kernel_class)
            if GP_DEFAULTS['apply_outputscale']:
                inner.covar_module.outputscale = params['outputscale']
            inner_mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, inner)
            fit_gp_with_fallback(inner_mll, inner, lik, xt, yt)
            inner.eval()
            lik.eval()
            with torch.no_grad():
                preds = inner(xs)
                return (preds.mean.numpy(),
                        np.sqrt(np.clip(preds.variance.numpy(), 1e-12, None)))

        # A GP fit runs through torch; snapshot the global generator so the main
        # result is what it would have been without the out-of-fold pass.
        score_training_molecules_out_of_fold(
            _fp, x_train_full, y_train_full, train_noise, args, s, rep, iteration,
            iteration_seed, file_no, model_name, val_slice=None,
            restore_torch_rng=True)

    return metrics[3]

def train_nn(model, train_loader, val_loader, criterion, optimizer, device, args, s, iteration, file_no, model_name, rep, 
             patience=None, tolerance=None, domain_labels_train=None, domain_labels_val=None):
    """
    Added domain_labels parameters for domain-aware losses

    Early stopping is defined once, in models/model_defaults.py, and shared with
    the experimental pipeline. Three things changed on 2026-08-26 (Chat E):

      * the best weights are RESTORED. This used to count patience and return
        whatever the last epoch produced -- up to twenty epochs past the
        validation optimum, and under injected noise those are twenty epochs
        spent memorising corrupted labels. It made QM9's neural degradation
        curves steeper for a procedural reason, and the stochastic uncertainty
        passes were drawn from the overfitted weights.
      * the validation loss is a MEAN over batches, not a sum. It was a sum
        compared against an absolute tolerance of 0.01, so the improvement
        threshold silently scaled with the number of validation batches.
      * improvement is strict, and patience matches the experimental side.
    """
    if patience is None:
        patience = NEURAL_DEFAULTS['training']['patience']
    if tolerance is None:
        tolerance = NEURAL_DEFAULTS['training']['improvement_tolerance']
    restore_best = NEURAL_DEFAULTS['training']['restore_best_weights']

    model.to(device)
    best_loss = float('inf')
    best_state = None
    best_epoch = -1
    epochs_no_improve = 0
    
    train_losses = []
    val_losses = []
    
    # Check if loss needs domain labels
    needs_domains = isinstance(criterion, (DomainWeightedLoss, DomainBalancedLoss, HeteroscedasticPerDomainLoss))

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        n_train_batches = 0
        batch_idx = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            
            # Pass domain labels if needed
            if needs_domains and domain_labels_train is not None:
                batch_domains = domain_labels_train[batch_idx:batch_idx+len(X_batch)]
                batch_domains = torch.tensor(batch_domains, dtype=torch.long).to(device)
                loss = criterion(outputs, y_batch, batch_domains)
            else:
                loss = criterion(outputs, y_batch)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_train_batches += 1
            batch_idx += len(X_batch)

        # Both losses are reduced the same way. The validation loss became a
        # mean so the early-stopping threshold would stop depending on the
        # batch count; leaving the training loss as a sum would put the two
        # curves in save_per_epoch_metrics on different scales.
        if NEURAL_DEFAULTS['training']['val_loss_reduction'] == 'mean' and n_train_batches:
            train_loss /= n_train_batches

        # Validation
        model.eval()
        val_loss = 0
        n_val_batches = 0
        batch_idx = 0
        with torch.no_grad():
            for X_val, y_val in (val_loader or []):
                X_val, y_val = X_val.to(device), y_val.to(device)
                val_outputs = model(X_val)
                
                if needs_domains and domain_labels_val is not None:
                    batch_domains = domain_labels_val[batch_idx:batch_idx+len(X_val)]
                    batch_domains = torch.tensor(batch_domains, dtype=torch.long).to(device)
                    loss = criterion(val_outputs, y_val, batch_domains)
                else:
                    loss = criterion(val_outputs, y_val)
                
                val_loss += loss.item()
                n_val_batches += 1
                batch_idx += len(X_val)

        # Mean, not sum: a summed loss makes the improvement threshold depend on
        # how many validation batches there happen to be.
        if NEURAL_DEFAULTS['training']['val_loss_reduction'] == 'mean' and n_val_batches:
            val_loss /= n_val_batches

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Early stopping check. With no validation set there is nothing to stop
        # on, so train the full epoch budget rather than stopping on a constant.
        if n_val_batches == 0:
            pass
        elif val_loss < best_loss - tolerance:
            best_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            if restore_best:
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Roll back to the best epoch. Without this the model returned is whatever
    # the patience counter happened to stop on.
    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best weights from epoch {best_epoch} (val loss {best_loss:.6f})")
    
    if args.save_per_epoch_metrics:
        save_per_epoch_metrics(
            train_losses=train_losses,
            val_losses=val_losses,
            filepath=args.filepath,
            model_name=model_name,
            rep=rep,
            sigma_noise=s,
            iteration=iteration,
            file_no=file_no
        )

def train_dnn_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, file_no, y_test_original, trial=None, 
                   domain_labels_train=None, domain_labels_val=None, domain_labels_test=None, train_noise=None):
    params = {}
    params_source = 'default'

    if hasattr(args, 'use_best_params') and args.use_best_params and not args.tuning:
        best_params = load_best_hyperparameters('dnn', rep)
        if best_params is not None:
            params = best_params
            params_source = 'tuned'
            print(f"Using tuned hyperparameters for dnn-{rep}")

    if not params:
        if args.tuning:
            params['hidden_size1'] = trial.suggest_categorical('hidden_size1', [32, 64, 128, 256, 512, 1024, 2048, 4096])
            params['hidden_size2'] = trial.suggest_categorical('hidden_size2', [32, 64, 128, 256, 512, 1024, 2048, 4096])
            params['activation'] = trial.suggest_categorical('activation', ['relu', 'tanh', 'softmax'])
            params_source = 'tuning_trial'
        else:
            params['hidden_size1'], params['hidden_size2'] = NEURAL_DEFAULTS['dnn']['hidden_sizes']
            params['activation'] = NEURAL_DEFAULTS['dnn']['activation']
            params_source = 'default'

    activation_map = {'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'softmax': nn.Softmax(dim=1)}
    activation = activation_map[params['activation']]

    loss_name = args.loss if hasattr(args, 'loss') else 'mse'
    loss_kwargs = {}
    
    if args.tuning and trial is not None:
        # Tune loss-specific hyperparameters
        if loss_name == 'huber':
            loss_kwargs['delta'] = trial.suggest_float('loss_delta', 0.1, 5.0)
        elif loss_name == 'cauchy':
            loss_kwargs['c'] = trial.suggest_float('loss_c', 0.1, 5.0)
        elif loss_name == 'focal':
            loss_kwargs['gamma'] = trial.suggest_float('loss_gamma', 0.5, 5.0)
            loss_kwargs['alpha'] = trial.suggest_float('loss_alpha', 0.1, 0.9)
        elif loss_name == 'truncated':
            loss_kwargs['quantile'] = trial.suggest_float('loss_quantile', 0.7, 0.99)
            loss_kwargs['base_loss'] = trial.suggest_categorical('loss_base', ['mse', 'mae'])
        elif loss_name == 'barron':
            loss_kwargs['alpha_init'] = trial.suggest_float('loss_alpha_init', 0.5, 4.0)
            loss_kwargs['scale_init'] = trial.suggest_float('loss_scale_init', 0.1, 5.0)
        elif loss_name == 'evidential':
            loss_kwargs['coeff'] = trial.suggest_float('loss_coeff', 0.001, 1.0, log=True)
        elif loss_name in ['domain_weighted', 'domain_balanced', 'adaptive_domain']:
            loss_kwargs['base_loss'] = trial.suggest_categorical('loss_base', ['mse', 'mae', 'huber'])
            if loss_name == 'adaptive_domain':
                loss_kwargs['adaptation_rate'] = trial.suggest_float('loss_adapt_rate', 0.001, 0.1, log=True)
    
    # Parse additional loss parameters from command line if provided
    if hasattr(args, 'loss_params') and args.loss_params:
        import json
        cli_loss_kwargs = json.loads(args.loss_params)
        loss_kwargs.update(cli_loss_kwargs)
    
    # Check if this loss needs domain labels
    needs_domains = loss_name in ['domain_weighted', 'domain_balanced', 'het_per_domain', 'adaptive_domain', 'mixture_domain']
    
    # Validate domain clustering is enabled for domain-aware losses
    if needs_domains and (domain_labels_train is None or args.k_domains <= 1):
        print(f"Warning: {loss_name} requires domain clustering but k_domains={args.k_domains}. Falling back to MSE.")
        loss_name = 'mse'
        needs_domains = False
    
    # Add num_domains for domain-aware losses
    if needs_domains:
        loss_kwargs['num_domains'] = args.k_domains

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

    # STEP 1: Split validation data if Bayesian
    is_bayesian = args.bayesian_transformation is not None
    
    if is_bayesian and args.uncertainty:
        split_idx = len(x_val) // 2
        x_val_train = x_val[:split_idx]
        y_val_train = y_val[:split_idx]
        x_val_cal = x_val[split_idx:]
        y_val_cal = y_val[split_idx:]
        
        x_val_tensor = torch.tensor(x_val_train, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val_train, dtype=torch.float32).view(-1, 1).to(device)
        
        # Split domain labels too if they exist
        if domain_labels_val is not None:
            domain_labels_val_train = domain_labels_val[:split_idx]
            domain_labels_val_cal = domain_labels_val[split_idx:]
        else:
            domain_labels_val_train = None
            domain_labels_val_cal = None
    else:
        x_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
        domain_labels_val_train = domain_labels_val

    val_loader = TorchDataLoader(TensorDataset(x_val_tensor, y_val_tensor), batch_size=NEURAL_DEFAULTS['training']['batch_size'], shuffle=False)
    train_loader = TorchDataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=NEURAL_DEFAULTS['training']['batch_size'], shuffle=True)

    # STEP 2: Get loss function
    from loss_functions import get_loss_function
    
    # Determine if we need domain-aware loss
    loss_name = args.loss if hasattr(args, 'loss') else 'mse'
    needs_domains = loss_name in ['domain_weighted', 'domain_balanced', 'het_per_domain']
    
    # Check if domain clustering is actually enabled
    if needs_domains and (domain_labels_train is None or args.k_domains <= 1):
        print(f"Warning: {loss_name} requires domain clustering but k_domains={args.k_domains}. Falling back to MSE.")
        loss_name = 'mse'
        needs_domains = False
    
    # Parse loss parameters if provided
    loss_kwargs = {}
    if hasattr(args, 'loss_params') and args.loss_params:
        import json
        loss_kwargs = json.loads(args.loss_params)
    
    # Add num_domains for domain-aware losses
    if needs_domains:
        loss_kwargs['num_domains'] = args.k_domains
    
    # STEP 3: Create model with appropriate output size
    if loss_name == 'heteroscedastic':
        # Heteroscedastic needs 2 outputs: mean and log_variance
        model = DNNRegressionModel(
            input_size=x_train.shape[1], 
            hidden_size1=params['hidden_size1'], 
            hidden_size2=params['hidden_size2']
        )
        # Modify final layer to output 2 values
        model.fc3 = nn.Linear(params['hidden_size2'], 2)
        model.activation = activation
        model.to(device)
        criterion = get_loss_function(loss_name)
    else:
        # Standard model with 1 output
        model = DNNRegressionModel(
            input_size=x_train.shape[1], 
            hidden_size1=params['hidden_size1'], 
            hidden_size2=params['hidden_size2']
        )
        model.activation = activation
        model.to(device)
        criterion = get_loss_function(loss_name, **loss_kwargs)

    # STEP 4: Apply Bayesian transformation if requested
    model_name = "dnn"
    if args.bayesian_transformation == "full":
        model = apply_bayesian_transformation(model)
        model_name = "bnn_full"
        criterion = bnn_elbo_criterion(criterion, model, len(x_train))
    elif args.bayesian_transformation == "last_layer":
        model = apply_bayesian_transformation_last_layer(model)
        model_name = "bnn_last"
        criterion = bnn_elbo_criterion(criterion, model, len(x_train))
    elif args.bayesian_transformation == "variational":
        model = apply_bayesian_transformation_last_layer_variational(model)
        model_name = "bnn_variational"
        # Use ELBO loss (MSE + KL divergence) for VBLL
        criterion = VBLLLoss(model, n_data=len(x_train))
    elif args.bayesian_transformation == "full_variational":
        model = apply_bayesian_transformation_full_variational(model)
        model_name = "bnn_full_variational"
        criterion = VBLLLoss(model, n_data=len(x_train))

    optimizer = torch.optim.Adam(model.parameters(), lr=NEURAL_DEFAULTS['training']['lr'])

    # STEP 5: Train with appropriate domain labels
    train_nn(model, train_loader, val_loader, criterion, optimizer, device, args, s, iteration, file_no, model_name, rep,
             domain_labels_train=domain_labels_train if needs_domains else None,
             domain_labels_val=domain_labels_val_train if needs_domains else None)
    
    model.eval()

    if is_bayesian and args.uncertainty:
        torch.manual_seed(iteration_seed)
        np.random.seed(iteration_seed)
        
        num_samples = NEURAL_DEFAULTS['training']['mc_passes']

        # Get calibration predictions
        x_val_cal_tensor = torch.tensor(x_val_cal, dtype=torch.float32).to(device)
        preds_cal = []
        with torch.no_grad():
            for _ in range(num_samples):
                output = model(x_val_cal_tensor).cpu().numpy()
                output, _ = split_predictive_head(output, loss_name)
                preds_cal.append(output)

        preds_cal = np.stack(preds_cal, axis=0)
        y_cal_pred_mean = preds_cal.mean(axis=0).flatten()
        y_cal_pred_std = preds_cal.std(axis=0).flatten()

        # Find optimal temperature
        temperature = calibrate_uncertainty_simple(y_cal_pred_mean, y_cal_pred_std, y_val_cal)

        # Get test predictions
        preds = []
        # The per-molecule variance the heteroscedastic and evidential heads
        # predict, kept alongside the prediction instead of sliced off. This
        # whole block runs only under -u/--uncertainty, so a run that is not
        # asked for uncertainty never holds it.
        head_vars = []
        with torch.no_grad():
            for _ in range(num_samples):
                output = model(x_test_tensor).cpu().numpy()
                output, variance = split_predictive_head(output, loss_name)
                preds.append(output)
                if variance is not None:
                    head_vars.append(variance)

        preds = np.stack(preds, axis=0)  # Shape: (num_samples, num_datapoints, 1)
        y_pred_mean = preds.mean(axis=0).flatten()

        # Decompose uncertainty — use VBLL decomposition if model has a VBLLLayer
        # Use the LAST VBLLLayer (output layer) for observation noise
        vbll_layers = [m for m in model.modules() if isinstance(m, VBLLLayer)]
        vbll_layer = vbll_layers[-1] if vbll_layers else None

        if vbll_layer is not None:
            learned_noise_var = vbll_layer.noise_var.item()
            epistemic, aleatoric, total = decompose_uncertainty_vbll(preds.squeeze(), learned_noise_var)
        elif head_vars:
            # The head predicted a variance per molecule, so the observation term
            # is that variance rather than absent. Without this the two losses
            # reported the spread over the stochastic passes and nothing else.
            epistemic, aleatoric, total = decompose_uncertainty_sampling_heteroscedastic(
                preds.squeeze(), np.stack(head_vars, axis=0).squeeze())
        else:
            epistemic, aleatoric, total = decompose_uncertainty_sampling(preds.squeeze(), num_samples)

        # The TOTAL predictive spread, not the spread of the MC passes.
        #
        # `preds.std(axis=0)` is the epistemic term alone. For a VBLL the layer
        # also carries a learned observation noise, and `total` -- computed just
        # above -- is sqrt(epistemic^2 + aleatoric^2). The reported column used to
        # be the MC spread, so the learned noise was computed and thrown away and
        # the VBLL models' coverage was a statement about the latent function
        # rather than about an observation (RERUN_PLAN.md 2.13). A plain BNN has
        # no learned noise, so total equals the MC spread there and nothing moves.
        y_pred_std_uncalibrated = (np.asarray(total).flatten()
                                   if total is not None
                                   else preds.std(axis=0).flatten())
        y_pred_std_calibrated = y_pred_std_uncalibrated * temperature

        # Apply calibration to epistemic (aleatoric stays None for standard BNN)
        if epistemic is not None:
            epistemic = epistemic * temperature

        y_pred = y_pred_mean

    else:
        # Non-Bayesian prediction
        with torch.no_grad():
            y_pred_tensor = model(x_test_tensor).cpu().numpy()
            # Evidential was missing from this branch, so its four outputs were
            # flattened into four times as many predictions as there are
            # molecules, and the metrics were computed against whatever that
            # lined up with. There is no uncertainty to keep here -- this branch
            # is the one that was not asked for any.
            y_pred_tensor, _ = split_predictive_head(y_pred_tensor, loss_name)
            y_pred = y_pred_tensor.flatten()
        
        y_pred_std_uncalibrated = None
        y_pred_std_calibrated = None
        temperature = None
        epistemic = None
        aleatoric = None

    # Calculate metrics normally
    metrics = calculate_regression_metrics(y_test, y_pred, logging=True)
    
    # Create full model name with loss function
    full_model_name = f"{model_name}_{loss_name}" if loss_name != 'mse' else model_name
    
    save_results(args.filepath, s, iteration, full_model_name, rep, args.sample_size, metrics, params_source, loss_name)

    def _build_dnn_for_fold(n_features, n_fit):
        """The same network the main path builds, for one inner fold.

        Mirrors STEP 3 and STEP 4 above. It is a separate function rather than a
        refactor of the main path because the main path's construction order also
        fixes the random stream the reported result was produced with, and moving
        it is a change to that result.
        """
        from loss_functions import get_loss_function
        if loss_name == 'heteroscedastic':
            m = DNNRegressionModel(input_size=n_features,
                                   hidden_size1=params['hidden_size1'],
                                   hidden_size2=params['hidden_size2'])
            m.fc3 = nn.Linear(params['hidden_size2'], 2)
            m.activation = activation
            m.to(device)
            crit = get_loss_function(loss_name)
        else:
            m = DNNRegressionModel(input_size=n_features,
                                   hidden_size1=params['hidden_size1'],
                                   hidden_size2=params['hidden_size2'])
            m.activation = activation
            m.to(device)
            crit = get_loss_function(loss_name, **loss_kwargs)

        if args.bayesian_transformation == "full":
            m = apply_bayesian_transformation(m)
            crit = bnn_elbo_criterion(crit, m, n_fit)
        elif args.bayesian_transformation == "last_layer":
            m = apply_bayesian_transformation_last_layer(m)
            crit = bnn_elbo_criterion(crit, m, n_fit)
        elif args.bayesian_transformation == "variational":
            m = apply_bayesian_transformation_last_layer_variational(m)
            crit = VBLLLoss(m, n_data=n_fit)
        elif args.bayesian_transformation == "full_variational":
            m = apply_bayesian_transformation_full_variational(m)
            crit = VBLLLoss(m, n_data=n_fit)
        m.to(device)
        return m, crit

    def _train_dnn_for_fold(built, x_fit, y_fit, x_es, y_es):
        m, crit = built
        xt = torch.tensor(np.asarray(x_fit), dtype=torch.float32).to(device)
        yt = torch.tensor(np.asarray(y_fit), dtype=torch.float32).view(-1, 1).to(device)
        xe = torch.tensor(np.asarray(x_es), dtype=torch.float32).to(device)
        ye = torch.tensor(np.asarray(y_es), dtype=torch.float32).view(-1, 1).to(device)
        loader = TorchDataLoader(TensorDataset(xt, yt), batch_size=NEURAL_DEFAULTS['training']['batch_size'], shuffle=True)
        es_loader = TorchDataLoader(TensorDataset(xe, ye), batch_size=NEURAL_DEFAULTS['training']['batch_size'], shuffle=False)
        opt = torch.optim.Adam(m.parameters(), lr=NEURAL_DEFAULTS['training']['lr'])
        train_nn(m, loader, es_loader, crit, opt, device, args, s, iteration,
                 file_no, 'oof_inner', rep)
        return m

    # *** UPDATED: Save uncertainty with decomposition ***
    if args.uncertainty and is_bayesian:
        save_uncertainty_values(
            y_pred_mean=y_pred,
            y_pred_std=y_pred_std_uncalibrated,
            y_true_original=y_test_original,
            y_true_noisy=y_test,
            filepath=args.filepath,
            model_name=full_model_name,
            rep=rep,
            sigma_noise=s,
            iteration=iteration,
            file_no=file_no,
            y_pred_std_calibrated=y_pred_std_calibrated,
            temperature=temperature,
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            split='test',
            **_held_out_noise_columns(train_noise, len(y_test), y_pred),
        )

        # The TRAINING molecules. This family fits on the training split alone --
        # validation is the early-stopping set, never a fit row -- so only the
        # training provenance rows are asked for.
        def _fp(x_fit, y_fit, x_score):
            # The early-stopping split is carved from THIS fold's own fit rows.
            # Reusing the outer validation set would early-stop against molecules
            # sharing scaffolds with the rows being scored, so the out-of-fold
            # uncertainty would not be on the same extrapolation footing as the
            # test set.
            nv = max(1, len(y_fit) // 5)
            inner_model = _train_dnn_for_fold(
                _build_dnn_for_fold(x_fit.shape[1], len(y_fit) - nv),
                x_fit[nv:], y_fit[nv:], x_fit[:nv], y_fit[:nv])
            inner_model.eval()
            xs = torch.tensor(np.asarray(x_score), dtype=torch.float32).to(device)
            draws = []
            with torch.no_grad():
                for _ in range(num_samples):
                    out = inner_model(xs).cpu().numpy()
                    if loss_name == 'heteroscedastic':
                        out = out[:, 0:1]
                    draws.append(out)
            draws = np.stack(draws, axis=0)
            return draws.mean(axis=0).flatten(), draws.std(axis=0).flatten()

        score_training_molecules_out_of_fold(
            _fp, x_train, y_train, train_noise, args, s, rep, iteration,
            iteration_seed, file_no, full_model_name,
            val_slice=None, restore_torch_rng=True)

    return metrics[3]

def train_bnn_last_standalone(x_train, y_train, x_test, y_test, x_val, y_val, 
                               args, s, rep, iteration, iteration_seed, file_no, 
                               y_test_original, train_noise=None):
    """
    Standalone BNN-Last training that matches phase2_train.py exactly.

    Four things were wrong here and all four are fixed (RERUN_PLAN.md 2.13).
    It built 128 then 64, which is NEURAL_DEFAULTS['dnn']['hidden_sizes'], and
    saved the row as 'mlp_bnn_last' -- NEURAL_DEFAULTS['mlp'] is 128 then 128.
    The row is named for the architecture it builds now. Its early-stopping loop
    counted patience but never snapshotted or restored weights, so it returned
    whatever epoch the counter stopped on, up to ten epochs past the validation
    optimum spent memorising corrupted labels -- the defect train_nn was
    rewritten to fix, and this was the last function still carrying it. Every
    constant was a literal; they come from the spec now. And it was fitted with
    plain MSE, like the other torchbnn models were until 2026-08-27.

    Reachable only as model_type 'mlp_bnn_last_standalone'
    (process_and_train.py), which no job script requests.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    import torchbnn as bnn
    import numpy as np

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(iteration_seed)

    _t = NEURAL_DEFAULTS['training']
    h1, h2 = NEURAL_DEFAULTS['dnn']['hidden_sizes']
    _drop = NEURAL_DEFAULTS['dnn']['dropout_rate']

    # Create model directly (not transform)
    model = nn.Sequential(
        nn.Linear(x_train.shape[1], h1),
        nn.ReLU(),
        nn.Dropout(_drop),
        nn.Linear(h1, h2),
        nn.ReLU(),
        nn.Dropout(_drop),
        bnn.BayesLinear(in_features=h2, out_features=1,
                        prior_mu=BAYESIAN_DEFAULTS['bnn_prior_mu'],
                        prior_sigma=BAYESIAN_DEFAULTS['bnn_prior_sigma'])
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=_t['lr'])
    criterion = bnn_elbo_criterion(nn.MSELoss(), model, len(x_train))
    
    # Tensors - Y is shape (N,) not (N, 1)
    X_tr_t = torch.FloatTensor(x_train).to(device)
    y_tr_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(x_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)
    X_test_t = torch.FloatTensor(x_test).to(device)
    
    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                              batch_size=_t['batch_size'], shuffle=True)

    # Training with early stopping
    best_val_loss = float('inf')
    patience = _t['patience']
    patience_counter = 0
    best_state = None
    best_epoch = -1

    for epoch in range(int(getattr(args, 'epochs', _t['epochs']) or _t['epochs'])):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            if pred.dim() > 1:
                pred = pred.squeeze()
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
        
        # Single-pass validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            if val_pred.dim() > 1:
                val_pred = val_pred.squeeze()
            val_loss = criterion(val_pred, y_val_t).item()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss - _t['improvement_tolerance']:
            best_val_loss = val_loss
            patience_counter = 0
            best_epoch = epoch
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if _t['restore_best_weights'] and best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored epoch {best_epoch} (val loss {best_val_loss:.4f})")

    # BNN prediction with MC sampling
    model.eval()
    n_samples = _t['mc_passes']
    predictions = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(X_test_t)
            if pred.dim() > 1:
                pred = pred.squeeze()
            predictions.append(pred.cpu().numpy())
    
    predictions = np.array(predictions)
    y_pred = predictions.mean(axis=0)
    y_pred_std = predictions.std(axis=0)
    
    # Use your existing metrics function
    metrics = calculate_regression_metrics(y_test, y_pred, logging=True)
    
    # The row is named for the network this function BUILDS: 128 then 64 is the
    # DNN topology, not the MLP's 128 then 128.
    save_results(args.filepath, s, iteration, 'dnn_bnn_last', rep, args.sample_size,
                 metrics, 'default', 'mse')
    
    # Always Bayesian (dedicated BNN Last function), consistent with train_dnn_model/train_mlp_model
    is_bayesian = True
    if args.uncertainty and is_bayesian:
        save_uncertainty_values(
            y_pred_mean=y_pred,
            y_pred_std=y_pred_std,
            y_true_original=y_test_original,
            y_true_noisy=y_test,
            filepath=args.filepath,
            model_name='dnn_bnn_last',
            rep=rep,
            sigma_noise=s,
            iteration=iteration,
            file_no=file_no,
            y_pred_std_calibrated=y_pred_std,
            temperature=1.0
        )

    return metrics[3]

def train_flexible_dnn_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, file_no, y_test_original, trial=None,
                            domain_labels_train=None, domain_labels_val=None, domain_labels_test=None, train_noise=None):
    params = {}
    params_source = 'default'
    if hasattr(args, 'use_best_params') and args.use_best_params and not args.tuning:
        best_params = load_best_hyperparameters('flexible_dnn', rep)
        if best_params is not None:
            params = best_params
            params_source = 'tuned'
            print(f"Using tuned hyperparameters for flexible_dnn-{rep}")
    if not params:
        if args.tuning:
            num_layers = trial.suggest_int("num_layers", 1, 4)
            hidden_sizes = []
            for i in range(num_layers):
                hidden_size = trial.suggest_categorical(f"hidden_size_{i}", [32, 64, 128, 256, 512, 1024])
                hidden_sizes.append(hidden_size)
            params['hidden_sizes'] = hidden_sizes
            params['activation'] = trial.suggest_categorical('activation', ['relu', 'tanh'])
            params_source = 'tuning_trial'
        else:
            params['hidden_sizes'] = getattr(args, 'hidden_sizes', None) or [128, 64]
            params['activation'] = 'relu'
            params_source = 'default' if not getattr(args, 'hidden_sizes', None) else 'cli'
    
    activation_map = {'relu': nn.ReLU(), 'tanh': nn.Tanh()}
    activation = activation_map[params['activation']]
    
    # Loss function setup
    loss_name = args.loss if hasattr(args, 'loss') else 'mse'
    loss_kwargs = {}
    
    if args.tuning and trial is not None:
        if loss_name == 'huber':
            loss_kwargs['delta'] = trial.suggest_float('loss_delta', 0.1, 5.0)
        elif loss_name == 'cauchy':
            loss_kwargs['c'] = trial.suggest_float('loss_c', 0.1, 5.0)
        elif loss_name == 'focal':
            loss_kwargs['gamma'] = trial.suggest_float('loss_gamma', 0.5, 5.0)
            loss_kwargs['alpha'] = trial.suggest_float('loss_alpha', 0.1, 0.9)
        elif loss_name == 'truncated':
            loss_kwargs['quantile'] = trial.suggest_float('loss_quantile', 0.7, 0.99)
            loss_kwargs['base_loss'] = trial.suggest_categorical('loss_base', ['mse', 'mae'])
        elif loss_name == 'barron':
            loss_kwargs['alpha_init'] = trial.suggest_float('loss_alpha_init', 0.5, 4.0)
            loss_kwargs['scale_init'] = trial.suggest_float('loss_scale_init', 0.1, 5.0)
        elif loss_name == 'evidential':
            loss_kwargs['coeff'] = trial.suggest_float('loss_coeff', 0.001, 1.0, log=True)
        elif loss_name in ['domain_weighted', 'domain_balanced', 'adaptive_domain']:
            loss_kwargs['base_loss'] = trial.suggest_categorical('loss_base', ['mse', 'mae', 'huber'])
            if loss_name == 'adaptive_domain':
                loss_kwargs['adaptation_rate'] = trial.suggest_float('loss_adapt_rate', 0.001, 0.1, log=True)
    
    if hasattr(args, 'loss_params') and args.loss_params:
        import json
        cli_loss_kwargs = json.loads(args.loss_params)
        loss_kwargs.update(cli_loss_kwargs)
    
    needs_domains = loss_name in ['domain_weighted', 'domain_balanced', 'het_per_domain', 'adaptive_domain', 'mixture_domain']
    
    if needs_domains and (domain_labels_train is None or args.k_domains <= 1):
        print(f"Warning: {loss_name} requires domain clustering but k_domains={args.k_domains}. Falling back to MSE.")
        loss_name = 'mse'
        needs_domains = False
    
    if needs_domains:
        loss_kwargs['num_domains'] = args.k_domains
    
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)
    
    is_bayesian = args.bayesian_transformation is not None
    
    if is_bayesian and args.uncertainty:
        split_idx = len(x_val) // 2
        x_val_train = x_val[:split_idx]
        y_val_train = y_val[:split_idx]
        x_val_cal = x_val[split_idx:]
        y_val_cal = y_val[split_idx:]
        
        x_val_tensor = torch.tensor(x_val_train, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val_train, dtype=torch.float32).view(-1, 1).to(device)
        
        if domain_labels_val is not None:
            domain_labels_val_train = domain_labels_val[:split_idx]
            domain_labels_val_cal = domain_labels_val[split_idx:]
        else:
            domain_labels_val_train = None
            domain_labels_val_cal = None
    else:
        x_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
        domain_labels_val_train = domain_labels_val
    
    train_loader = TorchDataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=NEURAL_DEFAULTS['training']['batch_size'], shuffle=True)
    val_loader = TorchDataLoader(TensorDataset(x_val_tensor, y_val_tensor), batch_size=NEURAL_DEFAULTS['training']['batch_size'], shuffle=False)
    
    if loss_name == 'heteroscedastic':
        model = FlexibleDNNRegressionModel(input_size=x_train.shape[1], hidden_sizes=params['hidden_sizes'], activation_fn=activation).to(device)
        model.network[-1] = nn.Linear(params['hidden_sizes'][-1], 2)
    elif loss_name == 'evidential':
        model = FlexibleDNNRegressionModel(input_size=x_train.shape[1], hidden_sizes=params['hidden_sizes'], activation_fn=activation).to(device)
        model.network[-1] = nn.Linear(params['hidden_sizes'][-1], 4)
    else:
        model = FlexibleDNNRegressionModel(input_size=x_train.shape[1], hidden_sizes=params['hidden_sizes'], activation_fn=activation).to(device)
    
    # Encode architecture in model name when custom hidden sizes provided
    arch_str = "_".join(str(h) for h in params['hidden_sizes'])
    if params['hidden_sizes'] == [128, 64]:
        model_name = "flexible_dnn"
    else:
        model_name = f"flexible_dnn_{arch_str}"

    if args.bayesian_transformation == "full":
        model = apply_bayesian_transformation(model)
        model_name = model_name.replace("flexible_dnn", "flexible_bnn_full")
        criterion = bnn_elbo_criterion(criterion, model, len(x_train))
    elif args.bayesian_transformation == "last_layer":
        model = apply_bayesian_transformation_last_layer(model)
        model_name = model_name.replace("flexible_dnn", "flexible_bnn_last")
        criterion = bnn_elbo_criterion(criterion, model, len(x_train))
    elif args.bayesian_transformation == "variational":
        model = apply_bayesian_transformation_last_layer_variational(model)
        model_name = model_name.replace("flexible_dnn", "flexible_bnn_variational")
    elif args.bayesian_transformation == "full_variational":
        model = apply_bayesian_transformation_full_variational(model)
        model_name = model_name.replace("flexible_dnn", "flexible_bnn_full_variational")

    from loss_functions import get_loss_function
    criterion = get_loss_function(loss_name, **loss_kwargs)
    if args.bayesian_transformation in ("variational", "full_variational"):
        criterion = VBLLLoss(model, n_data=len(x_train))
    optimizer = torch.optim.Adam(model.parameters(), lr=NEURAL_DEFAULTS['training']['lr'])
    
    train_nn(model, train_loader, val_loader, criterion, optimizer, device, args, s, iteration, file_no, model_name, rep,
             domain_labels_train=domain_labels_train if needs_domains else None,
             domain_labels_val=domain_labels_val_train if needs_domains else None)
    
    model.eval()
    
    if is_bayesian:
        torch.manual_seed(iteration_seed)
        np.random.seed(iteration_seed)
        
        num_samples = NEURAL_DEFAULTS['training']['mc_passes']
        
        if args.uncertainty:
            x_val_cal_tensor = torch.tensor(x_val_cal, dtype=torch.float32).to(device)
            preds_cal = []
            with torch.no_grad():
                for _ in range(num_samples):
                    output = model(x_val_cal_tensor).cpu().numpy()
                    output, _ = split_predictive_head(output, loss_name)
                    preds_cal.append(output)

            preds_cal = np.stack(preds_cal, axis=0)
            y_cal_pred_mean = preds_cal.mean(axis=0).flatten()
            y_cal_pred_std = preds_cal.std(axis=0).flatten()

            temperature = calibrate_uncertainty_simple(y_cal_pred_mean, y_cal_pred_std, y_val_cal)

        preds = []
        with torch.no_grad():
            for _ in range(num_samples):
                output = model(x_test_tensor).cpu().numpy()
                # The predicted variance is dropped here and nowhere else, because
                # this model writes no decomposition at all. `flexible_dnn` is in
                # EXCLUDED_MODELS and no figure reads it; wiring the variance
                # through would have nothing to write it to.
                output, _ = split_predictive_head(output, loss_name)
                preds.append(output)
        
        preds = np.stack(preds, axis=0)
        y_pred_mean = preds.mean(axis=0).flatten()
        y_pred_std_uncalibrated = preds.std(axis=0).flatten()
        
        if args.uncertainty:
            y_pred_std_calibrated = y_pred_std_uncalibrated * temperature
        else:
            y_pred_std_calibrated = None
            temperature = None
        
        y_pred = y_pred_mean
    else:
        with torch.no_grad():
            y_pred_tensor = model(x_test_tensor).cpu().numpy()
            if loss_name == 'heteroscedastic':
                y_pred = y_pred_tensor[:, 0].flatten()
            elif loss_name == 'evidential':
                y_pred = y_pred_tensor[:, 0].flatten()
            else:
                y_pred = y_pred_tensor.flatten()
        y_pred_std_uncalibrated = None
        y_pred_std_calibrated = None
        temperature = None
    
    metrics = calculate_regression_metrics(y_test, y_pred, logging=True)
    
    full_model_name = f"{model_name}_{loss_name}" if loss_name != 'mse' else model_name
    
    save_results(args.filepath, s, iteration, full_model_name, rep, args.sample_size, metrics, params_source, loss_name)
    
    if args.uncertainty and is_bayesian:
        save_uncertainty_values(
            y_pred_mean=y_pred,
            y_pred_std=y_pred_std_uncalibrated,
            y_true_original=y_test_original,
            y_true_noisy=y_test,
            filepath=args.filepath,
            model_name=full_model_name,
            rep=rep,
            sigma_noise=s,
            iteration=iteration,
            file_no=file_no,
            y_pred_std_calibrated=y_pred_std_calibrated,
            temperature=temperature
        )
    
    return metrics[3]

def train_mlp_variant_model(x_train, y_train, x_test, y_test, x_val, y_val, model_type, args, s, rep, iteration, iteration_seed, file_no, y_test_original, trial=None,
                           domain_labels_train=None, domain_labels_val=None, domain_labels_test=None, train_noise=None):
    params = {}
    params_source = 'default'

    if hasattr(args, 'use_best_params') and args.use_best_params and not args.tuning:
        best_params = load_best_hyperparameters(model_type, rep)
        if best_params is not None:
            params = best_params
            params_source = 'tuned'
            print(f"Using tuned hyperparameters for {model_type}-{rep}")

    if not params:
        if args.tuning:
            params['hidden_size'] = trial.suggest_categorical('hidden_size', [32, 64, 128, 256, 512, 1024])
            params['num_hidden_layers'] = trial.suggest_int('num_hidden_layers', 1, 5)
            params['dropout_rate'] = trial.suggest_float('dropout_rate', 0.1, 0.5)
            params['lr'] = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
            params_source = 'tuning_trial'
        else:
            params['hidden_size'] = NEURAL_DEFAULTS['mlp']['hidden_size']
            params['num_hidden_layers'] = NEURAL_DEFAULTS['mlp']['num_hidden_layers']
            params['dropout_rate'] = NEURAL_DEFAULTS['mlp']['dropout_rate']
            params['lr'] = NEURAL_DEFAULTS['training']['lr']
            params_source = 'default'

    # Loss function setup
    loss_name = args.loss if hasattr(args, 'loss') else 'mse'
    loss_kwargs = {}
    
    if args.tuning and trial is not None:
        if loss_name == 'huber':
            loss_kwargs['delta'] = trial.suggest_float('loss_delta', 0.1, 5.0)
        elif loss_name == 'cauchy':
            loss_kwargs['c'] = trial.suggest_float('loss_c', 0.1, 5.0)
        elif loss_name == 'focal':
            loss_kwargs['gamma'] = trial.suggest_float('loss_gamma', 0.5, 5.0)
            loss_kwargs['alpha'] = trial.suggest_float('loss_alpha', 0.1, 0.9)
        elif loss_name == 'truncated':
            loss_kwargs['quantile'] = trial.suggest_float('loss_quantile', 0.7, 0.99)
            loss_kwargs['base_loss'] = trial.suggest_categorical('loss_base', ['mse', 'mae'])
        elif loss_name == 'barron':
            loss_kwargs['alpha_init'] = trial.suggest_float('loss_alpha_init', 0.5, 4.0)
            loss_kwargs['scale_init'] = trial.suggest_float('loss_scale_init', 0.1, 5.0)
        elif loss_name == 'evidential':
            loss_kwargs['coeff'] = trial.suggest_float('loss_coeff', 0.001, 1.0, log=True)
        elif loss_name in ['domain_weighted', 'domain_balanced', 'adaptive_domain']:
            loss_kwargs['base_loss'] = trial.suggest_categorical('loss_base', ['mse', 'mae', 'huber'])
            if loss_name == 'adaptive_domain':
                loss_kwargs['adaptation_rate'] = trial.suggest_float('loss_adapt_rate', 0.001, 0.1, log=True)
    
    if hasattr(args, 'loss_params') and args.loss_params:
        import json
        cli_loss_kwargs = json.loads(args.loss_params)
        loss_kwargs.update(cli_loss_kwargs)
    
    needs_domains = loss_name in ['domain_weighted', 'domain_balanced', 'het_per_domain', 'adaptive_domain', 'mixture_domain']
    
    if needs_domains and (domain_labels_train is None or args.k_domains <= 1):
        print(f"Warning: {loss_name} requires domain clustering but k_domains={args.k_domains}. Falling back to MSE.")
        loss_name = 'mse'
        needs_domains = False
    
    if needs_domains:
        loss_kwargs['num_domains'] = args.k_domains

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

    # STEP 1: Split validation data if Bayesian
    is_bayesian = args.bayesian_transformation is not None
    
    if is_bayesian and args.uncertainty and x_val is not None and y_val is not None:
        split_idx = len(x_val) // 2
        x_val_train = x_val[:split_idx]
        y_val_train = y_val[:split_idx]
        x_val_cal = x_val[split_idx:]
        y_val_cal = y_val[split_idx:]
        
        x_val_tensor = torch.tensor(x_val_train, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val_train, dtype=torch.float32).view(-1, 1).to(device)
        
        if domain_labels_val is not None:
            domain_labels_val_train = domain_labels_val[:split_idx]
            domain_labels_val_cal = domain_labels_val[split_idx:]
        else:
            domain_labels_val_train = None
            domain_labels_val_cal = None
            
        val_loader = TorchDataLoader(TensorDataset(x_val_tensor, y_val_tensor), batch_size=NEURAL_DEFAULTS['training']['batch_size'], shuffle=False)
    elif x_val is not None and y_val is not None:
        x_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
        domain_labels_val_train = domain_labels_val
        val_loader = TorchDataLoader(TensorDataset(x_val_tensor, y_val_tensor), batch_size=NEURAL_DEFAULTS['training']['batch_size'], shuffle=False)
    else:
        domain_labels_val_train = None
        val_loader = None

    train_loader = TorchDataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=NEURAL_DEFAULTS['training']['batch_size'], shuffle=True)

    # Model setup with special output layers for certain losses
    if model_type == "mlp":
        model = MLPRegressor(input_size=x_train.shape[1], hidden_size=params['hidden_size'],
                             num_hidden_layers=params['num_hidden_layers'], dropout_rate=params['dropout_rate'])
    elif model_type == "residual_mlp":
        model = ResidualMLP(input_size=x_train.shape[1], hidden_size=128, num_layers=3)
    elif model_type == "factorization_mlp":
        model = FactorizationMLP(input_size=x_train.shape[1], hidden_size=128, factor_size=16)
    elif model_type == "mtl":
        model = MTLRegressionModel(input_size=x_train.shape[1], hidden_size=128, num_tasks=1)
    
    # Modify output layer for heteroscedastic or evidential losses if needed
    if loss_name == 'heteroscedastic':
        # Need to modify the last layer to output 2 values (mean and variance)
        if hasattr(model, 'fc_out'):
            model.fc_out = nn.Linear(model.fc_out.in_features, 2)
        elif hasattr(model, 'output_layer'):
            model.output_layer = nn.Linear(model.output_layer.in_features, 2)
    elif loss_name == 'evidential':
        # Need to modify the last layer to output 4 values
        if hasattr(model, 'fc_out'):
            model.fc_out = nn.Linear(model.fc_out.in_features, 4)
        elif hasattr(model, 'output_layer'):
            model.output_layer = nn.Linear(model.output_layer.in_features, 4)

    # The base loss is built HERE, before the Bayesian transformation, because
    # the transformation wraps it. It used to be built afterwards, which broke
    # NN-beta's Bayesian variants two ways at once (found 2026-08-27):
    #
    #   * `criterion = bnn_elbo_criterion(criterion, ...)` below read `criterion`
    #     before it existed, so MLP-BNN-Full and the last-layer variant raised
    #     UnboundLocalError and produced no rows at all.
    #   * had they got past that, the unconditional `criterion = get_loss_
    #     function(...)` that followed would have thrown the ELBO wrapper away
    #     and trained them on plain MSE with no KL term -- exactly the defect the
    #     KL term was added on 2026-08-27 to fix, reintroduced by ordering.
    #
    # train_dnn_model has always built it first; this is the same order.
    from loss_functions import get_loss_function
    criterion = get_loss_function(loss_name, **loss_kwargs)

    # Apply Bayesian transformation before moving to device
    if args.bayesian_transformation == "full":
        model = apply_bayesian_transformation(model)
        model_name = f"{model_type}_bnn_full"
        criterion = bnn_elbo_criterion(criterion, model, len(x_train))
    elif args.bayesian_transformation == "last_layer":
        model = apply_bayesian_transformation_last_layer(model)
        model_name = f"{model_type}_bnn_last"
        criterion = bnn_elbo_criterion(criterion, model, len(x_train))
    elif args.bayesian_transformation == "variational":
        model = apply_bayesian_transformation_last_layer_variational(model)
        model_name = f"{model_type}_bnn_variational"
        criterion = VBLLLoss(model, n_data=len(x_train))
    elif args.bayesian_transformation == "full_variational":
        model = apply_bayesian_transformation_full_variational(model)
        model_name = f"{model_type}_bnn_full_variational"
        criterion = VBLLLoss(model, n_data=len(x_train))
    else:
        model_name = model_type

    model.to(device)

    criterion = get_loss_function(loss_name, **loss_kwargs)

    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    # Train with domain labels if needed
    train_nn(model, train_loader, val_loader, criterion, optimizer, device, args, s, iteration, file_no, model_name, rep,
             domain_labels_train=domain_labels_train if needs_domains else None,
             domain_labels_val=domain_labels_val_train if needs_domains else None)
    
    model.eval()

    # STEP 2: Get predictions and calibrate if Bayesian
    if is_bayesian:
        torch.manual_seed(iteration_seed)
        np.random.seed(iteration_seed)
        
        num_samples = NEURAL_DEFAULTS['training']['mc_passes']
        
        if args.uncertainty:
            # Get calibration predictions
            x_val_cal_tensor = torch.tensor(x_val_cal, dtype=torch.float32).to(device)
            preds_cal = []
            with torch.no_grad():
                for _ in range(num_samples):
                    output = model(x_val_cal_tensor).cpu().numpy()
                    output, _ = split_predictive_head(output, loss_name)
                    preds_cal.append(output)

            preds_cal = np.stack(preds_cal, axis=0)
            y_cal_pred_mean = preds_cal.mean(axis=0).flatten()
            y_cal_pred_std = preds_cal.std(axis=0).flatten()

            temperature = calibrate_uncertainty_simple(y_cal_pred_mean, y_cal_pred_std, y_val_cal)

        # Get test predictions
        preds = []
        # The per-molecule variance the heteroscedastic and evidential heads
        # predict. This loop runs whether or not uncertainty was asked for, since
        # the prediction itself comes out of it, so the variance is kept ONLY
        # under -u/--uncertainty -- nothing else would read it.
        head_vars = []
        with torch.no_grad():
            for _ in range(num_samples):
                output = model(x_test_tensor).cpu().numpy()
                output, variance = split_predictive_head(output, loss_name)
                preds.append(output)
                if args.uncertainty and variance is not None:
                    head_vars.append(variance)

        preds = np.stack(preds, axis=0)
        y_pred_mean = preds.mean(axis=0).flatten()
        y_pred_std_uncalibrated = preds.std(axis=0).flatten()

        if args.uncertainty:
            y_pred_std_calibrated = y_pred_std_uncalibrated * temperature

            # Decompose uncertainty — use VBLL decomposition if model has a VBLLLayer
            # Use the LAST VBLLLayer (output layer) for observation noise
            vbll_layers = [m for m in model.modules() if isinstance(m, VBLLLayer)]
            vbll_layer = vbll_layers[-1] if vbll_layers else None

            if vbll_layer is not None:
                learned_noise_var = vbll_layer.noise_var.item()
                epistemic, aleatoric, total = decompose_uncertainty_vbll(preds.squeeze(), learned_noise_var)
            elif head_vars:
                # The head predicted a variance per molecule, so the observation
                # term is that variance rather than absent.
                epistemic, aleatoric, total = decompose_uncertainty_sampling_heteroscedastic(
                    preds.squeeze(), np.stack(head_vars, axis=0).squeeze())
            else:
                epistemic, aleatoric, total = decompose_uncertainty_sampling(preds.squeeze(), num_samples)

            # Apply calibration to epistemic (aleatoric stays None for standard BNN)
            if epistemic is not None:
                epistemic = epistemic * temperature
        else:
            y_pred_std_calibrated = None
            temperature = None
            epistemic = None
            aleatoric = None

        y_pred = y_pred_mean

    else:
        with torch.no_grad():
            y_pred_tensor = model(x_test_tensor).cpu().numpy()
            if loss_name == 'heteroscedastic':
                y_pred = y_pred_tensor[:, 0].flatten()
            elif loss_name == 'evidential':
                y_pred = y_pred_tensor[:, 0].flatten()
            else:
                y_pred = y_pred_tensor.flatten()
        y_pred_std_uncalibrated = None
        y_pred_std_calibrated = None
        temperature = None
        epistemic = None
        aleatoric = None

    metrics = calculate_regression_metrics(y_test, y_pred, logging=True)

    # Update model name with loss function if not MSE
    full_model_name = f"{model_name}_{loss_name}" if loss_name != 'mse' else model_name

    save_results(args.filepath, s, iteration, full_model_name, rep, args.sample_size, metrics, params_source, loss_name)

    def _build_mlp_for_fold(n_features, n_fit):
        """The same network the main path builds, for one inner fold. Mirrors the
        model-setup and Bayesian-transformation blocks above."""
        from loss_functions import get_loss_function
        if model_type == "mlp":
            m = MLPRegressor(input_size=n_features, hidden_size=params['hidden_size'],
                             num_hidden_layers=params['num_hidden_layers'],
                             dropout_rate=params['dropout_rate'])
        elif model_type == "residual_mlp":
            m = ResidualMLP(input_size=n_features, hidden_size=128, num_layers=3)
        elif model_type == "factorization_mlp":
            m = FactorizationMLP(input_size=n_features, hidden_size=128, factor_size=16)
        elif model_type == "mtl":
            m = MTLRegressionModel(input_size=n_features, hidden_size=128, num_tasks=1)

        if loss_name == 'heteroscedastic':
            if hasattr(m, 'fc_out'):
                m.fc_out = nn.Linear(m.fc_out.in_features, 2)
            elif hasattr(m, 'output_layer'):
                m.output_layer = nn.Linear(m.output_layer.in_features, 2)
        elif loss_name == 'evidential':
            if hasattr(m, 'fc_out'):
                m.fc_out = nn.Linear(m.fc_out.in_features, 4)
            elif hasattr(m, 'output_layer'):
                m.output_layer = nn.Linear(m.output_layer.in_features, 4)

        if args.bayesian_transformation == "full":
            m = apply_bayesian_transformation(m)
        elif args.bayesian_transformation == "last_layer":
            m = apply_bayesian_transformation_last_layer(m)
        elif args.bayesian_transformation == "variational":
            m = apply_bayesian_transformation_last_layer_variational(m)
        elif args.bayesian_transformation == "full_variational":
            m = apply_bayesian_transformation_full_variational(m)
        m.to(device)

        crit = get_loss_function(loss_name, **loss_kwargs)
        if args.bayesian_transformation in ("variational", "full_variational"):
            crit = VBLLLoss(m, n_data=n_fit)
        elif args.bayesian_transformation in ("full", "last_layer"):
            # The KL term, on the cross-fitting models too -- otherwise the
            # out-of-fold uncertainty comes from a different objective than the
            # test-set one it is compared with.
            crit = bnn_elbo_criterion(crit, m, n_fit)
        return m, crit

    def _train_mlp_for_fold(built, x_fit, y_fit, x_es, y_es):
        m, crit = built
        xt = torch.tensor(np.asarray(x_fit), dtype=torch.float32).to(device)
        yt = torch.tensor(np.asarray(y_fit), dtype=torch.float32).view(-1, 1).to(device)
        xe = torch.tensor(np.asarray(x_es), dtype=torch.float32).to(device)
        ye = torch.tensor(np.asarray(y_es), dtype=torch.float32).view(-1, 1).to(device)
        loader = TorchDataLoader(TensorDataset(xt, yt), batch_size=NEURAL_DEFAULTS['training']['batch_size'], shuffle=True)
        es_loader = TorchDataLoader(TensorDataset(xe, ye), batch_size=NEURAL_DEFAULTS['training']['batch_size'], shuffle=False)
        opt = torch.optim.Adam(m.parameters(), lr=params['lr'])
        train_nn(m, loader, es_loader, crit, opt, device, args, s, iteration,
                 file_no, 'oof_inner', rep)
        return m

    # STEP 3: Save uncertainty with calibration and decomposition
    if args.uncertainty and is_bayesian:
        save_uncertainty_values(
            y_pred_mean=y_pred,
            y_pred_std=y_pred_std_uncalibrated,
            y_true_original=y_test_original,
            y_true_noisy=y_test,
            filepath=args.filepath,
            model_name=full_model_name,
            rep=rep,
            sigma_noise=s,
            iteration=iteration,
            file_no=file_no,
            y_pred_std_calibrated=y_pred_std_calibrated,
            temperature=temperature,
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            split='test',
            **_held_out_noise_columns(train_noise, len(y_test), y_pred),
        )

        # The TRAINING molecules. This family fits on the training split alone;
        # validation is the early-stopping set and the calibration set, never a
        # fit row.
        def _fp(x_fit, y_fit, x_score):
            nv = max(1, len(y_fit) // 5)
            inner_model = _train_mlp_for_fold(
                _build_mlp_for_fold(x_fit.shape[1], len(y_fit) - nv),
                x_fit[nv:], y_fit[nv:], x_fit[:nv], y_fit[:nv])
            inner_model.eval()
            xs = torch.tensor(np.asarray(x_score), dtype=torch.float32).to(device)
            draws = []
            with torch.no_grad():
                for _ in range(num_samples):
                    out = inner_model(xs).cpu().numpy()
                    if loss_name in ('heteroscedastic', 'evidential'):
                        out = out[:, 0:1]
                    draws.append(out)
            draws = np.stack(draws, axis=0)
            return draws.mean(axis=0).flatten(), draws.std(axis=0).flatten()

        score_training_molecules_out_of_fold(
            _fp, x_train, y_train, train_noise, args, s, rep, iteration,
            iteration_seed, file_no, full_model_name,
            val_slice=None, restore_torch_rng=True)

    return metrics[3]

def train_rnn_variant_model(x_train, y_train, x_test, y_test, x_val, y_val, model_type, args, s, rep, iteration, iteration_seed, file_no, trial=None, train_noise=None):
    if model_type not in ["rnn", "gru"] or rep not in ['smiles', 'randomized_smiles']:
        raise ValueError("Invalid model type or representation for RNN/GRU training")

    params = {}
    params_source = 'default'

    if hasattr(args, 'use_best_params') and args.use_best_params and not args.tuning:
        best_params = load_best_hyperparameters(model_type, rep)
        if best_params is not None:
            params = best_params
            params_source = 'tuned'
            print(f"Using tuned hyperparameters for {model_type}-{rep}")

    if not params:
        if args.tuning:
            params['hidden_size'] = trial.suggest_categorical('hidden_size', [32, 64, 128, 256])
            params['num_layers'] = trial.suggest_int('num_layers', 1, 3)
            params['dropout_rate'] = trial.suggest_float('dropout_rate', 0.1, 0.5)
            params['lr'] = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
            params_source = 'tuning_trial'
        else:
            params['hidden_size'], params['num_layers'], params['dropout_rate'], params['lr'] = 128, 1, 0.2, 0.001
            params_source = 'default'

    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1).to(device)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1).to(device)

    if x_val is not None and y_val is not None:
        x_val_tensor = torch.tensor(x_val, dtype=torch.float32).unsqueeze(1).to(device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
        val_loader = TorchDataLoader(TensorDataset(x_val_tensor, y_val_tensor), batch_size=NEURAL_DEFAULTS['training']['batch_size'], shuffle=False)
    else:
        val_loader = None

    train_loader = TorchDataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=NEURAL_DEFAULTS['training']['batch_size'], shuffle=True)
    test_loader = TorchDataLoader(TensorDataset(x_test_tensor, y_test_tensor), batch_size=NEURAL_DEFAULTS['training']['batch_size'], shuffle=False)

    model = RNNRegressionModel(
        input_size=x_train.shape[1],
        hidden_size=params['hidden_size'],
        num_layers=params['num_layers']
    ) if model_type == "rnn" else GRURegressionModel(
        input_size=x_train.shape[1],
        hidden_size=params['hidden_size'],
        num_layers=params['num_layers']
    )
    criterion = nn.MSELoss()

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    trainer = ModelTrainer(model, lr=params['lr'])
    trainer.train(train_loader, val_loader, args, s, iteration, file_no, 'rnn', rep)

    y_pred_tensor = trainer.validate(test_loader)[1] 

    y_pred = np.argmax(y_pred_tensor, axis=1)

    metrics = calculate_regression_metrics(y_test, y_pred, logging=True)

    save_results(args.filepath, s, iteration, model_type, rep, args.sample_size, metrics,
                 params_source)

    return metrics[3]

"""
COMPLETE GRAPH MODEL TRAINING - ALL BUGS FIXED

Fixes:
1. Graph GP: Use Graph() constructor with STRING node labels (not atomic numbers)
2. GNN: Properly bundle noisy labels with graphs in custom dataset
3. Both: Use normalized targets for evaluation (same scale as training)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# GRAPH CONVERSION - FIXED
# =============================================================================

def smiles_to_grakel_graph(smiles):
    """
    Convert SMILES to grakel Graph object.
    
    CRITICAL: Node labels must be STRINGS (atomic symbols), not integers!
    CRITICAL: Use Graph() constructor, not array format!
    """
    from grakel import Graph
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumBonds() == 0:
        return None
    
    # Node labels as STRINGS (atomic symbols like 'C', 'N', 'O')
    node_labels = {atom.GetIdx(): atom.GetSymbol() for atom in mol.GetAtoms()}
    
    # Edge list
    edge_list = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_list.append((i, j))
    
    # Use Graph() constructor
    return Graph(edge_list, node_labels=node_labels)


def smiles_to_pyg_graph(smiles):
    """
    Convert SMILES to PyTorch Geometric Data object.
    
    Node features: [atomic_num, degree, formal_charge, is_aromatic]
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Node features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            int(atom.GetIsAromatic())
        ])
    
    x = torch.tensor(atom_features, dtype=torch.float)
    
    # Edge index (bidirectional)
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.zeros((2, 0), dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index)


# =============================================================================
# GNN DATASET - FIXED TO BUNDLE NOISY LABELS
# =============================================================================

class MolecularGraphDataset(Dataset):
    """
    Dataset that bundles PyG graphs with noisy labels.
    
    CRITICAL: This ensures noisy labels travel with graphs through shuffling!
    """
    def __init__(self, smiles_list, labels):
        self.data = []
        for smiles, label in zip(smiles_list, labels):
            graph = smiles_to_pyg_graph(smiles)
            if graph is not None:
                self.data.append((graph, label))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def collate_molecular_graphs(batch):
    """Custom collate function for MolecularGraphDataset."""
    graphs, labels = zip(*batch)
    batched_graph = Batch.from_data_list(graphs)
    labels = torch.tensor(labels, dtype=torch.float32)
    return batched_graph, labels


# =============================================================================
# GNN MODELS
# =============================================================================

# Removed 2026-08-27: a duplicate top-level definition that a later one in
# this file shadowed, so it never ran -- create_gnn_model (shadowed by the copy at 4106).
# scripts/test_no_shadowed_definitions.py fails if another one appears.


# =============================================================================
# TRAIN GNN 
# =============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool


# Removed 2026-08-27: a duplicate top-level definition that a later one in
# this file shadowed, so it never ran -- create_gnn_model (identical copy at 4106 shadows it).
# scripts/test_no_shadowed_definitions.py fails if another one appears.


def train_gnn(model_type, train_loader, test_loader, val_loader, args, s, 
              iteration, file_no, y_test_original_tensor, trial=None,
              y_train_noisy=None, y_test_noisy=None, y_val_noisy=None):
    """
    Train GNN using YOUR pipeline.
    
    Data objects in loaders have .y_noisy attached.
    """
    from utils import (calculate_regression_metrics, save_results,
                      save_uncertainty_values, calibrate_uncertainty_simple,
                      decompose_uncertainty_sampling)
    
    # Get num_node_features from first batch
    for batch in train_loader:
        num_node_features = batch.x.shape[1]
        break
    
    # Create model
    model = create_gnn_model(model_type, num_node_features, hidden_dim=128, num_layers=3, dropout=0.1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    _t = NEURAL_DEFAULTS['training']

    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=_t['lr'])
    criterion = nn.MSELoss()

    def _batch_targets(batch):
        # float(): the loader collates the per-graph scalar into a one-element
        # tensor, so reading it raw makes the target array (n, 1) against an
        # (n,) prediction -- which pearsonr refuses ('x and y must have the same
        # length along axis').
        return torch.tensor([float(data.y_noisy) for data in batch.to_data_list()],
                            dtype=torch.float32, device=device)

    # Early stopping, on the same rule as the tabular neural models.
    #
    # This loop used to be `for epoch in range(100)` with no validation loss in
    # it and no break, and `args.epochs` was never read. So the GNN fitted
    # corrupted labels for the whole budget while DNN, MLP and flexible_dnn
    # rolled back to their validation optimum, and any comparison of GNN
    # robustness against them was confounded by the stopping rule, in the
    # direction that makes the GNN look less robust (RERUN_PLAN.md 2.13).
    epochs = int(getattr(args, 'epochs', _t['epochs']) or _t['epochs'])
    patience = _t['patience']
    tolerance = _t['improvement_tolerance']
    best_val, patience_ctr, best_state, best_epoch = float('inf'), 0, None, -1

    print(f"Training {model_type.upper()} for up to {epochs} epochs "
          f"(patience {patience})...")
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            targets = _batch_targets(batch)
            optimizer.zero_grad()
            loss = criterion(model(batch), targets)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_sum, val_n = 0.0, 0
            for batch in val_loader:
                batch = batch.to(device)
                targets = _batch_targets(batch)
                val_sum += criterion(model(batch), targets).item() * targets.numel()
                val_n += targets.numel()
        val_loss = val_sum / max(val_n, 1)

        if val_loss < best_val - tolerance:
            best_val, patience_ctr, best_epoch = val_loss, 0, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    if _t['restore_best_weights'] and best_state is not None:
        model.load_state_dict(best_state)
    print(f"  stopped at epoch {epoch + 1}, restored epoch {best_epoch + 1} "
          f"(validation loss {best_val:.6f})")

    # Evaluation
    model.eval()
    
    if args.bayesian_transformation:
        # MC dropout
        predictions_list = []
        for _ in range(100):
            model.train()  # Keep dropout active
            preds = []
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    pred = model(batch).cpu().numpy()
                    preds.append(pred)
            predictions_list.append(np.concatenate(preds))
        
        predictions_array = np.array(predictions_list)
        predictions = predictions_array.mean(axis=0)
        
        # decompose_uncertainty_sampling(predictions_array, num_samples) returns
        # THREE values. This called it with one argument and unpacked two, so the
        # Bayesian branch of train_gnn raised TypeError on its first line and the
        # caller's blanket handler turned it into a missing row -- one of two
        # reasons no QM9 graph-model uncertainty has ever been written
        # (RERUN_PLAN.md 2.13).
        epistemic, aleatoric, _total = decompose_uncertainty_sampling(
            predictions_array, predictions_array.shape[0])
        
    else:
        # Deterministic
        preds = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                pred = model(batch).cpu().numpy()
                preds.append(pred)
        predictions = np.concatenate(preds)
        epistemic = None
        aleatoric = None
    
    # Calibrate on validation
    if epistemic is not None:
        val_preds_list = []
        for _ in range(100):
            model.train()
            val_preds = []
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    pred = model(batch).cpu().numpy()
                    val_preds.append(pred)
            val_preds_list.append(np.concatenate(val_preds))
        
        val_predictions_array = np.array(val_preds_list)
        val_predictions = val_predictions_array.mean(axis=0)
        val_epistemic, _val_aleatoric, _ = decompose_uncertainty_sampling(
            val_predictions_array, val_predictions_array.shape[0])
        
        # Get val targets
        val_targets = []
        for batch in val_loader:
            val_targets.extend([float(data.y_noisy) for data in batch.to_data_list()])
        val_targets = np.array(val_targets)
        
        n_cal = len(val_predictions) // 2
        temperature = calibrate_uncertainty_simple(
            val_predictions[:n_cal],
            val_epistemic[:n_cal],
            val_targets[:n_cal]
        )
        epistemic = epistemic * temperature
    else:
        temperature = 1.0
    
    # Get test targets (noisy, normalized)
    test_targets = []
    for batch in test_loader:
        test_targets.extend([float(data.y_noisy) for data in batch.to_data_list()])
    test_targets = np.array(test_targets)
    
    # CRITICAL: Evaluate on normalized targets (same scale as training)
    metrics = calculate_regression_metrics(test_targets, predictions)
    print(f"{model_type.upper()} - R²: {metrics[3]:.4f}, RMSE: {metrics[2]:.4f}")
    
    # Save
    model_name = f"{model_type}_bayesian" if args.bayesian_transformation else model_type
    # args.sample_size, like every other model. These two wrote len(test set)
    # into the same column, so `sample_size` meant the whole sample for most
    # rows and a tenth of it for the graph rows (RERUN_PLAN.md 2.13).
    save_results(args.filepath, s, iteration, model_name, 'graph', args.sample_size, metrics)
    
    if args.uncertainty:
        total_unc = epistemic if epistemic is not None else np.zeros_like(predictions)
        # The clean and the noisy label are the SAME array here, and always were:
        # a graph run has one label column. Under the old writer that made the
        # regressed-out "injected noise" exactly zero regardless of anything else.
        # The regression is gone; this is a test-split call, so the recorded noise
        # is not needed and the column is written as exactly 0.0.
        save_uncertainty_values(
            predictions, total_unc, test_targets, test_targets,
            args.filepath, model_name, 'graph', s, iteration, file_no,
            y_pred_std_calibrated=epistemic, temperature=temperature,
            epistemic_uncertainty=epistemic, aleatoric_uncertainty=aleatoric,
            split='test'
        )
    
    return metrics[3]

# =============================================================================
# TRAIN GRAPH GP - FIXED
# =============================================================================

"""
train_graph_gp for YOUR pipeline
Works with lists of PyG Data objects that have .y_noisy and .smiles
"""

import numpy as np
import torch
from rdkit import Chem


# Removed 2026-08-27: a duplicate top-level definition that a later one in
# this file shadowed, so it never ran -- pyg_to_grakel (identical copy at 4070 shadows it).
# scripts/test_no_shadowed_definitions.py fails if another one appears.


def train_graph_gp(train_graphs, y_train_noisy, test_graphs, y_test_noisy,
                   val_graphs, y_val_noisy, args, s, iteration, file_no,
                   y_test_original_tensor, trial=None):
    """
    Train Graph GP using YOUR pipeline.
    
    Args:
        train_graphs: List of PyG Data objects with .y_noisy
        y_train_noisy: Ignored (we use data.y_noisy from objects)
        test_graphs: List of PyG Data objects with .y_noisy
        y_test_noisy: Ignored
        val_graphs: List of PyG Data objects with .y_noisy
        y_val_noisy: Ignored
    """
    from grakel.kernels import WeisfeilerLehman, VertexHistogram
    from utils import (calculate_regression_metrics, save_results,
                      save_uncertainty_values, calibrate_uncertainty_simple,
                      decompose_uncertainty_gp)
    
    print(f"Converting {len(train_graphs)} molecules to grakel format...")
    
    # Convert PyG to grakel
    train_grakel = []
    train_labels = []
    for data in train_graphs:
        g = pyg_to_grakel(data)
        if g is not None:
            train_grakel.append(g)
            train_labels.append(float(data.y_noisy))
    
    train_labels = torch.tensor(train_labels, dtype=torch.float32)
    
    print(f"Computing WL kernel matrix for {len(train_grakel)} valid molecules...")
    
    # Initialize kernel
    kernel_obj = WeisfeilerLehman(n_iter=5, base_graph_kernel=VertexHistogram, normalize=True)
    
    # Compute kernel matrix
    K_train = kernel_obj.fit_transform(train_grakel)
    K_train_tensor = torch.tensor(K_train, dtype=torch.float32)
    
    print(f"Kernel stats: min={K_train_tensor.min():.4f}, max={K_train_tensor.max():.4f}, var={K_train_tensor.var():.4f}")
    
    # Add jitter
    jitter = 1e-3
    K_train_tensor = K_train_tensor + jitter * torch.eye(K_train_tensor.shape[0])
    
    # Ensure positive definiteness
    eigenvalues = torch.linalg.eigvalsh(K_train_tensor)
    min_eig = eigenvalues.min()
    if min_eig < 1e-6:
        extra_jitter = 1e-6 - min_eig + 1e-3
        K_train_tensor = K_train_tensor + extra_jitter * torch.eye(K_train_tensor.shape[0])
        print(f"Added extra jitter: {extra_jitter:.6f}")
    
    print(f"Training GP on kernel matrix of shape {K_train_tensor.shape}...")
    
    # Select noise parameter
    best_noise = 0.1
    best_nll = float('inf')
    
    for noise_candidate in [0.001, 0.01, 0.1, 0.5, 1.0]:
        K_noisy = K_train_tensor + noise_candidate * torch.eye(len(train_labels))
        try:
            L = torch.linalg.cholesky(K_noisy)
            alpha = torch.cholesky_solve(train_labels.unsqueeze(-1), L).squeeze()
            nll = 0.5 * (train_labels @ alpha) + torch.log(L.diag()).sum() + 0.5 * len(train_labels) * np.log(2 * np.pi)
            print(f"  Noise {noise_candidate:.4f}: NLL={nll:.4f}")
            if nll < best_nll:
                best_nll = nll
                best_noise = noise_candidate
        except Exception as e:
            print(f"  Noise {noise_candidate:.4f}: Failed - {e}")
    
    print(f"Selected noise: {best_noise:.4f}")
    
    # Predict on test
    print(f"Converting {len(test_graphs)} test molecules to grakel format...")
    test_grakel = []
    test_labels = []
    for data in test_graphs:
        g = pyg_to_grakel(data)
        if g is not None:
            test_grakel.append(g)
            test_labels.append(float(data.y_noisy))
    
    test_labels = np.array(test_labels)
    
    print(f"Computing kernel matrix between {len(test_grakel)} test and {len(train_grakel)} train graphs...")
    K_test_train = kernel_obj.transform(test_grakel)
    K_test_train_tensor = torch.tensor(K_test_train, dtype=torch.float32)
    
    # GP prediction
    K_noisy = K_train_tensor + best_noise * torch.eye(len(train_labels))
    L = torch.linalg.cholesky(K_noisy)
    alpha = torch.cholesky_solve(train_labels.unsqueeze(-1), L).squeeze()
    
    predictions = (K_test_train_tensor @ alpha).numpy()

    # The GP POSTERIOR variance, per molecule:
    #     var(x*) = k(x*, x*) - k*^T (K + sigma^2 I)^-1 k*
    #
    # `std` used to be `np.ones(len(test_grakel)) * np.sqrt(best_noise)` -- one
    # constant repeated for every molecule, which is the likelihood noise, not a
    # posterior. Every per-molecule statistic computed from it was degenerate,
    # and a "Graph GP uncertainty" column held one number (RERUN_PLAN.md 2.13).
    # The Cholesky factor is already here, so this costs one triangular solve.
    #
    # The kernel is built with normalize=True, so k(x*, x*) = 1 by construction.
    _v = torch.linalg.solve_triangular(L, K_test_train_tensor.T, upper=False)
    _post_var = (1.0 - (_v ** 2).sum(dim=0)).clamp_min(0.0)
    std = torch.sqrt(_post_var + best_noise).numpy()
    
    # Validation for calibration
    print(f"Converting {len(val_graphs)} validation molecules to grakel format...")
    val_grakel = []
    val_labels = []
    for data in val_graphs:
        g = pyg_to_grakel(data)
        if g is not None:
            val_grakel.append(g)
            val_labels.append(float(data.y_noisy))
    
    val_labels = np.array(val_labels)
    
    K_val_train = kernel_obj.transform(val_grakel)
    K_val_train_tensor = torch.tensor(K_val_train, dtype=torch.float32)
    
    val_predictions = (K_val_train_tensor @ alpha).numpy()
    _vv = torch.linalg.solve_triangular(L, K_val_train_tensor.T, upper=False)
    val_std = torch.sqrt((1.0 - (_vv ** 2).sum(dim=0)).clamp_min(0.0)
                         + best_noise).numpy()
    
    # decompose_uncertainty_gp returns THREE values. These unpacked two, so
    # train_graph_gp raised ValueError here every time it was reached
    # (RERUN_PLAN.md 2.13).
    epistemic, aleatoric, _total = decompose_uncertainty_gp(std, best_noise)
    val_epistemic, val_aleatoric, _ = decompose_uncertainty_gp(val_std, best_noise)
    
    # Calibrate
    n_cal = len(val_predictions) // 2
    temperature = calibrate_uncertainty_simple(
        val_predictions[:n_cal],
        val_epistemic[:n_cal],
        val_labels[:n_cal]
    )
    
    epistemic = epistemic * temperature
    total = np.sqrt(epistemic**2 + aleatoric**2)
    
    # CRITICAL: Evaluate on normalized targets (same scale as training)
    metrics = calculate_regression_metrics(test_labels, predictions)
    print(f"Graph GP - R²: {metrics[3]:.4f}, RMSE: {metrics[2]:.4f}")
    
    # Save
    # args.sample_size, like every other model. These two wrote len(test set)
    # into the same column, so `sample_size` meant the whole sample for most
    # rows and a tenth of it for the graph rows (RERUN_PLAN.md 2.13).
    save_results(args.filepath, s, iteration, 'graph_gp', 'graph', args.sample_size, metrics)
    
    if args.uncertainty:
        # Same array twice, same reason as train_gnn above. Test split, no
        # recorded noise needed, injected_noise written as exactly 0.0.
        save_uncertainty_values(
            predictions, total, test_labels, test_labels,
            args.filepath, 'graph_gp', 'graph', s, iteration, file_no,
            y_pred_std_calibrated=total, temperature=temperature,
            epistemic_uncertainty=epistemic, aleatoric_uncertainty=aleatoric,
            split='test'
        )
    
    return metrics[3]

def pyg_to_grakel(data):
    """
    Convert PyG Data object to grakel Graph.
    
    CRITICAL: Use STRING atomic symbols, not atomic numbers!
    """
    from grakel import Graph
    # Map atomic numbers to symbols
    atomic_num_to_symbol = {
        1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'
    }
    
    # Get atomic numbers from data.x (first column)
    atomic_numbers = data.x[:, 0].long().tolist()
    
    # Convert to string symbols
    node_labels = {}
    for i, atomic_num in enumerate(atomic_numbers):
        node_labels[i] = atomic_num_to_symbol.get(atomic_num, str(atomic_num))
    
    # Get edges
    edge_index = data.edge_index.t().tolist()
    edge_list = [(u, v) for u, v in edge_index]
    
    # Remove duplicate edges (undirected graph)
    edge_set = set()
    for u, v in edge_list:
        if u < v:
            edge_set.add((u, v))
        else:
            edge_set.add((v, u))
    edge_list = list(edge_set)
    
    # Create Graph
    return Graph(edge_list, node_labels=node_labels)


def create_gnn_model(model_type, num_node_features, hidden_dim=128, num_layers=3, dropout=0.1):
    """Create GNN model."""
    class GNNRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.convs = nn.ModuleList()
            
            if model_type == 'gcn':
                self.convs.append(GCNConv(num_node_features, hidden_dim))
                for _ in range(num_layers - 1):
                    self.convs.append(GCNConv(hidden_dim, hidden_dim))
            
            elif model_type == 'gat':
                heads = 4
                self.convs.append(GATConv(num_node_features, hidden_dim // heads, heads=heads))
                for _ in range(num_layers - 1):
                    self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads))
            
            elif model_type == 'gin':
                nn1 = nn.Sequential(nn.Linear(num_node_features, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
                self.convs.append(GINConv(nn1))
                for _ in range(num_layers - 1):
                    nn_layer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
                    self.convs.append(GINConv(nn_layer))
            
            self.dropout = dropout
            self.regression_head = nn.Linear(hidden_dim, 1)
        
        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                if i < len(self.convs) - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
            
            x = global_mean_pool(x, batch)
            return self.regression_head(x).squeeze()
    
    return GNNRegressor()


def train_conformal_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, file_no, base_model_type, calibration_size, y_test_original, trial=None, train_noise=None):
    # `calibration_size` is accepted and NOT used: the conformity scores below are
    # computed over the whole held-out validation split. That is deliberate now --
    # no model trains on validation, so all of it is a valid calibration set and
    # more points is the better estimator. The flag that fed this is commented
    # out in process_and_train.py, and callers pass None (RERUN_PLAN.md 2.13).
    from quantile_forest import RandomForestQuantileRegressor
    from torchcp.regression.predictor import SplitPredictor, ACIPredictor
    from torchcp.regression.score import ABS
    from torch.utils.data import TensorDataset, DataLoader
    
    params = {}
    params_source = 'default'
    conformal_model_name = f'conformal_{base_model_type}'
    
    # Load best params if available
    if hasattr(args, 'use_best_params') and args.use_best_params and not args.tuning:
        best_params = load_best_hyperparameters(conformal_model_name, rep)
        if best_params is not None:
            params = best_params
            params_source = 'tuned'
            print(f"Using tuned hyperparameters for {conformal_model_name}-{rep}")
    
    # Hyperparameter tuning or defaults
    if not params:
        if args.tuning and trial is not None:
            # Conformal-specific parameters
            alpha = trial.suggest_float('alpha', 0.05, 0.2)
            predictor_type = trial.suggest_categorical('predictor_type', ['split', 'aci'])            
            # ACI-specific parameter
            if predictor_type == 'aci':
                params['gamma'] = trial.suggest_float('gamma', 0.001, 0.1, log=True)
            
            # Base model hyperparameters
            if base_model_type == 'rf':
                params['n_estimators'] = trial.suggest_int('n_estimators', 50, 500)
                params['max_depth'] = trial.suggest_int('max_depth', 5, 20)
                params['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 20)
            elif base_model_type == 'dnn':
                params['hidden_size1'] = trial.suggest_categorical('hidden_size1', [64, 128, 256])
                params['hidden_size2'] = trial.suggest_categorical('hidden_size2', [32, 64, 128])
                params['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            elif base_model_type == 'xgboost':
                params['n_estimators'] = trial.suggest_int('n_estimators', 50, 500)
                params['max_depth'] = trial.suggest_int('max_depth', 3, 10)
                params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
            elif base_model_type == 'qrf':
                params['n_estimators'] = trial.suggest_int('n_estimators', 50, 500)
                params['max_depth'] = trial.suggest_int('max_depth', 5, 20)
                params['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 20)
            elif base_model_type == 'gauche':
                params['kernel_name'] = trial.suggest_categorical('kernel', [
                    'Tanimoto', 'BraunBlanquet', 'Dice', 'Faith', 'Forbes',
                    'InnerProduct', 'Intersection', 'MinMax', 'Otsuka'
                ])
                params['outputscale'] = trial.suggest_float('outputscale', 0.1, 10.0, log=True)
                params['likelihood_noise'] = trial.suggest_float('likelihood_noise', 1e-4, 0.1, log=True)
            params_source = 'tuning_trial'
        else:
            # The default branch used to leave `params` EMPTY, so every base
            # estimator below was built from the library's own defaults --
            # conformal_rf on max_features=1.0 instead of the spec's 0.3,
            # conformal_qrf on 100 trees instead of 300, conformal_xgboost on the
            # booster's learning_rate 0.3 instead of 0.1. The conformal models
            # were therefore not wrappers around the same base models they were
            # compared against, and the "rho > 0.99 with rf" case for excluding
            # them rested on a different forest (RERUN_PLAN.md 2.13).
            _spec_key = {'rf': 'rf', 'qrf': 'qrf', 'xgboost': 'xgboost'}.get(
                base_model_type)
            if _spec_key is not None:
                params = sklearn_params(_spec_key)
                params_source = 'default'
            else:
                params_source = 'default'
            predictor_type = 'split'
    
    # Extract conformal parameters
    alpha_list = args.alpha if hasattr(args, 'alpha') else [0.1]
    predictor_type = params.pop('predictor_type', 'split')
    gamma = params.pop('gamma', 0.01)  # For ACI
    
    # Train base model based on type
    if base_model_type in ['rf', 'xgboost']:
        if base_model_type == 'rf':
            from sklearn.ensemble import RandomForestRegressor
            base_model = RandomForestRegressor(random_state=iteration_seed, **params)
        else:
            from xgboost import XGBRegressor
            base_model = XGBRegressor(random_state=iteration_seed, **params)
        
        # Train on train set
        base_model.fit(x_train, y_train)
        
    elif base_model_type == 'qrf':
        base_model = RandomForestQuantileRegressor(random_state=iteration_seed, **params)
        base_model.fit(x_train, y_train)
        
    elif base_model_type == 'gauche':
        kernel_map = {
            'Tanimoto': gauche.kernels.fingerprint_kernels.tanimoto_kernel.TanimotoKernel,
            'BraunBlanquet': gauche.kernels.fingerprint_kernels.braun_blanquet_kernel.BraunBlanquetKernel,
            'Dice': gauche.kernels.fingerprint_kernels.dice_kernel.DiceKernel,
            'Faith': gauche.kernels.fingerprint_kernels.faith_kernel.FaithKernel,
            'Forbes': gauche.kernels.fingerprint_kernels.forbes_kernel.ForbesKernel,
            'InnerProduct': gauche.kernels.fingerprint_kernels.inner_product_kernel.InnerProductKernel,
            'Intersection': gauche.kernels.fingerprint_kernels.intersection_kernel.IntersectionKernel,
            'MinMax': gauche.kernels.fingerprint_kernels.minmax_kernel.MinMaxKernel,
            'Otsuka': gauche.kernels.fingerprint_kernels.otsuka_kernel.OtsukaKernel,
        }
        
        # The conformity scores below are computed on validation, so validation
        # must stay out of the fit -- and no model stacks it in any case (settled
        # 2026-08-27, RERUN_PLAN.md 2.12).
        x_full = x_train
        y_full = y_train

        x_train_tensor = torch.from_numpy(x_full).double()
        x_test_tensor = torch.from_numpy(x_test).double()
        y_train_tensor = torch.from_numpy(y_full).double()
        
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise=params.get('likelihood_noise', GP_DEFAULTS['likelihood_noise']))
        kernel_class = kernel_map[params.get('kernel_name', 'Tanimoto')]
        base_model = Gauche(x_train_tensor, y_train_tensor, likelihood, kernel_class)
        if GP_DEFAULTS['apply_outputscale']:
            base_model.covar_module.outputscale = params.get(
                'outputscale', GP_DEFAULTS['outputscale'])

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, base_model)
        gp_fit_method = fit_gp_with_fallback(mll, base_model, likelihood,
                                             x_train_tensor, y_train_tensor)
        
    elif base_model_type == 'dnn':
        learning_rate = params.pop('learning_rate', 0.001)
        
        # Remove dropout if it's in params (not supported by DNNRegressionModel)
        params.pop('dropout', None)
        
        base_model = DNNRegressionModel(
            input_size=x_train.shape[1], 
            hidden_size1=params.get('hidden_size1', 128), 
            hidden_size2=params.get('hidden_size2', 64)
        )
        base_model.to(device)
        
        x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
        x_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
        
        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=NEURAL_DEFAULTS['training']['batch_size'], shuffle=True)
        val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=NEURAL_DEFAULTS['training']['batch_size'], shuffle=False)
        
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(base_model.parameters(), lr=learning_rate)
        
        train_nn(base_model, train_loader, val_loader, criterion, optimizer, device, args, s, iteration, file_no, f'conformal_dnn', rep)

        # Save calibration metadata
        alpha_list = args.alpha if hasattr(args, 'alpha') and args.alpha else [0.1]
        save_calibration_metadata(
            filepath=args.filepath,
            model_name=f'conformal_{base_model_type}_split',
            rep=rep,
            sigma_noise=s,
            iteration=iteration,
            n_train=len(x_train),
            n_cal=len(x_val),
            n_val=0,
            n_test=len(x_test),
            alpha_list=alpha_list
        )
    
    # --- Step 1: Get predictions on calibration and test sets (point estimates) ---
    if base_model_type in ['rf', 'xgboost']:
        # Standard regressors: mean prediction
        y_val_pred  = np.asarray(base_model.predict(x_val)).reshape(-1)
        y_test_pred = np.asarray(base_model.predict(x_test)).reshape(-1)

    elif base_model_type == 'qrf':
        # Request median; handle 1D vs 2D outputs robustly
        _val = base_model.predict(x_val, quantiles=[0.5])
        _test = base_model.predict(x_test, quantiles=[0.5])
        _val = np.asarray(_val)
        _test = np.asarray(_test)
        y_val_pred  = (_val[:, 0]  if _val.ndim  == 2 else _val).reshape(-1)
        y_test_pred = (_test[:, 0] if _test.ndim == 2 else _test).reshape(-1)

    elif base_model_type == 'gauche':
        x_val_tensor = torch.from_numpy(x_val).double()
        x_test_tensor = torch.from_numpy(x_test).double()
        base_model.eval()
        with torch.no_grad():
            val_preds = base_model(x_val_tensor)
            test_preds = base_model(x_test_tensor)
            y_val_pred  = np.asarray(val_preds.mean.numpy()).reshape(-1)
            y_test_pred = np.asarray(test_preds.mean.numpy()).reshape(-1)

    elif base_model_type == 'dnn':
        x_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(device)
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
        base_model.eval()
        with torch.no_grad():
            y_val_pred  = base_model(x_val_tensor).cpu().numpy().reshape(-1)
            y_test_pred = base_model(x_test_tensor).cpu().numpy().reshape(-1)

    else:
        raise ValueError(f"Unsupported base_model_type: {base_model_type}")
    
    # Step 2: Calculate conformity scores on calibration set
    conformity_scores = np.abs(y_val - y_val_pred)

    y_val  = np.asarray(y_val).reshape(-1)
    y_test = np.asarray(y_test).reshape(-1)
    assert y_val_pred.shape  == y_val.shape,  (y_val_pred.shape,  y_val.shape)
    assert y_test_pred.shape == y_test.shape, (y_test_pred.shape, y_test.shape)
    
    all_results = []
    for alpha in alpha_list:
        # Step 3: Calculate quantile for conformal prediction
        if predictor_type == 'split':
            # Standard split conformal
            n_cal = len(conformity_scores)
            q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
            q_level = min(q_level, 1.0)
            quantile = np.quantile(conformity_scores, q_level)
            
        elif predictor_type == 'aci':
            # For ACI, we use adaptive quantile (simplified version)
            # In a full implementation, this would update online
            n_cal = len(conformity_scores)
            # Start with standard quantile and would adapt during online phase
            q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
            q_level = min(q_level, 1.0)
            quantile = np.quantile(conformity_scores, q_level)
            # Note: Full ACI would update this quantile adaptively as we see new data
        
        # Step 4: Generate prediction intervals
        y_lower = y_test_pred - quantile
        y_upper = y_test_pred + quantile
        
        # Step 5: Calculate coverage and metrics
        coverage = np.mean((y_test >= y_lower) & (y_test <= y_upper))
        avg_interval_size = np.mean(y_upper - y_lower)
        
        print(f"Conformal Prediction Results:")
        print(f"  Coverage: {coverage:.4f} (target: {1-alpha:.4f})")
        print(f"  Average Interval Size: {avg_interval_size:.4f}")
        
        # Calculate regression metrics
        metrics = calculate_regression_metrics(y_test, y_test_pred, logging=True)
        
        model_name = f'conformal_{base_model_type}_{predictor_type}'
        
        if args.uncertainty:
            save_conformal_intervals(
                y_pred=y_test_pred,
                y_lower=y_lower,
                y_upper=y_upper,
                y_true=y_test,
                filepath=args.filepath,
                model_name=model_name,
                rep=rep,
                sigma_noise=s,
                iteration=iteration,
                file_no=file_no,
                alpha=alpha
            )
            
            save_conformal_intervals(
                y_pred=y_test_pred,
                y_lower=y_lower,
                y_upper=y_upper,
                y_true=y_test,
                filepath=args.filepath,
                model_name=model_name,
                rep=rep,
                sigma_noise=s,
                iteration=iteration,
                file_no=file_no,
                alpha=alpha
            )
        
    # Save metrics once (outside alpha loop) using first iteration's metrics
    save_results(args.filepath, s, iteration, model_name, rep, args.sample_size, metrics, params_source)
    
    return metrics[3]  # Return R²

def train_conformal_graph_model(train_loader, test_loader, val_loader, args, s, iteration, file_no, base_model_type, calibration_size, y_test_original, trial=None,
                                y_train_noisy=None, y_test_noisy=None, y_val_noisy=None):
    """
    Conformal prediction for graph models.
    Note: y_train_noisy, y_test_noisy, y_val_noisy are the noisy+normalized targets from Rust.
    """
    from torchcp.regression.predictor import SplitPredictor, ACIPredictor
    from torchcp.regression.score import ABS
    from torch_geometric.loader import DataLoader as GeometricDataLoader
    
    params = {}
    params_source = 'default'
    conformal_model_name = f'conformal_{base_model_type}'
    
    # Load best params if available
    if hasattr(args, 'use_best_params') and args.use_best_params and not args.tuning:
        best_params = load_best_hyperparameters(conformal_model_name, 'graph')
        if best_params is not None:
            params = best_params
            params_source = 'tuned'
            print(f"Using tuned hyperparameters for {conformal_model_name}-graph")
    
    if not params:
        if trial is not None:
            # Conformal parameters
            alpha = trial.suggest_float('alpha', 0.05, 0.2)
            predictor_type = trial.suggest_categorical('predictor_type', ['split', 'aci'])
            params['alpha'] = alpha
            params['predictor_type'] = predictor_type
            
            # ACI-specific
            if predictor_type == 'aci':
                params['gamma'] = trial.suggest_float('gamma', 0.001, 0.1, log=True)
            
            # Base model parameters
            dim_h = trial.suggest_int('dim_h', 32, 256, step=32)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            
            params['dim_h'] = dim_h
            params['learning_rate'] = learning_rate
            params_source = 'tuning_trial'
        else:
            # Default parameters
            dim_h = 64
            alpha = 0.1
            learning_rate = 0.001
            predictor_type = 'split'
            
            params['dim_h'] = dim_h
            params['alpha'] = alpha
            params['learning_rate'] = learning_rate
            params['predictor_type'] = predictor_type
            params_source = 'default'
    
    # Extract parameters
    dim_h = params.pop('dim_h', 64)
    alpha = params.pop('alpha', 0.1)
    learning_rate = params.pop('learning_rate', 0.001)
    predictor_type = params.pop('predictor_type', 'split')
    gamma = params.pop('gamma', 0.01)
    
    # Remove any dropout or num_layers if they exist (not supported by your GNN classes)
    params.pop('dropout', None)
    params.pop('num_layers', None)
    
    # First, we need to update the graph data objects with the correct y values
    # Extract the original data from the loaders
    train_data_list = []
    for data in train_loader:
        train_data_list.extend([d for d in data.to_data_list()])
    
    val_data_list = []
    for data in val_loader:
        val_data_list.extend([d for d in data.to_data_list()])
    
    test_data_list = []
    for data in test_loader:
        test_data_list.extend([d for d in data.to_data_list()])
    
    # Update y values with the noisy targets
    if y_train_noisy is not None:
        for i, data in enumerate(train_data_list):
            data.y = y_train_noisy[i].unsqueeze(0) if y_train_noisy[i].dim() == 0 else y_train_noisy[i]
    
    if y_val_noisy is not None:
        for i, data in enumerate(val_data_list):
            data.y = y_val_noisy[i].unsqueeze(0) if y_val_noisy[i].dim() == 0 else y_val_noisy[i]
    
    if y_test_noisy is not None:
        for i, data in enumerate(test_data_list):
            data.y = y_test_noisy[i].unsqueeze(0) if y_test_noisy[i].dim() == 0 else y_test_noisy[i]
    
    # Create new DataLoaders with updated y values
    train_loader_updated = GeometricDataLoader(train_data_list, batch_size=64, shuffle=True)
    val_loader_updated = GeometricDataLoader(val_data_list, batch_size=64, shuffle=False)
    test_loader_updated = GeometricDataLoader(test_data_list, batch_size=64, shuffle=False)
    
    # The comment that used to sit here said "use only dim_h since that's what
    # your GNN classes accept". It was true of a `class GCN` that a SECOND
    # definition later in this file shadowed; the live GCN and GIN take
    # (num_node_features, hidden_dim, ...) and `GIN(dim_h=...)` raises
    # TypeError. So this path has never once constructed a model -- the failure
    # was swallowed by the caller's `except Exception` and came out as a missing
    # result row (RERUN_PLAN.md 2.13). The live classes also return
    # (prediction, embedding), which train_epochs cannot consume: it does
    # `out.detach().cpu().numpy()[:, 0]`.
    #
    # Refused by name rather than left to fail as a missing row. The conformal
    # graph wrapper needs deciding on, not patching blind: it is tier 4 in the
    # job generator and has never produced a number.
    raise NotImplementedError(
        f"the conformal graph wrapper (base model {base_model_type!r}) has "
        f"never been able to run. It builds its base network as "
        f"GIN(dim_h=...)/GCN(dim_h=...), a signature no live class in this file "
        f"has, and the live classes return (prediction, embedding) where "
        f"train_epochs expects a (batch, 1) tensor. See RERUN_PLAN.md 2.13.")
    
    base_model.to(device)
    
    # Train the base model
    train_loss, val_loss, train_target, train_y_target, trained_model = train_epochs(
        args.epochs, base_model, train_loader_updated, val_loader_updated, args, s, iteration, file_no, 
        f'conformal_{base_model_type}',
        y_train_noisy=y_train_noisy, y_val_noisy=y_val_noisy, learning_rate=learning_rate
    )
    
    # Create wrapper for torch-cp
    class GraphModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, batch):
            """
            batch: PyTorch Geometric batch object
            """
            self.model.eval()
            with torch.no_grad():
                output = self.model(batch)
                if output.dim() > 1:
                    output = output.squeeze()
                return output
    
    wrapped_model = GraphModelWrapper(trained_model)
    wrapped_model.eval()
    
    # Create score function
    score_function = ABS()
    
    # Create and calibrate predictor
    if predictor_type == 'split':
        # Create a custom predictor that works with graph data
        predictor = SplitPredictor(score_function=score_function, model=wrapped_model)
        
        # Manual calibration for graph data
        # We need to compute calibration scores manually because torch-cp's default calibrate
        # expects (X, y) tuples, but we have graph batches
        
        all_cal_preds = []
        all_cal_y = []
        
        wrapped_model.eval()
        with torch.no_grad():
            for batch in val_loader_updated:
                batch = batch.to(device)
                preds = wrapped_model(batch)
                all_cal_preds.append(preds.cpu())
                all_cal_y.append(batch.y.cpu())
        
        all_cal_preds = torch.cat(all_cal_preds)
        all_cal_y = torch.cat(all_cal_y)
        
        # Compute conformity scores
        scores = score_function(all_cal_preds, all_cal_y)
        
        # Compute quantile
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        q_level = min(q_level, 1.0)
        q_hat = torch.quantile(scores, q_level)
        
        # Store the quantile in the predictor
        predictor.q_hat = q_hat
        
    elif predictor_type == 'aci':
        # For ACI with graphs, we need a different approach
        # ACI requires online/adaptive learning
        predictor = ACIPredictor(score_function=score_function, model=wrapped_model, gamma=gamma)
        
        # Similar manual calibration
        all_cal_preds = []
        all_cal_y = []
        
        wrapped_model.eval()
        with torch.no_grad():
            for batch in val_loader_updated:
                batch = batch.to(device)
                preds = wrapped_model(batch)
                all_cal_preds.append(preds.cpu())
                all_cal_y.append(batch.y.cpu())
        
        all_cal_preds = torch.cat(all_cal_preds)
        all_cal_y = torch.cat(all_cal_y)
        
        scores = score_function(all_cal_preds, all_cal_y)
        
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        q_level = min(q_level, 1.0)
        q_hat = torch.quantile(scores, q_level)
        
        predictor.q_hat = q_hat
        predictor.alpha = alpha
    
    # Evaluate on test set
    all_test_preds = []
    all_test_y = []
    all_test_lower = []
    all_test_upper = []
    
    wrapped_model.eval()
    with torch.no_grad():
        for batch in test_loader_updated:
            batch = batch.to(device)
            preds = wrapped_model(batch)
            
            # Compute prediction intervals
            if hasattr(predictor, 'q_hat'):
                lower = preds - predictor.q_hat
                upper = preds + predictor.q_hat
            else:
                # Fallback
                lower = preds - 1.0
                upper = preds + 1.0
            
            all_test_preds.append(preds.cpu())
            all_test_y.append(batch.y.cpu())
            all_test_lower.append(lower.cpu())
            all_test_upper.append(upper.cpu())
    
    y_pred = torch.cat(all_test_preds).numpy().flatten()
    y_test = torch.cat(all_test_y).numpy().flatten()
    y_lower = torch.cat(all_test_lower).numpy().flatten()
    y_upper = torch.cat(all_test_upper).numpy().flatten()
    
    # Calculate metrics
    logging_flag = args.distribution not in ["domain_mpnn", "domain_tanimoto"]
    if not logging_flag:
        calculate_domain_metrics(y_pred, y_test, domain_labels_subset, target_domain)
    # (y_test, prediction), in that order. It was the other way round, and
    # r2_score is not symmetric: r2_score(y_pred, y_test) measures how well the
    # TRUTH explains the PREDICTION. This function's return value is what Optuna
    # maximises, so the search optimised the reversed statistic too
    # (RERUN_PLAN.md 2.13). MAE, MSE, RMSE and Pearson are symmetric and were
    # unaffected.
    metrics = calculate_regression_metrics(y_test, y_pred, logging=logging_flag)
    
    # Calculate coverage
    coverage = np.mean((y_test >= y_lower) & (y_test <= y_upper))
    avg_interval_size = np.mean(y_upper - y_lower)
    
    print(f"Conformal Prediction Results:")
    print(f"  Coverage: {coverage:.4f} (target: {1-alpha:.4f})")
    print(f"  Average Interval Size: {avg_interval_size:.4f}")
    
    model_name = f'conformal_{base_model_type}_{predictor_type}'
    save_results(args.filepath, s, iteration, model_name, 'graph', args.sample_size, metrics, params_source)
    
    if args.uncertainty:
        interval_width = y_upper - y_lower
        # z for the requested alpha, from the normal quantile -- not two
        # hardcoded numbers that are right for alpha 0.1 and 0.05 and wrong for
        # everything else (RERUN_PLAN.md 2.13). A conformal interval is not
        # Gaussian, so this is a pseudo-standard-deviation either way; it is at
        # least the right pseudo-standard-deviation now.
        from scipy.stats import norm as _norm
        y_pred_std = interval_width / (2 * _norm.ppf(1 - alpha / 2))
        
        save_uncertainty_values(
            y_pred_mean=y_pred,
            y_pred_std=y_pred_std,
            y_true_original=y_test_original.cpu().numpy().flatten(),
            y_true_noisy=y_test,
            filepath=args.filepath,
            model_name=model_name,
            rep='graph',
            sigma_noise=s,
            iteration=iteration,
            file_no=file_no,
        )
        
        save_conformal_intervals(
            y_pred=y_pred,
            y_lower=y_lower,
            y_upper=y_upper,
            y_true=y_test,
            filepath=args.filepath,
            model_name=model_name,
            rep='graph',
            sigma_noise=s,
            iteration=iteration,
            file_no=file_no,
            alpha=alpha
        )
    
    return metrics[3]  # Return R²

def load_best_hyperparameters(model_type, rep, results_dir='results'):
    """
    Load best hyperparameters from master config if available
    Returns: dict of hyperparameters or None if not found

    `results_dir` is resolved against THIS FILE, not the working directory. The
    job scripts `cd scripts` before running, and `scripts/results/` does not
    exist, so a relative path meant the tuned branch could never fire -- and it
    said nothing, it just fell through to the defaults and reset params_source
    to 'default' (RERUN_PLAN.md 2.13). A job asked for tuned parameters and got
    library ones, with nothing on the row to show it.
    """
    if not os.path.isabs(results_dir):
        results_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            results_dir)
    master_file = os.path.join(results_dir, 'master_tuned_hyperparameters.json')
    decisions_file = os.path.join(results_dir, 'hyperparameter_decisions.json')

    missing = [f for f in (master_file, decisions_file) if not os.path.exists(f)]
    if missing:
        # Say so, by name. Degrading quietly to the defaults is what hid this.
        print(f"      [tuned params] not used for {model_type}/{rep}: "
              f"{', '.join(missing)} does not exist", flush=True)
        return None
    
    with open(decisions_file, 'r') as f:
        decisions = json.load(f)
    
    if model_type not in decisions or decisions[model_type] != "USE_TUNED":
        return None
    
    with open(master_file, 'r') as f:
        master_params = json.load(f)
    
    if model_type in master_params and rep in master_params[model_type]:
        return master_params[model_type][rep]
    
    return None

def calibrate_bayesian_uncertainty(model, cal_loader, device, num_samples=None):
    """
    Calibrate BNN uncertainty estimates using variance scaling.
    Returns optimal scaling factor T.
    """
    # From the shared spec, not a default in the signature (RERUN_PLAN.md 2.13).
    if num_samples is None:
        num_samples = NEURAL_DEFAULTS['training']['mc_passes']
    model.eval()
    all_means = []
    all_stds = []
    all_targets = []
    
    # Collect predictions on calibration set
    for X_batch, y_batch in cal_loader:
        X_batch = X_batch.to(device)
        preds = []
        with torch.no_grad():
            for _ in range(num_samples):
                preds.append(model(X_batch).cpu().numpy())
        
        preds = np.array(preds)
        all_means.append(preds.mean(axis=0).flatten())
        all_stds.append(preds.std(axis=0).flatten())
        all_targets.append(y_batch.numpy().flatten())
    
    y_mean = np.concatenate(all_means)
    y_std = np.concatenate(all_stds)
    y_true = np.concatenate(all_targets)
    
    # Find optimal temperature T by minimizing negative log-likelihood
    def nll(T):
        scaled_std = y_std * T
        # Avoid log(0)
        scaled_std = np.maximum(scaled_std, 1e-6)
        nll = 0.5 * np.log(2 * np.pi * scaled_std**2) + 0.5 * ((y_true - y_mean)**2 / scaled_std**2)
        return nll.mean()
    
    from scipy.optimize import minimize_scalar
    result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
    optimal_T = result.x
    
    print(f"Optimal temperature for variance scaling: {optimal_T:.4f}")
    return optimal_T

def compute_calibration_metrics(y_true, y_pred_mean, y_pred_std):
    """
    Compute comprehensive calibration metrics.
    """
    metrics = {}
    
    # 1. Negative Log-Likelihood (lower is better)
    y_pred_std = np.maximum(y_pred_std, 1e-6)  # Avoid log(0)
    nll = 0.5 * np.log(2 * np.pi * y_pred_std**2) + 0.5 * ((y_true - y_pred_mean)**2 / y_pred_std**2)
    metrics['nll'] = nll.mean()
    
    # 2. Check if standardized errors follow N(0,1)
    z_scores = (y_true - y_pred_mean) / y_pred_std
    metrics['z_mean'] = z_scores.mean()  # Should be ~0
    metrics['z_std'] = z_scores.std()    # Should be ~1
    
    # 3. Expected Calibration Error (ECE) - binned approach
    n_bins = 10
    percentiles = np.linspace(0, 100, n_bins + 1)
    
    bin_edges = []
    for p_low, p_high in zip(percentiles[:-1], percentiles[1:]):
        # Get prediction intervals for this percentile
        z_low = -np.percentile(np.abs(z_scores), p_high)
        z_high = np.percentile(np.abs(z_scores), p_high)
        
        # Expected vs actual coverage
        expected_coverage = (p_high - p_low) / 100
        in_interval = np.abs(z_scores) <= np.percentile(np.abs(z_scores), p_high)
        actual_coverage = in_interval.mean()
        
        bin_edges.append({
            'expected': expected_coverage,
            'actual': actual_coverage,
            'count': len(z_scores)
        })
    
    # ECE is weighted average of |expected - actual|
    ece = sum(abs(b['expected'] - b['actual']) * b['count'] for b in bin_edges) / len(z_scores)
    metrics['ece'] = ece
    
    # 4. Coverage at standard confidence levels
    for confidence in [0.68, 0.95, 0.99]:
        z_threshold = {0.68: 1.0, 0.95: 1.96, 0.99: 2.576}[confidence]
        within_interval = np.abs(z_scores) <= z_threshold
        metrics[f'coverage_{int(confidence*100)}'] = within_interval.mean()
    
    # 5. Sharpness (average uncertainty)
    metrics['sharpness'] = y_pred_std.mean()
    
    return metrics

def calibrate_quantile_predictions(y_cal_true, y_cal_pred_lower, y_cal_pred_upper):
    """
    Calibrate prediction intervals using isotonic regression.
    """
    # Fit isotonic regression to lower and upper bounds
    iso_lower = IsotonicRegression(out_of_bounds='clip')
    iso_upper = IsotonicRegression(out_of_bounds='clip')
    
    iso_lower.fit(y_cal_pred_lower, y_cal_true)
    iso_upper.fit(y_cal_pred_upper, y_cal_true)
    
    return iso_lower, iso_upper

class MetaWeightNet(nn.Module):
    """
    Small network that predicts sample weights based on loss values.
    Takes loss as input, outputs weight in [0,1].
    """
    def __init__(self, hidden_size=100):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, loss):
        """
        Args:
            loss: (batch_size, 1) - loss values for each sample
        Returns:
            weights: (batch_size, 1) - weights in [0,1]
        """
        x = torch.relu(self.fc1(loss))
        x = torch.sigmoid(self.fc2(x))  # Weight between 0 and 1
        return x

def train_meta_weight_net(
    x_train, y_train, x_test, y_test, x_val, y_val,
    args, s, rep, iteration, iteration_seed, file_no, y_test_original,
    trial=None
, train_noise=None):
    """
    Train DNN with Meta-Weight-Net for sample reweighting.
    
    Optional: Use uncertainty alongside loss for weighting.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if we should use uncertainty
    use_uncertainty = getattr(args, 'use_uncertainty_weighting', False)
    loss_name = args.loss if hasattr(args, 'loss') else 'mse'
    
    # If using uncertainty, need a loss that outputs it
    if use_uncertainty and loss_name not in ['heteroscedastic', 'evidential', 'evidential_cauchy', 
                                              'evidential_laplace', 'sample_adaptive_barron', 'stratified']:
        print(f"Warning: use_uncertainty=True but {loss_name} doesn't output uncertainty. Using heteroscedastic.")
        loss_name = 'heteroscedastic'
    
    # Hyperparameters
    if trial is not None:
        hidden_size1 = trial.suggest_categorical('hidden_size1', [64, 128, 256])
        hidden_size2 = trial.suggest_categorical('hidden_size2', [32, 64, 128])
        meta_lr = trial.suggest_float('meta_lr', 1e-5, 1e-3, log=True)
        meta_hidden = trial.suggest_int('meta_hidden', 50, 200)
    else:
        hidden_size1, hidden_size2 = 128, 64
        meta_lr = 1e-4
        meta_hidden = 100
    
    # Determine output size
    output_size_map = {
        'heteroscedastic': 2, 'evidential': 4, 'evidential_cauchy': 4,
        'evidential_laplace': 4, 'sample_adaptive_barron': 2, 'stratified': 2,
    }
    output_size = output_size_map.get(loss_name, 1)
    
    # Create main model
    model = DNNRegressionModel(
        input_size=x_train.shape[1],
        hidden_size1=hidden_size1,
        hidden_size2=hidden_size2
    ).to(device)
    if output_size > 1:
        model.fc3 = nn.Linear(hidden_size2, output_size)
    
    # Create meta-weight network
    # Input: loss (+ uncertainty if enabled)
    meta_input_size = 2 if use_uncertainty else 1
    meta_net = nn.Sequential(
        nn.Linear(meta_input_size, meta_hidden),
        nn.ReLU(),
        nn.Linear(meta_hidden, 1),
        nn.Sigmoid()  # Weight in [0,1]
    ).to(device)
    
    # Optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=NEURAL_DEFAULTS['training']['lr'])
    meta_optimizer = torch.optim.Adam(meta_net.parameters(), lr=meta_lr)
    
    # Loss functions
    from loss_functions import get_loss_function
    loss_kwargs = {}
    if hasattr(args, 'loss_params') and args.loss_params:
        import json
        loss_kwargs = json.loads(args.loss_params)
    
    criterion = get_loss_function(loss_name, **loss_kwargs)
    criterion_for_weighting = nn.MSELoss(reduction='none')
    
    # Prepare data
    x_train_t = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    x_val_t = torch.tensor(x_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
    x_test_t = torch.tensor(x_test, dtype=torch.float32).to(device)
    
    train_dataset = TensorDataset(x_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=NEURAL_DEFAULTS['training']['batch_size'], shuffle=True)
    
    train_losses = []
    val_losses = []
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        meta_net.train()
        epoch_loss = 0
        
        for X_batch, y_batch in train_loader:
            # Forward pass
            y_pred = model(X_batch)
            
            # Extract mean prediction for loss calculation
            if output_size > 1:
                y_pred_mean = y_pred[:, 0:1]
            else:
                y_pred_mean = y_pred
            
            loss_per_sample = criterion_for_weighting(y_pred_mean, y_batch)
            
            # Get weights from meta-net
            if use_uncertainty:
                # Extract uncertainty
                if loss_name == 'heteroscedastic':
                    log_var = y_pred[:, 1:2]
                    uncertainty = torch.sqrt(torch.exp(log_var))
                
                elif loss_name in ['evidential', 'evidential_cauchy', 'evidential_laplace']:
                    v = F.softplus(y_pred[:, 1:2]) + 1.0
                    alpha = F.softplus(y_pred[:, 2:3]) + 1.0
                    beta = F.softplus(y_pred[:, 3:4])
                    epistemic = beta / torch.clamp(alpha - 1, min=1e-6)
                    aleatoric = beta / (v * torch.clamp(alpha - 1, min=1e-6))
                    uncertainty = torch.sqrt(epistemic + aleatoric)
                
                elif loss_name == 'sample_adaptive_barron':
                    alpha_raw = y_pred[:, 1:2]
                    alpha = torch.sigmoid(alpha_raw) * 3.9 + 0.1
                    uncertainty = 1.0 / (alpha + 1e-6)
                
                elif loss_name == 'stratified':
                    uncertainty_logit = y_pred[:, 1:2]
                    uncertainty = torch.sigmoid(uncertainty_logit)
                
                # Combine loss and uncertainty for meta-net
                meta_input = torch.cat([loss_per_sample.detach(), uncertainty.detach()], dim=1)
            else:
                meta_input = loss_per_sample.detach()
            
            weights = meta_net(meta_input)
            
            # Weighted loss
            weighted_loss = (weights * loss_per_sample).mean()
            
            # Update main model
            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()
            
            # Meta-update every 5 epochs
            if epoch % 5 == 0:
                model.eval()
                y_val_pred = model(x_val_t)
                if output_size > 1:
                    y_val_pred_mean = y_val_pred[:, 0:1]
                else:
                    y_val_pred_mean = y_val_pred
                val_loss_for_meta = criterion_for_weighting(y_val_pred_mean, y_val_t).mean()
                model.train()
                
                meta_optimizer.zero_grad()
                val_loss_for_meta.backward()
                meta_optimizer.step()
            
            epoch_loss += weighted_loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            y_val_pred = model(x_val_t)
            val_loss = criterion(y_val_pred, y_val_t)
        val_losses.append(val_loss.item())
        
        if epoch % 10 == 0:
            unc_str = "+unc" if use_uncertainty else ""
            print(f"Epoch {epoch}: Train={avg_train_loss:.4f}, Val={val_loss.item():.4f} {unc_str}")
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        pred_test = model(x_test_t).cpu().numpy()
        if output_size > 1:
            y_pred = pred_test[:, 0]
        else:
            y_pred = pred_test.flatten()
        
        # Get sample weights for analysis
        train_pred = model(x_train_t)
        if output_size > 1:
            train_pred_mean = train_pred[:, 0:1]
        else:
            train_pred_mean = train_pred
        
        train_losses_final = criterion_for_weighting(train_pred_mean, y_train_t)
        
        if use_uncertainty:
            if loss_name == 'heteroscedastic':
                log_var = train_pred[:, 1:2]
                train_unc = torch.sqrt(torch.exp(log_var))
            # ... other uncertainty extractions ...
            else:
                train_unc = torch.zeros_like(train_losses_final)
            
            meta_input = torch.cat([train_losses_final, train_unc], dim=1)
        else:
            meta_input = train_losses_final
        
        sample_weights = meta_net(meta_input).cpu().numpy().flatten()
    
    # Calculate metrics
    metrics = calculate_regression_metrics(y_test, y_pred, logging=True)
    
    # Save results
    model_name = f"meta_weight_net_{loss_name}" if loss_name != 'mse' else "meta_weight_net"
    if use_uncertainty:
        model_name += "_withunc"
    
    save_results(args.filepath, s, iteration, model_name, rep,
                args.sample_size, metrics, 'default', loss_name)
    
    # Save sample weights
    if args.uncertainty:
        import pandas as pd
        import os
        weights_dir = os.path.dirname(args.filepath.replace('.csv', '_sample_weights/'))
        os.makedirs(weights_dir, exist_ok=True)
        
        weights_df = pd.DataFrame({
            'sample_idx': np.arange(len(sample_weights)),
            'weight': sample_weights,
            'loss': train_losses_final.cpu().numpy().flatten(),
            'y_true': y_train,
            'y_pred': train_pred_mean.cpu().numpy().flatten()
        })
        
        weights_file = os.path.join(
            weights_dir,
            f"weights_{model_name}_{rep}_sigma{s}_iter{iteration}_file{file_no}.csv"
        )
        weights_df.to_csv(weights_file, index=False)
    
    if args.save_per_epoch_metrics:
        save_per_epoch_metrics(
            train_losses=train_losses,
            val_losses=val_losses,
            filepath=args.filepath,
            model_name=model_name,
            rep=rep,
            sigma_noise=s,
            iteration=iteration,
            file_no=file_no
        )
    
    return metrics[3]

def train_dividemix_dnn(
    x_train, y_train, x_test, y_test, x_val, y_val,
    args, s, rep, iteration, iteration_seed, file_no, y_test_original,
    trial=None
, train_noise=None):
    """
    DivideMix for regression with DNNs.
    Two networks co-teach each other, using GMM to separate clean/noisy samples.
    
    Optional: Use uncertainty to enhance clean/noisy separation.
    """
    from sklearn.mixture import GaussianMixture
    from loss_functions import get_loss_function
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if we should use uncertainty
    use_uncertainty = getattr(args, 'use_uncertainty_weighting', False)
    loss_name = args.loss if hasattr(args, 'loss') else 'mse'
    
    if use_uncertainty and loss_name not in ['heteroscedastic', 'evidential', 'evidential_cauchy', 
                                              'evidential_laplace', 'sample_adaptive_barron', 'stratified']:
        print(f"Warning: use_uncertainty=True but {loss_name} doesn't output uncertainty. Using heteroscedastic.")
        loss_name = 'heteroscedastic'
    
    # Hyperparameters
    if trial is not None:
        hidden_size1 = trial.suggest_categorical('hidden_size1', [64, 128, 256])
        hidden_size2 = trial.suggest_categorical('hidden_size2', [32, 64, 128])
        warmup_epochs = trial.suggest_int('warmup_epochs', 10, 30)
        forget_rate = trial.suggest_float('forget_rate', 0.1, 0.4)
    else:
        hidden_size1, hidden_size2 = 128, 64
        warmup_epochs = 20
        forget_rate = 0.25
    
    # Get loss function
    loss_kwargs = {}
    if hasattr(args, 'loss_params') and args.loss_params:
        import json
        loss_kwargs = json.loads(args.loss_params)
    
    # Determine output size based on loss
    output_size_map = {
        'heteroscedastic': 2,
        'evidential': 4,
        'evidential_cauchy': 4,
        'evidential_laplace': 4,
        'sample_adaptive_barron': 2,
        'stratified': 2,
    }
    output_size = output_size_map.get(loss_name, 1)
    
    # Create two networks
    model_f = DNNRegressionModel(
        input_size=x_train.shape[1],
        hidden_size1=hidden_size1,
        hidden_size2=hidden_size2
    ).to(device)
    if output_size > 1:
        model_f.fc3 = nn.Linear(hidden_size2, output_size)
    
    model_g = DNNRegressionModel(
        input_size=x_train.shape[1],
        hidden_size1=hidden_size1,
        hidden_size2=hidden_size2
    ).to(device)
    if output_size > 1:
        model_g.fc3 = nn.Linear(hidden_size2, output_size)
    
    optimizer_f = torch.optim.Adam(model_f.parameters(), lr=0.001)
    optimizer_g = torch.optim.Adam(model_g.parameters(), lr=0.001)
    
    # Get loss function
    criterion = get_loss_function(loss_name, **loss_kwargs)
    
    # For sample selection, we need per-sample losses (MSE with reduction='none')
    selection_criterion = nn.MSELoss(reduction='none')
    
    # Prepare data
    x_train_t = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    x_val_t = torch.tensor(x_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
    x_test_t = torch.tensor(x_test, dtype=torch.float32).to(device)
    
    # Phase 1: Warmup - train both networks normally
    print(f"Warmup phase with {loss_name} loss...")
    for epoch in range(warmup_epochs):
        model_f.train()
        model_g.train()
        
        # Train F
        optimizer_f.zero_grad()
        pred_f = model_f(x_train_t)
        loss_f = criterion(pred_f, y_train_t)
        loss_f.backward()
        optimizer_f.step()
        
        # Train G
        optimizer_g.zero_grad()
        pred_g = model_g(x_train_t)
        loss_g = criterion(pred_g, y_train_t)
        loss_g.backward()
        optimizer_g.step()
        
        if epoch % 5 == 0:
            print(f"Warmup epoch {epoch}: Loss_F={loss_f.item():.4f}, Loss_G={loss_g.item():.4f}")
    
    # Phase 2: Co-teaching with sample selection
    print("Co-teaching phase...")
    train_losses = []
    val_losses = []
    
    for epoch in range(args.epochs - warmup_epochs):
        model_f.eval()
        model_g.eval()
        
        # Get losses from both networks (use MSE for sample selection)
        with torch.no_grad():
            pred_f = model_f(x_train_t)
            pred_g = model_g(x_train_t)
            
            # Extract mean predictions for sample selection
            if output_size > 1:
                pred_f_mean = pred_f[:, 0:1]
                pred_g_mean = pred_g[:, 0:1]
            else:
                pred_f_mean = pred_f
                pred_g_mean = pred_g
            
            loss_f = selection_criterion(pred_f_mean, y_train_t).squeeze().cpu().numpy()
            loss_g = selection_criterion(pred_g_mean, y_train_t).squeeze().cpu().numpy()
            
            # Extract uncertainty if enabled
            if use_uncertainty:
                if loss_name == 'heteroscedastic':
                    log_var_f = pred_f[:, 1]
                    log_var_g = pred_g[:, 1]
                    unc_f = torch.sqrt(torch.exp(log_var_f)).cpu().numpy()
                    unc_g = torch.sqrt(torch.exp(log_var_g)).cpu().numpy()
                
                elif loss_name in ['evidential', 'evidential_cauchy', 'evidential_laplace']:
                    # Extract from F
                    v_f = torch.clamp(F.softplus(pred_f[:, 1]) + 1.0, min=1.0).cpu().numpy()
                    alpha_f = torch.clamp(F.softplus(pred_f[:, 2]) + 1.0, min=1.0).cpu().numpy()
                    beta_f = torch.clamp(F.softplus(pred_f[:, 3]), min=1e-6).cpu().numpy()
                    epistemic_f = beta_f / np.maximum(alpha_f - 1, 1e-6)
                    aleatoric_f = beta_f / (v_f * np.maximum(alpha_f - 1, 1e-6))
                    unc_f = np.sqrt(epistemic_f + aleatoric_f)
                    
                    # Extract from G
                    v_g = torch.clamp(F.softplus(pred_g[:, 1]) + 1.0, min=1.0).cpu().numpy()
                    alpha_g = torch.clamp(F.softplus(pred_g[:, 2]) + 1.0, min=1.0).cpu().numpy()
                    beta_g = torch.clamp(F.softplus(pred_g[:, 3]), min=1e-6).cpu().numpy()
                    epistemic_g = beta_g / np.maximum(alpha_g - 1, 1e-6)
                    aleatoric_g = beta_g / (v_g * np.maximum(alpha_g - 1, 1e-6))
                    unc_g = np.sqrt(epistemic_g + aleatoric_g)
                
                elif loss_name == 'sample_adaptive_barron':
                    alpha_f = torch.sigmoid(pred_f[:, 1]) * 3.9 + 0.1
                    alpha_g = torch.sigmoid(pred_g[:, 1]) * 3.9 + 0.1
                    unc_f = (1.0 / (alpha_f + 1e-6)).cpu().numpy()
                    unc_g = (1.0 / (alpha_g + 1e-6)).cpu().numpy()
                
                elif loss_name == 'stratified':
                    unc_f = torch.sigmoid(pred_f[:, 1]).cpu().numpy()
                    unc_g = torch.sigmoid(pred_g[:, 1]).cpu().numpy()
                
                # Combine loss with uncertainty for better separation
                # High uncertainty samples are more likely noisy
                avg_uncertainty = (unc_f + unc_g) / 2
                # Normalize uncertainty
                unc_norm = (avg_uncertainty - avg_uncertainty.min()) / (avg_uncertainty.max() - avg_uncertainty.min() + 1e-8)
                
                # Weight loss by uncertainty (higher uncertainty = treat as noisier)
                loss_f = loss_f * (1 + unc_norm)
                loss_g = loss_g * (1 + unc_norm)
        
        # Fit GMM to losses to separate clean/noisy
        avg_loss = (loss_f + loss_g) / 2
        gmm = GaussianMixture(n_components=2, random_state=iteration_seed)
        gmm.fit(avg_loss.reshape(-1, 1))
        
        # Predict clean vs noisy (component with lower mean is "clean")
        prob_clean = gmm.predict_proba(avg_loss.reshape(-1, 1))
        clean_component = np.argmin(gmm.means_)
        clean_prob = prob_clean[:, clean_component]
        
        # Select top (1 - forget_rate) samples as clean
        num_clean = int(len(x_train) * (1 - forget_rate))
        clean_indices = np.argsort(clean_prob)[-num_clean:]
        
        # Co-teaching: F selects samples for G, G selects for F
        split = len(clean_indices) // 2
        indices_for_f = clean_indices[:split]
        indices_for_g = clean_indices[split:]
        
        # Train F on samples selected by G (using the actual loss function)
        model_f.train()
        optimizer_f.zero_grad()
        pred_f = model_f(x_train_t[indices_for_g])
        loss_f = criterion(pred_f, y_train_t[indices_for_g])
        loss_f.backward()
        optimizer_f.step()
        
        # Train G on samples selected by F
        model_g.train()
        optimizer_g.zero_grad()
        pred_g = model_g(x_train_t[indices_for_f])
        loss_g = criterion(pred_g, y_train_t[indices_for_f])
        loss_g.backward()
        optimizer_g.step()
        
        avg_train_loss = (loss_f.item() + loss_g.item()) / 2
        train_losses.append(avg_train_loss)
        
        # Validation
        model_f.eval()
        model_g.eval()
        with torch.no_grad():
            pred_f_val = model_f(x_val_t)
            pred_g_val = model_g(x_val_t)
            val_loss_f = criterion(pred_f_val, y_val_t)
            val_loss_g = criterion(pred_g_val, y_val_t)
            val_loss = (val_loss_f + val_loss_g) / 2
        val_losses.append(val_loss.item())
        
        if epoch % 10 == 0:
            unc_str = "+unc" if use_uncertainty else ""
            print(f"Epoch {epoch}: Train={avg_train_loss:.4f}, Val={val_loss.item():.4f}, "
                  f"Clean={num_clean}/{len(x_train)} {unc_str}")
    
    # Test: average predictions from both networks
    model_f.eval()
    model_g.eval()
    with torch.no_grad():
        pred_f_test = model_f(x_test_t).cpu().numpy()
        pred_g_test = model_g(x_test_t).cpu().numpy()
        
        # Extract mean predictions
        if output_size > 1:
            pred_f_test = pred_f_test[:, 0]
            pred_g_test = pred_g_test[:, 0]
        else:
            pred_f_test = pred_f_test.flatten()
            pred_g_test = pred_g_test.flatten()
        
        y_pred = (pred_f_test + pred_g_test) / 2
    
    # Calculate metrics
    metrics = calculate_regression_metrics(y_test, y_pred, logging=True)
    
    # Save results
    model_name = f"dividemix_dnn_{loss_name}" if loss_name != 'mse' else "dividemix_dnn"
    if use_uncertainty:
        model_name += "_withunc"
    
    save_results(args.filepath, s, iteration, model_name, rep,
                args.sample_size, metrics, 'default', loss_name)
    
    # Save per-epoch metrics if requested
    if args.save_per_epoch_metrics:
        save_per_epoch_metrics(
            train_losses=train_losses,
            val_losses=val_losses,
            filepath=args.filepath,
            model_name=model_name,
            rep=rep,
            sigma_noise=s,
            iteration=iteration,
            file_no=file_no
        )
    
    return metrics[3]

def train_early_learning_regularization(
    x_train, y_train, x_test, y_test, x_val, y_val,
    args, s, rep, iteration, iteration_seed, file_no, y_test_original,
    trial=None
, train_noise=None):
    """
    Early Learning Regularization: weight samples by how consistently 
    they had low loss in early training epochs.
    
    Optional: Combine early learning with uncertainty estimates.
    """
    from loss_functions import get_loss_function
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if we should use uncertainty
    use_uncertainty = getattr(args, 'use_uncertainty_weighting', False)
    loss_name = args.loss if hasattr(args, 'loss') else 'mse'
    
    if use_uncertainty and loss_name not in ['heteroscedastic', 'evidential', 'evidential_cauchy', 
                                              'evidential_laplace', 'sample_adaptive_barron', 'stratified']:
        print(f"Warning: use_uncertainty=True but {loss_name} doesn't output uncertainty. Using heteroscedastic.")
        loss_name = 'heteroscedastic'
    
    # Hyperparameters
    if trial is not None:
        hidden_size1 = trial.suggest_categorical('hidden_size1', [64, 128, 256])
        hidden_size2 = trial.suggest_categorical('hidden_size2', [32, 64, 128])
        early_epochs = trial.suggest_int('early_epochs', 5, 20)
        weight_decay_rate = trial.suggest_float('weight_decay_rate', 0.9, 0.99)
    else:
        hidden_size1, hidden_size2 = 128, 64
        early_epochs = 10
        weight_decay_rate = 0.95
    
    # Get loss function
    loss_kwargs = {}
    if hasattr(args, 'loss_params') and args.loss_params:
        import json
        loss_kwargs = json.loads(args.loss_params)
    
    # Determine output size
    output_size_map = {
        'heteroscedastic': 2, 'evidential': 4, 'evidential_cauchy': 4,
        'evidential_laplace': 4, 'sample_adaptive_barron': 2, 'stratified': 2,
    }
    output_size = output_size_map.get(loss_name, 1)
    
    # Create model
    model = DNNRegressionModel(
        input_size=x_train.shape[1],
        hidden_size1=hidden_size1,
        hidden_size2=hidden_size2
    ).to(device)
    if output_size > 1:
        model.fc3 = nn.Linear(hidden_size2, output_size)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=NEURAL_DEFAULTS['training']['lr'])
    
    # Loss functions
    criterion = get_loss_function(loss_name, **loss_kwargs)
    tracking_criterion = nn.MSELoss(reduction='none')  # For tracking per-sample loss
    
    # Prepare data
    x_train_t = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    x_val_t = torch.tensor(x_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
    x_test_t = torch.tensor(x_test, dtype=torch.float32).to(device)
    
    # Track early learning: store per-sample losses for first few epochs
    early_losses = []
    early_uncertainties = [] if use_uncertainty else None
    
    # Phase 1: Track early learning (no weighting yet)
    print(f"Phase 1: Tracking early learning for {early_epochs} epochs...")
    for epoch in range(early_epochs):
        model.train()
        optimizer.zero_grad()
        
        pred = model(x_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()
        
        # Track per-sample losses and uncertainties
        model.eval()
        with torch.no_grad():
            pred_tracking = model(x_train_t)
            
            if output_size > 1:
                pred_mean = pred_tracking[:, 0:1]
            else:
                pred_mean = pred_tracking
            
            sample_losses = tracking_criterion(pred_mean, y_train_t).squeeze().cpu().numpy()
            early_losses.append(sample_losses)
            
            # Track uncertainty if enabled
            if use_uncertainty:
                if loss_name == 'heteroscedastic':
                    log_var = pred_tracking[:, 1]
                    unc = torch.sqrt(torch.exp(log_var)).cpu().numpy()
                
                elif loss_name in ['evidential', 'evidential_cauchy', 'evidential_laplace']:
                    v = torch.clamp(F.softplus(pred_tracking[:, 1]) + 1.0, min=1.0).cpu().numpy()
                    alpha = torch.clamp(F.softplus(pred_tracking[:, 2]) + 1.0, min=1.0).cpu().numpy()
                    beta = torch.clamp(F.softplus(pred_tracking[:, 3]), min=1e-6).cpu().numpy()
                    epistemic = beta / np.maximum(alpha - 1, 1e-6)
                    aleatoric = beta / (v * np.maximum(alpha - 1, 1e-6))
                    unc = np.sqrt(epistemic + aleatoric)
                
                elif loss_name == 'sample_adaptive_barron':
                    alpha = torch.sigmoid(pred_tracking[:, 1]) * 3.9 + 0.1
                    unc = (1.0 / (alpha + 1e-6)).cpu().numpy()
                
                elif loss_name == 'stratified':
                    unc = torch.sigmoid(pred_tracking[:, 1]).cpu().numpy()
                
                early_uncertainties.append(unc)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss={loss.item():.4f}")
    
    # Compute sample weights based on early learning
    early_losses = np.array(early_losses)  # shape: (early_epochs, n_samples)
    
    # Average loss in early epochs
    avg_early_loss = early_losses.mean(axis=0)
    
    # Convert to weights: lower early loss = higher weight
    weights = np.exp(-avg_early_loss / avg_early_loss.mean())
    
    # Combine with uncertainty if enabled
    if use_uncertainty:
        early_uncertainties = np.array(early_uncertainties)
        avg_early_unc = early_uncertainties.mean(axis=0)
        
        # Lower uncertainty = higher weight
        unc_weights = np.exp(-avg_early_unc / avg_early_unc.mean())
        
        # Combine: samples that are both low-loss AND low-uncertainty in early epochs get highest weight
        weights = weights * unc_weights
        print(f"Combined early loss + uncertainty weights")
    
    # Normalize
    weights = weights / weights.sum() * len(weights)
    weights = torch.tensor(weights, dtype=torch.float32).to(device)
    
    print(f"Sample weights - Min: {weights.min():.3f}, Max: {weights.max():.3f}, Mean: {weights.mean():.3f}")
    
    # Phase 2: Train with weighted loss
    print(f"Phase 2: Training with early learning weights...")
    train_losses = []
    val_losses = []
    
    for epoch in range(args.epochs - early_epochs):
        model.train()
        optimizer.zero_grad()
        
        pred = model(x_train_t)
        
        # Compute per-sample loss
        if output_size > 1:
            pred_mean = pred[:, 0:1]
        else:
            pred_mean = pred
        
        sample_loss = tracking_criterion(pred_mean, y_train_t).squeeze()
        
        # Apply weights
        weighted_loss = (weights * sample_loss).mean()
        
        weighted_loss.backward()
        optimizer.step()
        
        train_losses.append(weighted_loss.item())
        
        # Decay weights over time (trust early learning less as training progresses)
        weights = weights * weight_decay_rate
        
        # Validation
        model.eval()
        with torch.no_grad():
            pred_val = model(x_val_t)
            val_loss = criterion(pred_val, y_val_t)
        val_losses.append(val_loss.item())
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train={weighted_loss.item():.4f}, Val={val_loss.item():.4f}")
    
    # Test
    model.eval()
    with torch.no_grad():
        pred_test = model(x_test_t).cpu().numpy()
        if output_size > 1:
            y_pred = pred_test[:, 0]
        else:
            y_pred = pred_test.flatten()
    
    # Calculate metrics
    metrics = calculate_regression_metrics(y_test, y_pred, logging=True)
    
    # Save results
    model_name = f"early_learning_{loss_name}" if loss_name != 'mse' else "early_learning"
    if use_uncertainty:
        model_name += "_withunc"
    
    save_results(args.filepath, s, iteration, model_name, rep,
                args.sample_size, metrics, 'default', loss_name)
    
    # Save per-epoch metrics
    if args.save_per_epoch_metrics:
        save_per_epoch_metrics(
            train_losses=train_losses,
            val_losses=val_losses,
            filepath=args.filepath,
            model_name=model_name,
            rep=rep,
            sigma_noise=s,
            iteration=iteration,
            file_no=file_no
        )
    
    return metrics[3]

def train_multistage_cleaning(
    x_train, y_train, x_test, y_test, x_val, y_val,
    args, s, rep, iteration, iteration_seed, file_no, y_test_original,
    trial=None
, train_noise=None):
    """
    Multi-stage cleaning: iteratively remove high-loss + high-uncertainty samples.
    
    Stage 1: Train with all data
    Stage 2: Remove worst 10% (high loss + high uncertainty)
    Stage 3: Retrain on cleaned data
    Repeat n_stages times
    """
    from loss_functions import get_loss_function
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters
    if trial is not None:
        hidden_size1 = trial.suggest_categorical('hidden_size1', [64, 128, 256])
        hidden_size2 = trial.suggest_categorical('hidden_size2', [32, 64, 128])
        n_stages = trial.suggest_int('n_stages', 2, 5)
        removal_rate = trial.suggest_float('removal_rate', 0.05, 0.2)
    else:
        hidden_size1, hidden_size2 = 128, 64
        n_stages = 3
        removal_rate = 0.1  # Remove 10% worst samples per stage
    
    # Get loss function (must support uncertainty)
    loss_name = args.loss if hasattr(args, 'loss') else 'heteroscedastic'
    if loss_name not in ['heteroscedastic', 'evidential', 'evidential_cauchy', 
                         'evidential_laplace', 'sample_adaptive_barron', 'stratified']:
        print(f"Warning: {loss_name} doesn't output uncertainty. Using heteroscedastic.")
        loss_name = 'heteroscedastic'
    
    loss_kwargs = {}
    if hasattr(args, 'loss_params') and args.loss_params:
        import json
        loss_kwargs = json.loads(args.loss_params)
    
    # Determine output size
    output_size_map = {
        'heteroscedastic': 2, 'evidential': 4, 'evidential_cauchy': 4,
        'evidential_laplace': 4, 'sample_adaptive_barron': 2, 'stratified': 2,
    }
    output_size = output_size_map[loss_name]
    
    # Prepare data
    x_train_t = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    x_val_t = torch.tensor(x_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
    x_test_t = torch.tensor(x_test, dtype=torch.float32).to(device)
    
    # Keep track of which samples to keep
    keep_mask = np.ones(len(x_train), dtype=bool)
    
    tracking_criterion = nn.MSELoss(reduction='none')
    
    # Multi-stage cleaning
    for stage in range(n_stages):
        print(f"\n=== Stage {stage+1}/{n_stages} ===")
        print(f"Training on {keep_mask.sum()}/{len(x_train)} samples")
        
        # Get current clean data
        x_current = x_train_t[keep_mask]
        y_current = y_train_t[keep_mask]
        
        # Create model
        model = DNNRegressionModel(
            input_size=x_train.shape[1],
            hidden_size1=hidden_size1,
            hidden_size2=hidden_size2
        ).to(device)
        model.fc3 = nn.Linear(hidden_size2, output_size)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=NEURAL_DEFAULTS['training']['lr'])
        criterion = get_loss_function(loss_name, **loss_kwargs)
        
        # Train on current clean set
        epochs_per_stage = args.epochs // n_stages
        for epoch in range(epochs_per_stage):
            model.train()
            optimizer.zero_grad()
            
            pred = model(x_current)
            loss = criterion(pred, y_current)
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                print(f"  Epoch {epoch}: Loss={loss.item():.4f}")
        
        # Evaluate on ALL training data to identify noisy samples
        if stage < n_stages - 1:  # Don't clean on last stage
            model.eval()
            with torch.no_grad():
                pred_all = model(x_train_t).cpu().numpy()
                
                # Extract mean and uncertainty
                if loss_name == 'heteroscedastic':
                    pred_mean = pred_all[:, 0]
                    log_var = pred_all[:, 1]
                    uncertainty = np.sqrt(np.exp(log_var))
                
                elif loss_name in ['evidential', 'evidential_cauchy', 'evidential_laplace']:
                    gamma = pred_all[:, 0]
                    v = np.maximum(pred_all[:, 1], 1.0)
                    alpha = np.maximum(pred_all[:, 2], 1.0)
                    beta = np.maximum(pred_all[:, 3], 1e-6)
                    pred_mean = gamma
                    epistemic = beta / np.maximum(alpha - 1, 1e-6)
                    aleatoric = beta / (v * np.maximum(alpha - 1, 1e-6))
                    uncertainty = np.sqrt(epistemic + aleatoric)
                
                elif loss_name == 'sample_adaptive_barron':
                    pred_mean = pred_all[:, 0]
                    alpha = pred_all[:, 1]
                    alpha = 1 / (1 + np.exp(-alpha)) * 3.9 + 0.1  # Sigmoid to [0.1, 4.0]
                    uncertainty = 1.0 / (alpha + 1e-6)
                
                elif loss_name == 'stratified':
                    pred_mean = pred_all[:, 0]
                    uncertainty_logit = pred_all[:, 1]
                    uncertainty = 1 / (1 + np.exp(-uncertainty_logit))
                
                # Compute loss on kept samples only
                pred_mean_t = torch.tensor(pred_mean[keep_mask], dtype=torch.float32).to(device)
                y_current_np = y_current.cpu().numpy().flatten()
                sample_loss = np.abs(pred_mean[keep_mask] - y_current_np)
            
            # Score: high loss + high uncertainty = likely noisy
            # Only score samples we're currently keeping
            current_indices = np.where(keep_mask)[0]
            uncertainty_current = uncertainty[keep_mask]
            
            # Normalize
            loss_norm = (sample_loss - sample_loss.min()) / (sample_loss.max() - sample_loss.min() + 1e-8)
            unc_norm = (uncertainty_current - uncertainty_current.min()) / (uncertainty_current.max() - uncertainty_current.min() + 1e-8)
            
            # Combined score
            noise_score = loss_norm + unc_norm
            
            # Remove worst samples
            n_remove = int(len(current_indices) * removal_rate)
            worst_idx_in_current = np.argsort(noise_score)[-n_remove:]
            worst_idx_global = current_indices[worst_idx_in_current]
            
            keep_mask[worst_idx_global] = False
            
            print(f"  Removed {n_remove} samples. Remaining: {keep_mask.sum()}/{len(x_train)}")
    
    # Final test
    model.eval()
    with torch.no_grad():
        pred_test = model(x_test_t).cpu().numpy()
        y_pred = pred_test[:, 0]  # Extract mean
    
    # Calculate metrics
    metrics = calculate_regression_metrics(y_test, y_pred, logging=True)
    
    # Save results
    model_name = f"multistage_{loss_name}"
    save_results(args.filepath, s, iteration, model_name, rep,
                args.sample_size, metrics, 'default', loss_name)
    
    # Save cleaning info
    if args.uncertainty:
        import pandas as pd
        import os
        cleaning_dir = os.path.dirname(args.filepath.replace('.csv', '_cleaning/'))
        os.makedirs(cleaning_dir, exist_ok=True)
        
        cleaning_df = pd.DataFrame({
            'sample_idx': np.arange(len(keep_mask)),
            'kept': keep_mask.astype(int),
            'y_true': y_train
        })
        
        cleaning_file = os.path.join(
            cleaning_dir,
            f"cleaning_{model_name}_{rep}_sigma{s}_iter{iteration}_file{file_no}.csv"
        )
        cleaning_df.to_csv(cleaning_file, index=False)
        print(f"Saved cleaning info to {cleaning_file}")
    
    return metrics[3]

def train_uncertainty_curriculum(
    x_train, y_train, x_test, y_test, x_val, y_val,
    args, s, rep, iteration, iteration_seed, file_no, y_test_original,
    trial=None
, train_noise=None):
    """
    Uncertainty Curriculum: Start training on certain samples,
    gradually add uncertain ones.
    
    Phase 1: Train on low-uncertainty samples (top 50%)
    Phase 2: Gradually add medium uncertainty samples
    Phase 3: Train on all samples
    """
    from loss_functions import get_loss_function
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Must use a loss that outputs uncertainty
    loss_name = args.loss if hasattr(args, 'loss') else 'heteroscedastic'
    if loss_name not in ['heteroscedastic', 'evidential', 'evidential_cauchy', 
                         'evidential_laplace', 'sample_adaptive_barron', 'stratified']:
        print(f"Warning: {loss_name} doesn't output uncertainty. Using heteroscedastic.")
        loss_name = 'heteroscedastic'
    
    # Hyperparameters
    if trial is not None:
        hidden_size1 = trial.suggest_categorical('hidden_size1', [64, 128, 256])
        hidden_size2 = trial.suggest_categorical('hidden_size2', [32, 64, 128])
        warmup_epochs = trial.suggest_int('warmup_epochs', 10, 30)
        n_stages = trial.suggest_int('n_stages', 3, 5)
    else:
        hidden_size1, hidden_size2 = 128, 64
        warmup_epochs = 20
        n_stages = 4
    
    # Get loss function
    loss_kwargs = {}
    if hasattr(args, 'loss_params') and args.loss_params:
        import json
        loss_kwargs = json.loads(args.loss_params)
    
    # Determine output size
    output_size_map = {
        'heteroscedastic': 2, 'evidential': 4, 'evidential_cauchy': 4,
        'evidential_laplace': 4, 'sample_adaptive_barron': 2, 'stratified': 2,
    }
    output_size = output_size_map[loss_name]
    
    # Create model
    model = DNNRegressionModel(
        input_size=x_train.shape[1],
        hidden_size1=hidden_size1,
        hidden_size2=hidden_size2
    ).to(device)
    model.fc3 = nn.Linear(hidden_size2, output_size)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=NEURAL_DEFAULTS['training']['lr'])
    criterion = get_loss_function(loss_name, **loss_kwargs)
    
    # Prepare data
    x_train_t = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    x_val_t = torch.tensor(x_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
    x_test_t = torch.tensor(x_test, dtype=torch.float32).to(device)
    
    # Phase 0: Warmup to get initial uncertainty estimates
    print(f"Warmup phase to estimate uncertainties...")
    for epoch in range(warmup_epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(x_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f"Warmup epoch {epoch}: Loss={loss.item():.4f}")
    
    # Get uncertainty estimates for all training samples
    model.eval()
    with torch.no_grad():
        pred_all = model(x_train_t).cpu().numpy()
        
        # Extract uncertainty
        if loss_name == 'heteroscedastic':
            log_var = pred_all[:, 1]
            uncertainty = np.sqrt(np.exp(log_var))
        
        elif loss_name in ['evidential', 'evidential_cauchy', 'evidential_laplace']:
            v = np.maximum(pred_all[:, 1], 1.0)
            alpha = np.maximum(pred_all[:, 2], 1.0)
            beta = np.maximum(pred_all[:, 3], 1e-6)
            epistemic = beta / np.maximum(alpha - 1, 1e-6)
            aleatoric = beta / (v * np.maximum(alpha - 1, 1e-6))
            uncertainty = np.sqrt(epistemic + aleatoric)
        
        elif loss_name == 'sample_adaptive_barron':
            alpha = 1 / (1 + np.exp(-pred_all[:, 1])) * 3.9 + 0.1
            uncertainty = 1.0 / (alpha + 1e-6)
        
        elif loss_name == 'stratified':
            uncertainty = 1 / (1 + np.exp(-pred_all[:, 1]))
    
    # Sort samples by uncertainty (low to high)
    sorted_indices = np.argsort(uncertainty)
    
    print(f"Uncertainty range: {uncertainty.min():.4f} to {uncertainty.max():.4f}")
    
    # Curriculum: gradually add samples from low to high uncertainty
    train_losses = []
    val_losses = []
    
    epochs_per_stage = (args.epochs - warmup_epochs) // n_stages
    
    for stage in range(n_stages):
        # Determine how many samples to include
        fraction = (stage + 1) / n_stages
        n_samples = int(len(x_train) * fraction)
        
        # Include lowest-uncertainty samples up to this fraction
        current_indices = sorted_indices[:n_samples]
        
        print(f"\n=== Stage {stage+1}/{n_stages}: Training on {n_samples}/{len(x_train)} samples ===")
        
        # Get current training set
        x_current = x_train_t[current_indices]
        y_current = y_train_t[current_indices]
        
        # Train on current subset
        for epoch in range(epochs_per_stage):
            model.train()
            optimizer.zero_grad()
            
            pred = model(x_current)
            loss = criterion(pred, y_current)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Validation
            if epoch % 5 == 0:
                model.eval()
                with torch.no_grad():
                    pred_val = model(x_val_t)
                    val_loss = criterion(pred_val, y_val_t)
                val_losses.append(val_loss.item())
                
                print(f"  Epoch {epoch}: Train={loss.item():.4f}, Val={val_loss.item():.4f}")
    
    # Final test
    model.eval()
    with torch.no_grad():
        pred_test = model(x_test_t).cpu().numpy()
        y_pred = pred_test[:, 0]  # Extract mean
    
    # Calculate metrics
    metrics = calculate_regression_metrics(y_test, y_pred, logging=True)
    
    # Save results
    model_name = f"uncertainty_curriculum_{loss_name}"
    save_results(args.filepath, s, iteration, model_name, rep,
                args.sample_size, metrics, 'default', loss_name)
    
    # Save curriculum info
    if args.uncertainty:
        import pandas as pd
        import os
        curriculum_dir = os.path.dirname(args.filepath.replace('.csv', '_curriculum/'))
        os.makedirs(curriculum_dir, exist_ok=True)
        
        curriculum_df = pd.DataFrame({
            'sample_idx': np.arange(len(uncertainty)),
            'uncertainty': uncertainty,
            'order_added': np.argsort(sorted_indices),  # When was this sample added to training
            'y_true': y_train
        })
        
        curriculum_file = os.path.join(
            curriculum_dir,
            f"curriculum_{model_name}_{rep}_sigma{s}_iter{iteration}_file{file_no}.csv"
        )
        curriculum_df.to_csv(curriculum_file, index=False)
        print(f"Saved curriculum info to {curriculum_file}")
    
    # Save per-epoch metrics
    if args.save_per_epoch_metrics:
        save_per_epoch_metrics(
            train_losses=train_losses,
            val_losses=val_losses,
            filepath=args.filepath,
            model_name=model_name,
            rep=rep,
            sigma_noise=s,
            iteration=iteration,
            file_no=file_no
        )
    
    return metrics[3]

def train_confident_learning(
    x_train, y_train, x_test, y_test, x_val, y_val,
    args, s, rep, iteration, iteration_seed, file_no, y_test_original,
    trial=None
, train_noise=None):
    """
    Confident Learning (Northcupt et al. 2021) adapted for regression.
    
    Identifies likely mislabeled samples as those with:
    - High prediction error
    - Low model uncertainty (confident mistakes)
    
    Optional: Use distance metrics to identify outliers
    """
    from loss_functions import get_loss_function
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Must use uncertainty-producing loss
    loss_name = args.loss if hasattr(args, 'loss') else 'heteroscedastic'
    if loss_name not in ['heteroscedastic', 'evidential', 'evidential_cauchy', 
                         'evidential_laplace', 'sample_adaptive_barron', 'stratified']:
        print(f"Warning: {loss_name} doesn't output uncertainty. Using heteroscedastic.")
        loss_name = 'heteroscedastic'
    
    # Hyperparameters
    if trial is not None:
        hidden_size1 = trial.suggest_categorical('hidden_size1', [64, 128, 256])
        hidden_size2 = trial.suggest_categorical('hidden_size2', [32, 64, 128])
        warmup_epochs = trial.suggest_int('warmup_epochs', 20, 50)
        error_percentile = trial.suggest_float('error_percentile', 70, 95)
        unc_percentile = trial.suggest_float('unc_percentile', 10, 40)
        use_distance = trial.suggest_categorical('use_distance', [True, False]) if hasattr(args, 'use_distance') else args.use_distance
    else:
        hidden_size1, hidden_size2 = 128, 64
        warmup_epochs = 30
        error_percentile = 85  # Top 15% errors
        unc_percentile = 25    # Bottom 25% uncertainty (most confident)
        use_distance = getattr(args, 'use_distance', False)
    
    # Loss setup
    loss_kwargs = {}
    if hasattr(args, 'loss_params') and args.loss_params:
        import json
        loss_kwargs = json.loads(args.loss_params)
    
    output_size_map = {
        'heteroscedastic': 2, 'evidential': 4, 'evidential_cauchy': 4,
        'evidential_laplace': 4, 'sample_adaptive_barron': 2, 'stratified': 2,
    }
    output_size = output_size_map[loss_name]
    
    # Prepare data
    x_train_t = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    x_val_t = torch.tensor(x_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
    x_test_t = torch.tensor(x_test, dtype=torch.float32).to(device)
    
    # Phase 1: Train initial model to identify noisy samples
    print(f"Phase 1: Training initial model for {warmup_epochs} epochs...")
    
    model = DNNRegressionModel(
        input_size=x_train.shape[1],
        hidden_size1=hidden_size1,
        hidden_size2=hidden_size2
    ).to(device)
    model.fc3 = nn.Linear(hidden_size2, output_size)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=NEURAL_DEFAULTS['training']['lr'])
    criterion = get_loss_function(loss_name, **loss_kwargs)
    
    # Train initial model
    for epoch in range(warmup_epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(x_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: Loss={loss.item():.4f}")
    
    # Phase 2: Identify likely mislabeled samples
    print("Phase 2: Identifying likely mislabeled samples...")
    
    model.eval()
    with torch.no_grad():
        pred_all = model(x_train_t).cpu().numpy()
        
        # Extract mean and uncertainty
        if loss_name == 'heteroscedastic':
            pred_mean = pred_all[:, 0]
            log_var = pred_all[:, 1]
            uncertainty = np.sqrt(np.exp(log_var))
        
        elif loss_name in ['evidential', 'evidential_cauchy', 'evidential_laplace']:
            gamma = pred_all[:, 0]
            v = np.maximum(pred_all[:, 1], 1.0)
            alpha = np.maximum(pred_all[:, 2], 1.0)
            beta = np.maximum(pred_all[:, 3], 1e-6)
            pred_mean = gamma
            epistemic = beta / np.maximum(alpha - 1, 1e-6)
            aleatoric = beta / (v * np.maximum(alpha - 1, 1e-6))
            uncertainty = np.sqrt(epistemic + aleatoric)
        
        elif loss_name == 'sample_adaptive_barron':
            pred_mean = pred_all[:, 0]
            alpha = 1 / (1 + np.exp(-pred_all[:, 1])) * 3.9 + 0.1
            uncertainty = 1.0 / (alpha + 1e-6)
        
        elif loss_name == 'stratified':
            pred_mean = pred_all[:, 0]
            uncertainty = 1 / (1 + np.exp(-pred_all[:, 1]))
    
    # Calculate errors
    errors = np.abs(pred_mean - y_train)
    
    # Identify "confident errors":
    # - High error (above threshold)
    # - Low uncertainty (below threshold)
    error_threshold = np.percentile(errors, error_percentile)
    unc_threshold = np.percentile(uncertainty, unc_percentile)
    
    high_error = errors > error_threshold
    low_uncertainty = uncertainty < unc_threshold
    
    confident_errors = high_error & low_uncertainty
    
    print(f"  High error samples: {high_error.sum()}/{len(y_train)}")
    print(f"  Low uncertainty samples: {low_uncertainty.sum()}/{len(y_train)}")
    print(f"  Confident errors (likely mislabeled): {confident_errors.sum()}/{len(y_train)}")
    
    # Optional: Refine with distance metrics
    if use_distance:
        from distance_metrics import identify_outliers_by_distance
        
        distance_method = getattr(args, 'distance_metric', 'tanimoto')
        outlier_mask, avg_distances = identify_outliers_by_distance(
            x_train, rep, method=distance_method, threshold_percentile=80
        )
        
        # Samples that are BOTH confident errors AND distance outliers are most suspect
        refined_noisy = confident_errors & outlier_mask
        
        print(f"  Distance outliers: {outlier_mask.sum()}/{len(y_train)}")
        print(f"  Refined noisy (error + unc + distance): {refined_noisy.sum()}/{len(y_train)}")
        
        noisy_mask = refined_noisy
    else:
        noisy_mask = confident_errors
    
    # Keep clean samples
    clean_mask = ~noisy_mask
    
    print(f"  Final: Keeping {clean_mask.sum()}/{len(y_train)} samples")
    
    # Phase 3: Retrain on cleaned data
    print("Phase 3: Retraining on cleaned data...")
    
    x_clean = x_train_t[clean_mask]
    y_clean = y_train_t[clean_mask]
    
    # Create fresh model
    model2 = DNNRegressionModel(
        input_size=x_train.shape[1],
        hidden_size1=hidden_size1,
        hidden_size2=hidden_size2
    ).to(device)
    model2.fc3 = nn.Linear(hidden_size2, output_size)
    
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(args.epochs - warmup_epochs):
        model2.train()
        optimizer2.zero_grad()
        pred = model2(x_clean)
        loss = criterion(pred, y_clean)
        loss.backward()
        optimizer2.step()
        
        train_losses.append(loss.item())
        
        # Validation
        model2.eval()
        with torch.no_grad():
            pred_val = model2(x_val_t)
            val_loss = criterion(pred_val, y_val_t)
        val_losses.append(val_loss.item())
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: Train={loss.item():.4f}, Val={val_loss.item():.4f}")
    
    # Test
    model2.eval()
    with torch.no_grad():
        pred_test = model2(x_test_t).cpu().numpy()
        y_pred = pred_test[:, 0]  # Extract mean
    
    # Calculate metrics
    metrics = calculate_regression_metrics(y_test, y_pred, logging=True)
    
    # Save results
    model_name = f"confident_learning_{loss_name}"
    if use_distance:
        model_name += "_dist"
    
    save_results(args.filepath, s, iteration, model_name, rep,
                args.sample_size, metrics, 'default', loss_name)
    
    # Save cleaning info
    if args.uncertainty:
        import pandas as pd
        import os
        cleaning_dir = os.path.dirname(args.filepath.replace('.csv', '_cleaning/'))
        os.makedirs(cleaning_dir, exist_ok=True)
        
        cleaning_df = pd.DataFrame({
            'sample_idx': np.arange(len(noisy_mask)),
            'kept': (~noisy_mask).astype(int),
            'y_true': y_train,
            'pred_mean': pred_mean,
            'error': errors,
            'uncertainty': uncertainty,
            'confident_error': confident_errors.astype(int),
        })
        
        if use_distance:
            cleaning_df['outlier'] = outlier_mask.astype(int)
        
        cleaning_file = os.path.join(
            cleaning_dir,
            f"cleaning_{model_name}_{rep}_sigma{s}_iter{iteration}_file{file_no}.csv"
        )
        cleaning_df.to_csv(cleaning_file, index=False)
    
    if args.save_per_epoch_metrics:
        save_per_epoch_metrics(
            train_losses=train_losses,
            val_losses=val_losses,
            filepath=args.filepath,
            model_name=model_name,
            rep=rep,
            sigma_noise=s,
            iteration=iteration,
            file_no=file_no
        )
    
    return metrics[3]

def train_small_loss_trick(
    x_train, y_train, x_test, y_test, x_val, y_val,
    args, s, rep, iteration, iteration_seed, file_no, y_test_original,
    trial=None
, train_noise=None):
    """
    Small-Loss Trick (Han et al. 2018) with molecular distance filtering.
    
    Strategy:
    1. Train model for warmup period
    2. Select samples with smallest losses (most likely clean)
    3. Optional: Filter out isolated small-loss samples using distance
    4. Retrain on selected samples
    """
    from loss_functions import get_loss_function
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Loss setup
    loss_name = args.loss if hasattr(args, 'loss') else 'mse'
    
    # Hyperparameters
    if trial is not None:
        hidden_size1 = trial.suggest_categorical('hidden_size1', [64, 128, 256])
        hidden_size2 = trial.suggest_categorical('hidden_size2', [32, 64, 128])
        warmup_epochs = trial.suggest_int('warmup_epochs', 20, 50)
        keep_fraction = trial.suggest_float('keep_fraction', 0.5, 0.9)
        use_distance = trial.suggest_categorical('use_distance', [True, False]) if hasattr(args, 'use_distance') else args.use_distance
    else:
        hidden_size1, hidden_size2 = 128, 64
        warmup_epochs = 30
        keep_fraction = 0.7  # Keep 70% with smallest losses
        use_distance = getattr(args, 'use_distance', False)
    
    # Loss function setup
    loss_kwargs = {}
    if hasattr(args, 'loss_params') and args.loss_params:
        import json
        loss_kwargs = json.loads(args.loss_params)
    
    # Determine output size
    output_size_map = {
        'heteroscedastic': 2, 'evidential': 4, 'evidential_cauchy': 4,
        'evidential_laplace': 4, 'sample_adaptive_barron': 2, 'stratified': 2,
    }
    output_size = output_size_map.get(loss_name, 1)
    
    # Prepare data
    x_train_t = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    x_val_t = torch.tensor(x_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
    x_test_t = torch.tensor(x_test, dtype=torch.float32).to(device)
    
    # Phase 1: Warmup training
    print(f"Phase 1: Warmup training for {warmup_epochs} epochs...")
    
    model = DNNRegressionModel(
        input_size=x_train.shape[1],
        hidden_size1=hidden_size1,
        hidden_size2=hidden_size2
    ).to(device)
    if output_size > 1:
        model.fc3 = nn.Linear(hidden_size2, output_size)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=NEURAL_DEFAULTS['training']['lr'])
    criterion = get_loss_function(loss_name, **loss_kwargs)
    tracking_criterion = nn.MSELoss(reduction='none')  # For sample selection
    
    # Train
    for epoch in range(warmup_epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(x_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: Loss={loss.item():.4f}")
    
    # Phase 2: Select small-loss samples
    print("Phase 2: Selecting small-loss samples...")
    
    model.eval()
    with torch.no_grad():
        pred_all = model(x_train_t).cpu().numpy()
        
        # Extract mean predictions for loss calculation
        if output_size > 1:
            pred_mean = pred_all[:, 0]
        else:
            pred_mean = pred_all.flatten()
        
        # Calculate per-sample losses
        pred_mean_t = torch.tensor(pred_mean, dtype=torch.float32).view(-1, 1).to(device)
        sample_losses = tracking_criterion(pred_mean_t, y_train_t).squeeze().cpu().numpy()
    
    # Select samples with smallest losses
    n_keep = int(len(y_train) * keep_fraction)
    loss_sorted_indices = np.argsort(sample_losses)
    small_loss_indices = loss_sorted_indices[:n_keep]
    
    print(f"  Selected {n_keep}/{len(y_train)} samples with smallest losses")
    print(f"  Loss threshold: {sample_losses[small_loss_indices[-1]]:.4f}")
    
    # Optional: Refine with distance filtering
    if use_distance:
        from distance_metrics import compute_molecular_distances
        
        distance_method = getattr(args, 'distance_metric', 'tanimoto')
        
        print(f"  Applying distance filtering with {distance_method}...")
        
        # Compute distances among small-loss samples
        x_small_loss = x_train[small_loss_indices]
        
        if len(small_loss_indices) > 1:
            distance_matrix = compute_molecular_distances(
                x_train, rep, method=distance_method,
                subset_indices=small_loss_indices
            )
            
            # For each small-loss sample, compute average distance to other small-loss samples
            avg_distances = np.zeros(len(small_loss_indices))
            for i in range(len(small_loss_indices)):
                distances_i = np.concatenate([
                    distance_matrix[i, :i],
                    distance_matrix[i, i+1:]
                ])
                avg_distances[i] = distances_i.mean()
            
            # Keep samples that are close to other small-loss samples
            # (isolated small-loss samples might be lucky outliers)
            distance_threshold = np.percentile(avg_distances, 75)  # Keep 75% closest
            close_to_cluster = avg_distances < distance_threshold
            
            # Final selection: small loss + close to others
            refined_indices = small_loss_indices[close_to_cluster]
            
            print(f"  Distance-filtered: {len(refined_indices)}/{n_keep} samples")
            print(f"  Removed {n_keep - len(refined_indices)} isolated samples")
            
            keep_indices = refined_indices
        else:
            keep_indices = small_loss_indices
    else:
        keep_indices = small_loss_indices
    
    # Create clean dataset
    x_clean = x_train_t[keep_indices]
    y_clean = y_train_t[keep_indices]
    
    # Phase 3: Retrain on clean data
    print(f"Phase 3: Retraining on {len(keep_indices)} clean samples...")
    
    # Create fresh model
    model2 = DNNRegressionModel(
        input_size=x_train.shape[1],
        hidden_size1=hidden_size1,
        hidden_size2=hidden_size2
    ).to(device)
    if output_size > 1:
        model2.fc3 = nn.Linear(hidden_size2, output_size)
    
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(args.epochs - warmup_epochs):
        model2.train()
        optimizer2.zero_grad()
        pred = model2(x_clean)
        loss = criterion(pred, y_clean)
        loss.backward()
        optimizer2.step()
        
        train_losses.append(loss.item())
        
        # Validation
        model2.eval()
        with torch.no_grad():
            pred_val = model2(x_val_t)
            val_loss = criterion(pred_val, y_val_t)
        val_losses.append(val_loss.item())
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: Train={loss.item():.4f}, Val={val_loss.item():.4f}")
    
    # Test
    model2.eval()
    with torch.no_grad():
        pred_test = model2(x_test_t).cpu().numpy()
        if output_size > 1:
            y_pred = pred_test[:, 0]
        else:
            y_pred = pred_test.flatten()
    
    # Calculate metrics
    metrics = calculate_regression_metrics(y_test, y_pred, logging=True)
    
    # Save results
    model_name = f"small_loss_{loss_name}" if loss_name != 'mse' else "small_loss"
    if use_distance:
        model_name += "_dist"
    
    save_results(args.filepath, s, iteration, model_name, rep,
                args.sample_size, metrics, 'default', loss_name)
    
    # Save selection info
    if args.uncertainty:
        import pandas as pd
        import os
        selection_dir = os.path.dirname(args.filepath.replace('.csv', '_selection/'))
        os.makedirs(selection_dir, exist_ok=True)
        
        keep_mask = np.zeros(len(y_train), dtype=bool)
        keep_mask[keep_indices] = True
        
        selection_df = pd.DataFrame({
            'sample_idx': np.arange(len(y_train)),
            'kept': keep_mask.astype(int),
            'loss': sample_losses,
            'y_true': y_train,
            'y_pred': pred_mean,
        })
        
        selection_file = os.path.join(
            selection_dir,
            f"selection_{model_name}_{rep}_sigma{s}_iter{iteration}_file{file_no}.csv"
        )
        selection_df.to_csv(selection_file, index=False)
    
    if args.save_per_epoch_metrics:
        save_per_epoch_metrics(
            train_losses=train_losses,
            val_losses=val_losses,
            filepath=args.filepath,
            model_name=model_name,
            rep=rep,
            sigma_noise=s,
            iteration=iteration,
            file_no=file_no
        )
    
    return metrics[3]

def train_mentornet(
    x_train, y_train, x_test, y_test, x_val, y_val,
    args, s, rep, iteration, iteration_seed, file_no, y_test_original,
    trial=None
, train_noise=None):
    """
    MentorNet (Jiang et al. 2018) adapted for molecular regression.
    
    A mentor network learns to weight training samples for a student network.
    The mentor observes loss history and optionally uncertainty/distance features.
    """
    from loss_functions import get_loss_function
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check for distance and uncertainty features
    use_distance = getattr(args, 'use_distance', False)
    loss_name = args.loss if hasattr(args, 'loss') else 'mse'
    
    # If using advanced loss, mentor can use uncertainty
    use_uncertainty = loss_name in ['heteroscedastic', 'evidential', 'evidential_cauchy', 
                                      'evidential_laplace', 'sample_adaptive_barron', 'stratified']
    
    # Hyperparameters
    if trial is not None:
        hidden_size1 = trial.suggest_categorical('hidden_size1', [64, 128, 256])
        hidden_size2 = trial.suggest_categorical('hidden_size2', [32, 64, 128])
        mentor_hidden = trial.suggest_int('mentor_hidden', 50, 200)
        mentor_lr = trial.suggest_float('mentor_lr', 1e-5, 1e-3, log=True)
        history_length = trial.suggest_int('history_length', 3, 10)
    else:
        hidden_size1, hidden_size2 = 128, 64
        mentor_hidden = 100
        mentor_lr = 1e-4
        history_length = 5  # Track last 5 epochs of losses
    
    # Loss function setup
    loss_kwargs = {}
    if hasattr(args, 'loss_params') and args.loss_params:
        import json
        loss_kwargs = json.loads(args.loss_params)
    
    output_size_map = {
        'heteroscedastic': 2, 'evidential': 4, 'evidential_cauchy': 4,
        'evidential_laplace': 4, 'sample_adaptive_barron': 2, 'stratified': 2,
    }
    output_size = output_size_map.get(loss_name, 1)
    
    # Prepare data
    x_train_t = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    x_val_t = torch.tensor(x_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
    x_test_t = torch.tensor(x_test, dtype=torch.float32).to(device)
    
    # Compute distance features if needed
    distance_features = None
    if use_distance:
        from distance_metrics import compute_molecular_distances
        
        distance_method = getattr(args, 'distance_metric', 'tanimoto')
        print(f"Computing {distance_method} distances for mentor features...")
        
        distance_matrix = compute_molecular_distances(x_train, rep, method=distance_method)
        
        # Features: average distance to nearest k neighbors
        k = min(10, len(x_train) - 1)
        nearest_k_distances = np.partition(distance_matrix, k, axis=1)[:, 1:k+1]  # Exclude self
        distance_features = torch.tensor(
            nearest_k_distances.mean(axis=1), dtype=torch.float32
        ).view(-1, 1).to(device)
    
    # Create student model
    student = DNNRegressionModel(
        input_size=x_train.shape[1],
        hidden_size1=hidden_size1,
        hidden_size2=hidden_size2
    ).to(device)
    if output_size > 1:
        student.fc3 = nn.Linear(hidden_size2, output_size)
    
    student_optimizer = torch.optim.Adam(student.parameters(), lr=0.001)
    criterion = get_loss_function(loss_name, **loss_kwargs)
    tracking_criterion = nn.MSELoss(reduction='none')
    
    # Create mentor network
    # Input: current loss + loss history + optional (uncertainty + distance)
    mentor_input_size = 1 + history_length  # current loss + history
    if use_uncertainty:
        mentor_input_size += 1  # add uncertainty
    if use_distance:
        mentor_input_size += 1  # add distance feature
    
    mentor = nn.Sequential(
        nn.Linear(mentor_input_size, mentor_hidden),
        nn.ReLU(),
        nn.Linear(mentor_hidden, mentor_hidden),
        nn.ReLU(),
        nn.Linear(mentor_hidden, 1),
        nn.Sigmoid()  # Weight in [0, 1]
    ).to(device)
    
    mentor_optimizer = torch.optim.Adam(mentor.parameters(), lr=mentor_lr)
    
    # Track loss history for each sample
    loss_history = torch.zeros(len(y_train), history_length).to(device)
    
    train_losses = []
    val_losses = []
    
    print("Training with MentorNet...")
    
    # Training loop
    for epoch in range(args.epochs):
        student.train()
        mentor.train()
        
        # Forward pass through student
        student_optimizer.zero_grad()
        pred = student(x_train_t)
        
        # Extract mean for loss calculation
        if output_size > 1:
            pred_mean = pred[:, 0:1]
        else:
            pred_mean = pred
        
        # Calculate per-sample losses
        sample_losses = tracking_criterion(pred_mean, y_train_t).squeeze()
        
        # Update loss history (shift and add current)
        loss_history = torch.roll(loss_history, -1, dims=1)
        loss_history[:, -1] = sample_losses.detach()
        
        # Prepare mentor input
        mentor_input = [
            sample_losses.detach().view(-1, 1),
            loss_history
        ]
        
        # Add uncertainty if available
        if use_uncertainty:
            if loss_name == 'heteroscedastic':
                log_var = pred[:, 1].detach()
                uncertainty = torch.sqrt(torch.exp(log_var))
            elif loss_name in ['evidential', 'evidential_cauchy', 'evidential_laplace']:
                v = torch.clamp(F.softplus(pred[:, 1]) + 1.0, min=1.0)
                alpha = torch.clamp(F.softplus(pred[:, 2]) + 1.0, min=1.0)
                beta = torch.clamp(F.softplus(pred[:, 3]), min=1e-6)
                epistemic = beta / torch.clamp(alpha - 1, min=1e-6)
                aleatoric = beta / (v * torch.clamp(alpha - 1, min=1e-6))
                uncertainty = torch.sqrt(epistemic + aleatoric)
            elif loss_name == 'sample_adaptive_barron':
                alpha = torch.sigmoid(pred[:, 1]) * 3.9 + 0.1
                uncertainty = 1.0 / (alpha + 1e-6)
            elif loss_name == 'stratified':
                uncertainty = torch.sigmoid(pred[:, 1])
            
            mentor_input.append(uncertainty.detach().view(-1, 1))
        
        # Add distance features
        if use_distance:
            mentor_input.append(distance_features)
        
        # Concatenate mentor inputs
        mentor_input = torch.cat(mentor_input, dim=1)
        
        # Get weights from mentor
        sample_weights = mentor(mentor_input).squeeze()
        
        # Weighted loss for student
        weighted_loss = (sample_weights * sample_losses).mean()
        
        # Update student
        weighted_loss.backward()
        student_optimizer.step()
        
        train_losses.append(weighted_loss.item())
        
        # Update mentor every 5 epochs based on validation performance
        if epoch % 5 == 0 and epoch > 0:
            student.eval()
            with torch.no_grad():
                pred_val = student(x_val_t)
                val_loss = criterion(pred_val, y_val_t)
            student.train()
            
            # Mentor objective: minimize validation loss
            # This encourages mentor to weight samples that improve generalization
            mentor_optimizer.zero_grad()
            val_loss.backward()
            mentor_optimizer.step()
            
            val_losses.append(val_loss.item())
        
        if epoch % 10 == 0:
            avg_weight = sample_weights.mean().item()
            print(f"Epoch {epoch}: Train={weighted_loss.item():.4f}, "
                  f"Avg Weight={avg_weight:.3f}")
    
    # Test
    student.eval()
    with torch.no_grad():
        pred_test = student(x_test_t).cpu().numpy()
        if output_size > 1:
            y_pred = pred_test[:, 0]
        else:
            y_pred = pred_test.flatten()
    
    # Calculate metrics
    metrics = calculate_regression_metrics(y_test, y_pred, logging=True)
    
    # Save results
    model_name = f"mentornet_{loss_name}" if loss_name != 'mse' else "mentornet"
    if use_uncertainty:
        model_name += "_unc"
    if use_distance:
        model_name += "_dist"
    
    save_results(args.filepath, s, iteration, model_name, rep,
                args.sample_size, metrics, 'default', loss_name)
    
    # Save sample weights
    if args.uncertainty:
        import pandas as pd
        import os
        
        # Get final weights
        student.eval()
        with torch.no_grad():
            pred_final = student(x_train_t)
            if output_size > 1:
                pred_mean_final = pred_final[:, 0:1]
            else:
                pred_mean_final = pred_final
            
            final_losses = tracking_criterion(pred_mean_final, y_train_t).squeeze()
            
            # Prepare mentor input
            mentor_input_final = [
                final_losses.view(-1, 1),
                loss_history
            ]
            
            if use_uncertainty:
                if loss_name == 'heteroscedastic':
                    log_var = pred_final[:, 1]
                    uncertainty = torch.sqrt(torch.exp(log_var))
                elif loss_name in ['evidential', 'evidential_cauchy', 'evidential_laplace']:
                    v = torch.clamp(F.softplus(pred_final[:, 1]) + 1.0, min=1.0)
                    alpha = torch.clamp(F.softplus(pred_final[:, 2]) + 1.0, min=1.0)
                    beta = torch.clamp(F.softplus(pred_final[:, 3]), min=1e-6)
                    epistemic = beta / torch.clamp(alpha - 1, min=1e-6)
                    aleatoric = beta / (v * torch.clamp(alpha - 1, min=1e-6))
                    uncertainty = torch.sqrt(epistemic + aleatoric)
                elif loss_name == 'sample_adaptive_barron':
                    alpha = torch.sigmoid(pred_final[:, 1]) * 3.9 + 0.1
                    uncertainty = 1.0 / (alpha + 1e-6)
                elif loss_name == 'stratified':
                    uncertainty = torch.sigmoid(pred_final[:, 1])
                
                mentor_input_final.append(uncertainty.view(-1, 1))
            
            if use_distance:
                mentor_input_final.append(distance_features)
            
            mentor_input_final = torch.cat(mentor_input_final, dim=1)
            final_weights = mentor(mentor_input_final).squeeze().cpu().numpy()
        
        weights_dir = os.path.dirname(args.filepath.replace('.csv', '_sample_weights/'))
        os.makedirs(weights_dir, exist_ok=True)
        
        weights_df = pd.DataFrame({
            'sample_idx': np.arange(len(final_weights)),
            'weight': final_weights,
            'loss': final_losses.cpu().numpy(),
            'y_true': y_train,
            'y_pred': pred_mean_final.cpu().numpy().flatten()
        })
        
        weights_file = os.path.join(
            weights_dir,
            f"weights_{model_name}_{rep}_sigma{s}_iter{iteration}_file{file_no}.csv"
        )
        weights_df.to_csv(weights_file, index=False)
    
    if args.save_per_epoch_metrics:
        save_per_epoch_metrics(
            train_losses=train_losses,
            val_losses=val_losses,
            filepath=args.filepath,
            model_name=model_name,
            rep=rep,
            sigma_noise=s,
            iteration=iteration,
            file_no=file_no
        )
    
    return metrics[3]

def train_contrast_to_divide(
    x_train, y_train, x_test, y_test, x_val, y_val,
    args, s, rep, iteration, iteration_seed, file_no, y_test_original,
    trial=None
, train_noise=None):
    """
    Contrast-to-Divide (Yao et al. 2020) adapted for molecular regression.
    
    Uses contrastive learning to separate clean from noisy samples:
    1. Similar molecules should have similar predictions (consistency)
    2. Samples with high prediction variance across augmentations are noisy
    3. Optional: Use distance metrics to define molecular similarity
    """
    from loss_functions import get_loss_function
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    use_distance = getattr(args, 'use_distance', False)
    loss_name = args.loss if hasattr(args, 'loss') else 'mse'
    
    # Hyperparameters
    if trial is not None:
        hidden_size1 = trial.suggest_categorical('hidden_size1', [64, 128, 256])
        hidden_size2 = trial.suggest_categorical('hidden_size2', [32, 64, 128])
        contrast_epochs = trial.suggest_int('contrast_epochs', 20, 50)
        consistency_weight = trial.suggest_float('consistency_weight', 0.1, 1.0)
        clean_threshold = trial.suggest_float('clean_threshold', 0.6, 0.9)
    else:
        hidden_size1, hidden_size2 = 128, 64
        contrast_epochs = 30
        consistency_weight = 0.5  # Balance between prediction loss and consistency
        clean_threshold = 0.75  # Top 75% consistent samples are "clean"
    
    # Loss function setup
    loss_kwargs = {}
    if hasattr(args, 'loss_params') and args.loss_params:
        import json
        loss_kwargs = json.loads(args.loss_params)
    
    output_size_map = {
        'heteroscedastic': 2, 'evidential': 4, 'evidential_cauchy': 4,
        'evidential_laplace': 4, 'sample_adaptive_barron': 2, 'stratified': 2,
    }
    output_size = output_size_map.get(loss_name, 1)
    
    # Prepare data
    x_train_t = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    x_val_t = torch.tensor(x_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
    x_test_t = torch.tensor(x_test, dtype=torch.float32).to(device)
    
    # Compute molecular similarity if using distance
    similarity_matrix = None
    if use_distance:
        from distance_metrics import compute_molecular_distances
        
        distance_method = getattr(args, 'distance_metric', 'tanimoto')
        print(f"Computing {distance_method} distances for contrastive learning...")
        
        distance_matrix = compute_molecular_distances(x_train, rep, method=distance_method)
        
        # Convert distance to similarity (higher similarity = lower distance)
        max_dist = distance_matrix.max()
        similarity_matrix = 1.0 - (distance_matrix / (max_dist + 1e-8))
        similarity_matrix = torch.tensor(similarity_matrix, dtype=torch.float32).to(device)
    
    # Create model
    model = DNNRegressionModel(
        input_size=x_train.shape[1],
        hidden_size1=hidden_size1,
        hidden_size2=hidden_size2
    ).to(device)
    if output_size > 1:
        model.fc3 = nn.Linear(hidden_size2, output_size)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=NEURAL_DEFAULTS['training']['lr'])
    criterion = get_loss_function(loss_name, **loss_kwargs)
    tracking_criterion = nn.MSELoss(reduction='none')
    
    # Phase 1: Contrastive learning with consistency regularization
    print(f"Phase 1: Contrastive learning for {contrast_epochs} epochs...")
    
    consistency_scores = []
    
    for epoch in range(contrast_epochs):
        model.train()
        
        # Create augmented views by adding small noise to inputs
        # (molecular augmentation via noise in feature space)
        noise_scale = 0.01
        x_aug1 = x_train_t + torch.randn_like(x_train_t) * noise_scale
        x_aug2 = x_train_t + torch.randn_like(x_train_t) * noise_scale
        
        optimizer.zero_grad()
        
        # Predictions on original and augmented views
        pred_orig = model(x_train_t)
        pred_aug1 = model(x_aug1)
        pred_aug2 = model(x_aug2)
        
        # Extract mean predictions
        if output_size > 1:
            pred_orig_mean = pred_orig[:, 0:1]
            pred_aug1_mean = pred_aug1[:, 0:1]
            pred_aug2_mean = pred_aug2[:, 0:1]
        else:
            pred_orig_mean = pred_orig
            pred_aug1_mean = pred_aug1
            pred_aug2_mean = pred_aug2
        
        # Prediction loss
        pred_loss = criterion(pred_orig, y_train_t)
        
        # Consistency loss: predictions should be similar across augmentations
        consistency_loss = F.mse_loss(pred_orig_mean, pred_aug1_mean) + \
                          F.mse_loss(pred_orig_mean, pred_aug2_mean)
        
        # Optional: Distance-based consistency
        # Similar molecules should have similar predictions
        if use_distance:
            # For each sample, compute consistency with similar molecules
            # Sample a batch to avoid O(n^2) computation
            batch_size = min(128, len(x_train))
            batch_idx = torch.randperm(len(x_train))[:batch_size]
            
            pred_batch = pred_orig_mean[batch_idx]
            sim_batch = similarity_matrix[batch_idx]
            
            # Pairwise prediction differences
            pred_diff = torch.abs(pred_batch.unsqueeze(1) - pred_batch.unsqueeze(0))
            
            # Weight by similarity (high similarity = should have small difference)
            dist_consistency = (sim_batch * pred_diff).mean()
            
            consistency_loss = consistency_loss + dist_consistency
        
        # Combined loss
        total_loss = pred_loss + consistency_weight * consistency_loss
        
        total_loss.backward()
        optimizer.step()
        
        # Track consistency scores (low = more consistent = likely clean)
        if epoch >= contrast_epochs - 5:  # Last 5 epochs
            model.eval()
            with torch.no_grad():
                pred1 = model(x_train_t)
                pred2 = model(x_train_t + torch.randn_like(x_train_t) * noise_scale)
                pred3 = model(x_train_t + torch.randn_like(x_train_t) * noise_scale)
                
                if output_size > 1:
                    pred1_mean = pred1[:, 0:1]
                    pred2_mean = pred2[:, 0:1]
                    pred3_mean = pred3[:, 0:1]
                else:
                    pred1_mean = pred1
                    pred2_mean = pred2
                    pred3_mean = pred3
                
                # Variance across predictions
                pred_stack = torch.cat([pred1_mean, pred2_mean, pred3_mean], dim=1)
                consistency = pred_stack.std(dim=1).cpu().numpy()
                consistency_scores.append(consistency)
            model.train()
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: Pred Loss={pred_loss.item():.4f}, "
                  f"Consistency={consistency_loss.item():.4f}")
    
    # Phase 2: Divide into clean/noisy based on consistency
    print("Phase 2: Dividing dataset into clean/noisy...")
    
    # Average consistency scores from last 5 epochs
    avg_consistency = np.mean(consistency_scores, axis=0)
    
    # Lower consistency = more stable = likely clean
    consistency_threshold = np.percentile(avg_consistency, clean_threshold * 100)
    clean_mask = avg_consistency < consistency_threshold
    
    print(f"  Identified {clean_mask.sum()}/{len(y_train)} clean samples")
    print(f"  Consistency threshold: {consistency_threshold:.4f}")
    
    # Optional: Refine with distance
    if use_distance:
        # Check if "clean" samples form a coherent cluster
        from distance_metrics import compute_molecular_distances
        
        clean_indices = np.where(clean_mask)[0]
        
        if len(clean_indices) > 10:
            # Compute distances among clean samples
            distance_matrix_clean = compute_molecular_distances(
                x_train, rep, method=distance_method,
                subset_indices=clean_indices
            )
            
            # Average distance to other clean samples
            avg_distances = np.zeros(len(clean_indices))
            for i in range(len(clean_indices)):
                distances_i = np.concatenate([
                    distance_matrix_clean[i, :i],
                    distance_matrix_clean[i, i+1:]
                ])
                avg_distances[i] = distances_i.mean()
            
            # Remove isolated "clean" samples (might be false positives)
            distance_threshold = np.percentile(avg_distances, 75)
            close_to_cluster = avg_distances < distance_threshold
            
            refined_clean_indices = clean_indices[close_to_cluster]
            
            refined_clean_mask = np.zeros(len(y_train), dtype=bool)
            refined_clean_mask[refined_clean_indices] = True
            
            print(f"  Distance-refined: {refined_clean_mask.sum()}/{clean_mask.sum()} clean samples")
            
            clean_mask = refined_clean_mask
    
    # Phase 3: Retrain on clean samples
    print(f"Phase 3: Retraining on {clean_mask.sum()} clean samples...")
    
    x_clean = x_train_t[clean_mask]
    y_clean = y_train_t[clean_mask]
    
    # Fresh model
    model2 = DNNRegressionModel(
        input_size=x_train.shape[1],
        hidden_size1=hidden_size1,
        hidden_size2=hidden_size2
    ).to(device)
    if output_size > 1:
        model2.fc3 = nn.Linear(hidden_size2, output_size)
    
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(args.epochs - contrast_epochs):
        model2.train()
        optimizer2.zero_grad()
        pred = model2(x_clean)
        loss = criterion(pred, y_clean)
        loss.backward()
        optimizer2.step()
        
        train_losses.append(loss.item())
        
        # Validation
        model2.eval()
        with torch.no_grad():
            pred_val = model2(x_val_t)
            val_loss = criterion(pred_val, y_val_t)
        val_losses.append(val_loss.item())
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: Train={loss.item():.4f}, Val={val_loss.item():.4f}")
    
    # Test
    model2.eval()
    with torch.no_grad():
        pred_test = model2(x_test_t).cpu().numpy()
        if output_size > 1:
            y_pred = pred_test[:, 0]
        else:
            y_pred = pred_test.flatten()
    
    # Calculate metrics
    metrics = calculate_regression_metrics(y_test, y_pred, logging=True)
    
    # Save results
    model_name = f"contrast_divide_{loss_name}" if loss_name != 'mse' else "contrast_divide"
    if use_distance:
        model_name += "_dist"
    
    save_results(args.filepath, s, iteration, model_name, rep,
                args.sample_size, metrics, 'default', loss_name)
    
    # Save division info
    if args.uncertainty:
        import pandas as pd
        import os
        division_dir = os.path.dirname(args.filepath.replace('.csv', '_division/'))
        os.makedirs(division_dir, exist_ok=True)
        
        division_df = pd.DataFrame({
            'sample_idx': np.arange(len(clean_mask)),
            'clean': clean_mask.astype(int),
            'consistency_score': avg_consistency,
            'y_true': y_train,
        })
        
        division_file = os.path.join(
            division_dir,
            f"division_{model_name}_{rep}_sigma{s}_iter{iteration}_file{file_no}.csv"
        )
        division_df.to_csv(division_file, index=False)
    
    if args.save_per_epoch_metrics:
        save_per_epoch_metrics(
            train_losses=train_losses,
            val_losses=val_losses,
            filepath=args.filepath,
            model_name=model_name,
            rep=rep,
            sigma_noise=s,
            iteration=iteration,
            file_no=file_no
        )
    
    return metrics[3]

def train_distance_based_selection(
    x_train, y_train, x_test, y_test, x_val, y_val,
    args, s, rep, iteration, iteration_seed, file_no, y_test_original,
    trial=None
, train_noise=None):
    """
    Distance-Based Selection - pure molecular distance approach.
    
    Strategy:
    1. Identify outlier molecules (far from distribution)
    2. Optional: Train initial model and combine distance with loss/uncertainty
    3. Remove outliers and retrain
    
    Rationale: Noisy labels often occur on atypical/outlier molecules
    """
    from loss_functions import get_loss_function
    from distance_metrics import (
        identify_outliers_by_distance,
        distance_weighted_sample_selection
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    loss_name = args.loss if hasattr(args, 'loss') else 'mse'
    distance_method = getattr(args, 'distance_metric', 'tanimoto')
    
    # Check if we should use loss/uncertainty in addition to distance
    use_loss_unc = loss_name in ['heteroscedastic', 'evidential', 'evidential_cauchy', 
                                   'evidential_laplace', 'sample_adaptive_barron', 'stratified']
    
    # Hyperparameters
    if trial is not None:
        hidden_size1 = trial.suggest_categorical('hidden_size1', [64, 128, 256])
        hidden_size2 = trial.suggest_categorical('hidden_size2', [32, 64, 128])
        outlier_percentile = trial.suggest_float('outlier_percentile', 75, 95)
        warmup_epochs = trial.suggest_int('warmup_epochs', 10, 30) if use_loss_unc else 0
        combine_strategy = trial.suggest_categorical('combine_strategy', 
                                                     ['distance_only', 'distance_loss', 'distance_loss_unc']) if use_loss_unc else 'distance_only'
    else:
        hidden_size1, hidden_size2 = 128, 64
        outlier_percentile = 85  # Remove top 15% most distant
        warmup_epochs = 20 if use_loss_unc else 0
        combine_strategy = 'distance_loss_unc' if use_loss_unc else 'distance_only'
    
    # Loss function setup
    loss_kwargs = {}
    if hasattr(args, 'loss_params') and args.loss_params:
        import json
        loss_kwargs = json.loads(args.loss_params)
    
    output_size_map = {
        'heteroscedastic': 2, 'evidential': 4, 'evidential_cauchy': 4,
        'evidential_laplace': 4, 'sample_adaptive_barron': 2, 'stratified': 2,
    }
    output_size = output_size_map.get(loss_name, 1)
    
    # Prepare data
    x_train_t = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    x_val_t = torch.tensor(x_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
    x_test_t = torch.tensor(x_test, dtype=torch.float32).to(device)
    
    # Phase 1: Identify distance-based outliers
    print(f"Phase 1: Identifying molecular outliers using {distance_method} distance...")
    
    outlier_mask, avg_distances = identify_outliers_by_distance(
        x_train, rep, method=distance_method,
        threshold_percentile=outlier_percentile
    )
    
    print(f"  Distance-based outliers: {outlier_mask.sum()}/{len(y_train)}")
    print(f"  Distance range: {avg_distances.min():.4f} to {avg_distances.max():.4f}")
    
    # Phase 2: Optionally refine with loss/uncertainty
    if combine_strategy != 'distance_only' and use_loss_unc:
        print(f"Phase 2: Training initial model for {warmup_epochs} epochs...")
        
        # Train initial model
        model = DNNRegressionModel(
            input_size=x_train.shape[1],
            hidden_size1=hidden_size1,
            hidden_size2=hidden_size2
        ).to(device)
        if output_size > 1:
            model.fc3 = nn.Linear(hidden_size2, output_size)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=NEURAL_DEFAULTS['training']['lr'])
        criterion = get_loss_function(loss_name, **loss_kwargs)
        tracking_criterion = nn.MSELoss(reduction='none')
        
        for epoch in range(warmup_epochs):
            model.train()
            optimizer.zero_grad()
            pred = model(x_train_t)
            loss = criterion(pred, y_train_t)
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: Loss={loss.item():.4f}")
        
        # Get predictions and uncertainties
        model.eval()
        with torch.no_grad():
            pred_all = model(x_train_t).cpu().numpy()
            
            # Extract mean and uncertainty
            if output_size > 1:
                pred_mean = pred_all[:, 0]
            else:
                pred_mean = pred_all.flatten()
            
            # Calculate losses
            pred_mean_t = torch.tensor(pred_mean, dtype=torch.float32).view(-1, 1).to(device)
            losses = tracking_criterion(pred_mean_t, y_train_t).squeeze().cpu().numpy()
            
            # Extract uncertainty if available
            if use_loss_unc:
                if loss_name == 'heteroscedastic':
                    log_var = pred_all[:, 1]
                    uncertainties = np.sqrt(np.exp(log_var))
                
                elif loss_name in ['evidential', 'evidential_cauchy', 'evidential_laplace']:
                    v = np.maximum(pred_all[:, 1], 1.0)
                    alpha = np.maximum(pred_all[:, 2], 1.0)
                    beta = np.maximum(pred_all[:, 3], 1e-6)
                    epistemic = beta / np.maximum(alpha - 1, 1e-6)
                    aleatoric = beta / (v * np.maximum(alpha - 1, 1e-6))
                    uncertainties = np.sqrt(epistemic + aleatoric)
                
                elif loss_name == 'sample_adaptive_barron':
                    alpha = 1 / (1 + np.exp(-pred_all[:, 1])) * 3.9 + 0.1
                    uncertainties = 1.0 / (alpha + 1e-6)
                
                elif loss_name == 'stratified':
                    uncertainties = 1 / (1 + np.exp(-pred_all[:, 1]))
            else:
                uncertainties = None
        
        # Combine distance with loss/uncertainty
        print("  Combining distance with model predictions...")
        
        if combine_strategy == 'distance_loss':
            # Use distance-weighted sample selection
            keep_mask, scores = distance_weighted_sample_selection(
                x_train, losses, None, rep,
                method=distance_method,
                keep_fraction=1.0 - (outlier_percentile / 100.0)
            )
        
        elif combine_strategy == 'distance_loss_unc':
            # Use all three: distance, loss, uncertainty
            keep_mask, scores = distance_weighted_sample_selection(
                x_train, losses, uncertainties, rep,
                method=distance_method,
                keep_fraction=1.0 - (outlier_percentile / 100.0)
            )
        
        noisy_mask = ~keep_mask
        
        print(f"  Combined selection: Keeping {keep_mask.sum()}/{len(y_train)} samples")
        print(f"  Removed {noisy_mask.sum()} samples (distance + loss + unc)")
    
    else:
        # Pure distance-based selection
        noisy_mask = outlier_mask
        keep_mask = ~noisy_mask
    
    # Phase 3: Train on cleaned data
    print(f"Phase 3: Training on {keep_mask.sum()} selected samples...")
    
    x_clean = x_train_t[keep_mask]
    y_clean = y_train_t[keep_mask]
    
    # Create fresh model
    model_final = DNNRegressionModel(
        input_size=x_train.shape[1],
        hidden_size1=hidden_size1,
        hidden_size2=hidden_size2
    ).to(device)
    if output_size > 1:
        model_final.fc3 = nn.Linear(hidden_size2, output_size)
    
    optimizer_final = torch.optim.Adam(model_final.parameters(), lr=0.001)
    criterion_final = get_loss_function(loss_name, **loss_kwargs)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(args.epochs - warmup_epochs):
        model_final.train()
        optimizer_final.zero_grad()
        pred = model_final(x_clean)
        loss = criterion_final(pred, y_clean)
        loss.backward()
        optimizer_final.step()
        
        train_losses.append(loss.item())
        
        # Validation
        model_final.eval()
        with torch.no_grad():
            pred_val = model_final(x_val_t)
            val_loss = criterion_final(pred_val, y_val_t)
        val_losses.append(val_loss.item())
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: Train={loss.item():.4f}, Val={val_loss.item():.4f}")
    
    # Test
    model_final.eval()
    with torch.no_grad():
        pred_test = model_final(x_test_t).cpu().numpy()
        if output_size > 1:
            y_pred = pred_test[:, 0]
        else:
            y_pred = pred_test.flatten()
    
    # Calculate metrics
    metrics = calculate_regression_metrics(y_test, y_pred, logging=True)
    
    # Save results
    model_name = f"distance_select_{distance_method}"
    if combine_strategy != 'distance_only':
        model_name += f"_{combine_strategy}"
    if loss_name != 'mse':
        model_name += f"_{loss_name}"
    
    save_results(args.filepath, s, iteration, model_name, rep,
                args.sample_size, metrics, 'default', loss_name)
    
    # Save selection info
    if args.uncertainty:
        import pandas as pd
        import os
        selection_dir = os.path.dirname(args.filepath.replace('.csv', '_selection/'))
        os.makedirs(selection_dir, exist_ok=True)
        
        selection_df = pd.DataFrame({
            'sample_idx': np.arange(len(keep_mask)),
            'kept': keep_mask.astype(int),
            'avg_distance': avg_distances,
            'outlier': outlier_mask.astype(int),
            'y_true': y_train,
        })
        
        if combine_strategy != 'distance_only' and use_loss_unc:
            selection_df['loss'] = losses
            if uncertainties is not None:
                selection_df['uncertainty'] = uncertainties
        
        selection_file = os.path.join(
            selection_dir,
            f"selection_{model_name}_{rep}_sigma{s}_iter{iteration}_file{file_no}.csv"
        )
        selection_df.to_csv(selection_file, index=False)
    
    if args.save_per_epoch_metrics:
        save_per_epoch_metrics(
            train_losses=train_losses,
            val_losses=val_losses,
            filepath=args.filepath,
            model_name=model_name,
            rep=rep,
            sigma_noise=s,
            iteration=iteration,
            file_no=file_no
        )
    
    return metrics[3]

def train_heteroscedastic_gp(
    x_train, y_train, x_test, y_test, x_val, y_val,
    args, s, rep, iteration, iteration_seed, file_no, y_test_original,
    trial=None
, train_noise=None):
    """Heteroscedastic Gaussian Process with learned noise variance"""
    import gpytorch
    from gpytorch.models import ExactGP
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.distributions import MultivariateNormal
    import torch.nn as nn
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get kernel
    kernel_type = args.kernel if hasattr(args, 'kernel') else 'tanimoto'
    
    # Prepare data
    x_train_t = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    x_val_t = torch.tensor(x_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)
    x_test_t = torch.tensor(x_test, dtype=torch.float32).to(device)
    
    # Two-model approach: GP for mean + NN for noise
    class HeteroscedasticGPModel(ExactGP):
        def __init__(self, train_x, train_y, likelihood, kernel_type='tanimoto'):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            
            if kernel_type == 'tanimoto':
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    TanimotoKernel()
                )
            elif kernel_type == 'rbf':
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel()
                )
            elif kernel_type == 'matern':
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.MaternKernel(nu=2.5)
                )
        
        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return MultivariateNormal(mean_x, covar_x)
    
    # Separate neural network for noise prediction
    class NoiseModel(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Softplus()  # Ensure positive output
            )
        
        def forward(self, x):
            return self.net(x).squeeze(-1) + 1e-4  # Add small epsilon
    
    # Create models
    likelihood = GaussianLikelihood().to(device)
    gp_model = HeteroscedasticGPModel(
        x_train_t, y_train_t, likelihood, kernel_type
    ).to(device)
    noise_model = NoiseModel(x_train.shape[1]).to(device)

    # Training
    gp_model.train()
    likelihood.train()
    noise_model.train()

    optimizer = torch.optim.Adam([
        {'params': gp_model.parameters(), 'lr': 0.1},
        {'params': noise_model.parameters(), 'lr': 0.001}
    ], lr=0.1)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)

    print(f"Training Heteroscedastic GP with {kernel_type} kernel...")

    for epoch in range(args.epochs):
        # Train GP
        optimizer.zero_grad()
        output = gp_model(x_train_t)
        gp_loss = -mll(output, y_train_t)
        
        # Train noise model
        with torch.no_grad():
            gp_pred = gp_model(x_train_t).mean
        residuals = (y_train_t - gp_pred) ** 2
        pred_var = noise_model(x_train_t)
        
        # Negative log-likelihood for noise model
        noise_loss = torch.mean(0.5 * torch.log(pred_var) + residuals / (2 * pred_var))
        
        # Combined loss
        total_loss = gp_loss + noise_loss
        total_loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: GP Loss = {gp_loss.item():.4f}, Noise Loss = {noise_loss.item():.4f}")
    
    # Test
    gp_model.eval()
    likelihood.eval()
    noise_model.eval()
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # GP predictions
        pred_dist = gp_model(x_test_t)
        pred_mean = pred_dist.mean.cpu().numpy()
        
        # Epistemic uncertainty (from GP)
        epistemic_var = pred_dist.variance.cpu().numpy()
        epistemic_std = np.sqrt(epistemic_var)
        
        # Aleatoric uncertainty (learned noise)
        aleatoric_var = noise_model(x_test_t).cpu().numpy()
        aleatoric_std = np.sqrt(aleatoric_var)
        
        # Total uncertainty
        total_std = np.sqrt(epistemic_var + aleatoric_var)
    
    # Calculate metrics
    metrics = calculate_regression_metrics(y_test, pred_mean, logging=True)
    
    # Save results
    model_name = f"het_gp_{kernel_type}"
    save_results(args.filepath, s, iteration, model_name, rep,
                args.sample_size, metrics, 'default', 'het_gp')
    
    # Save uncertainty (total std as uncalibrated)
    if args.uncertainty:
        save_uncertainty_values(
            y_pred_mean=pred_mean,
            y_pred_std=total_std,
            # These two are the whole point of this model, and the call used to
            # omit them, so the columns came out NaN. It and evidential_kernel
            # are the only models in the file that produce a per-molecule
            # ALEATORIC term (RERUN_PLAN.md 2.13).
            epistemic_uncertainty=epistemic_std,
            aleatoric_uncertainty=aleatoric_std,
            y_true_original=y_test_original,
            y_true_noisy=y_test,
            filepath=args.filepath,
            model_name=model_name,
            rep=rep,
            sigma_noise=s,
            iteration=iteration,
            file_no=file_no,
        )

    
    return metrics[3]

def train_evidential_kernel(
    x_train, y_train, x_test, y_test, x_val, y_val,
    args, s, rep, iteration, iteration_seed, file_no, y_test_original,
    trial=None
, train_noise=None):
    """Evidential Kernel: GP predicting evidential parameters"""
    import gpytorch
    from gpytorch.models import ExactGP
    from gpytorch.likelihoods import MultitaskGaussianLikelihood
    from gpytorch.distributions import MultitaskMultivariateNormal
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get kernel
    kernel_type = args.kernel if hasattr(args, 'kernel') else 'tanimoto'
    
    # Prepare data
    x_train_t = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    x_test_t = torch.tensor(x_test, dtype=torch.float32).to(device)
    
    # Evidential GP Model (outputs 4 parameters)
    class EvidentialGPModel(ExactGP):
        def __init__(self, train_x, train_y, likelihood, kernel_type='tanimoto'):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.MultitaskMean(
                gpytorch.means.ConstantMean(), num_tasks=4
            )
            
            if kernel_type == 'tanimoto':
                base_kernel = TanimotoKernel()
            elif kernel_type == 'rbf':
                base_kernel = gpytorch.kernels.RBFKernel()
            elif kernel_type == 'matern':
                base_kernel = gpytorch.kernels.MaternKernel(nu=2.5)
            
            self.covar_module = gpytorch.kernels.MultitaskKernel(
                gpytorch.kernels.ScaleKernel(base_kernel),
                num_tasks=4,
                rank=1
            )
        
        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return MultitaskMultivariateNormal(mean_x, covar_x)
    
    # Create synthetic targets for evidential parameters
    # Initialize with reasonable values
    gamma_init = y_train_t.clone()
    v_init = torch.ones_like(y_train_t) * 5.0
    alpha_init = torch.ones_like(y_train_t) * 2.0
    beta_init = torch.ones_like(y_train_t) * 0.1
    
    y_train_multi = torch.stack([gamma_init, v_init, alpha_init, beta_init], dim=-1)
    
    # Create model
    likelihood = MultitaskGaussianLikelihood(num_tasks=4).to(device)
    model = EvidentialGPModel(x_train_t, y_train_multi, likelihood, kernel_type).to(device)
    
    # Training
    model.train()
    likelihood.train()
    
    # FIX: Use a single parameter group
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    # Don't add likelihood parameters separately - they're already in model.parameters()
    
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    print(f"Training Evidential Kernel with {kernel_type} kernel...")
    
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        output = model(x_train_t)
        loss = -mll(output, y_train_multi)
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: Loss = {loss.item():.4f}")
    
    # Test
    model.eval()
    likelihood.eval()
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred_dist = likelihood(model(x_test_t))
        pred_params = pred_dist.mean.cpu().numpy()
        
        gamma = pred_params[:, 0]
        v = np.maximum(pred_params[:, 1], 1.0)
        alpha = np.maximum(pred_params[:, 2], 1.0)
        beta = np.maximum(pred_params[:, 3], 1e-6)
        
        # Compute uncertainties
        epistemic_std = np.sqrt(beta / (alpha - 1))
        aleatoric_std = np.sqrt(beta / (v * (alpha - 1)))
        total_std = np.sqrt(epistemic_std**2 + aleatoric_std**2)
    
    # Calculate metrics
    metrics = calculate_regression_metrics(y_test, gamma, logging=True)
    
    # Save results
    model_name = f"evidential_kernel_{kernel_type}"
    save_results(args.filepath, s, iteration, model_name, rep,
                args.sample_size, metrics, 'default', 'evidential_kernel')
    
    # Save uncertainty (total std as uncalibrated)
    if args.uncertainty:
        save_uncertainty_values(
            y_pred_mean=gamma,
            y_pred_std=total_std,
            # These two are the whole point of this model, and the call used to
            # omit them, so the columns came out NaN. It and evidential_kernel
            # are the only models in the file that produce a per-molecule
            # ALEATORIC term (RERUN_PLAN.md 2.13).
            epistemic_uncertainty=epistemic_std,
            aleatoric_uncertainty=aleatoric_std,
            y_true_original=y_test_original,
            y_true_noisy=y_test,
            filepath=args.filepath,
            model_name=model_name,
            rep=rep,
            sigma_noise=s,
            iteration=iteration,
            file_no=file_no,
        )

    
    return metrics[3]

def train_ntk_gnn(
    train_loader, test_loader, val_loader, args, s, iteration, file_no,
    y_test_original, trial,
    y_train_noisy=None, y_test_noisy=None, y_val_noisy=None
, train_noise=None):
    """Neural Tangent Kernel of GNN for molecular graphs.

    REFUSED. It takes y_train_noisy, y_test_noisy and y_val_noisy and never
    references any of them -- it reads `data.y`, the untouched PyG attribute, so
    it trains and scores on the CLEAN labels whatever noise level the run is at.
    Its degradation curve would be flat by construction, and that flatness would
    read as robustness (RERUN_PLAN.md 2.13). It is not in the job generator's
    roster and has never produced a number.
    """
    raise NotImplementedError(
        "train_ntk_gnn ignores the noisy labels entirely -- it reads data.y, "
        "the clean PyG attribute, so it would be trained and scored on clean "
        "labels at every noise level and its flat curve would read as "
        "robustness. It is not in the study roster. See RERUN_PLAN.md 2.13.")

    import torch
    import torch.nn as nn
    from torch_geometric.nn import GINConv, global_mean_pool
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters
    if trial is not None:
        dim_h = trial.suggest_categorical('dim_h', [32, 64, 128])
        n_layers = trial.suggest_int('n_layers', 2, 4)
        ridge_lambda = trial.suggest_float('ridge_lambda', 1e-4, 1e-2, log=True)
    else:
        dim_h = 64
        n_layers = 3
        ridge_lambda = 1e-3
    
    # GNN Model
    class GNNModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, n_layers):
            super().__init__()
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            
            # First layer
            mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            
            # Hidden layers
            for _ in range(n_layers - 1):
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
                self.convs.append(GINConv(mlp))
                self.bns.append(nn.BatchNorm1d(hidden_dim))
            
            # Output layer
            self.fc = nn.Linear(hidden_dim, 1)
        
        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            
            for conv, bn in zip(self.convs, self.bns):
                x = conv(x, edge_index)
                x = bn(x)
                x = torch.relu(x)
            
            x = global_mean_pool(x, batch)
            return self.fc(x)
    
    # Get input dimension and initialize model FIRST
    input_dim = None
    model = None
    
    for batch in train_loader:
        input_dim = batch.x.size(1)
        break
    
    if input_dim is None:
        print("Error: Empty train_loader, cannot determine input dimension")
        return 0.0
    
    # NOW initialize model (after we know input_dim)
    model = GNNModel(input_dim, dim_h, n_layers).to(device)
    
    print(f"Computing Neural Tangent Kernel for GNN...")
    print(f"  Architecture: {input_dim} → {dim_h} (×{n_layers}) → 1")
    print(f"  Warning: This is computationally expensive!")
    
    # Collect training data
    X_train, y_train = [], []
    for data in train_loader:
        X_train.append(data.to(device))
        y_train.append(data.y.to(device))
    y_train = torch.cat(y_train, dim=0)
    
    n_train = len(X_train)
    
    if n_train > 1000:
        print(f"  WARNING: n_train = {n_train} is large. This may take a very long time.")
        print(f"  Consider reducing sample size for NTK-GNN.")
    
    # Compute NTK matrix
    print("  Computing NTK matrix...")
    K = torch.zeros(n_train, n_train, device=device)
    
    for i in range(n_train):
        model.zero_grad()
        output_i = model(X_train[i])
        
        # Get gradients
        grads_i = torch.autograd.grad(
            output_i, model.parameters(),
            create_graph=False, retain_graph=False
        )
        grads_i_flat = torch.cat([g.flatten() for g in grads_i])
        
        for j in range(i, n_train):
            model.zero_grad()
            output_j = model(X_train[j])
            
            grads_j = torch.autograd.grad(
                output_j, model.parameters(),
                create_graph=False, retain_graph=False
            )
            grads_j_flat = torch.cat([g.flatten() for g in grads_j])
            
            # NTK: inner product of gradients
            K[i, j] = torch.dot(grads_i_flat, grads_j_flat)
            K[j, i] = K[i, j]
        
        if (i + 1) % 100 == 0:
            print(f"    Processed {i+1}/{n_train} samples")
    
    # Kernel ridge regression
    print("  Solving kernel ridge regression...")
    K_reg = K + ridge_lambda * torch.eye(n_train, device=device)
    alpha = torch.linalg.solve(K_reg, y_train)
    
    # Test predictions
    print("  Computing test predictions...")
    X_test, y_test = [], []
    for data in test_loader:
        X_test.append(data.to(device))
        y_test.append(data.y.to(device))
    y_test = torch.cat(y_test, dim=0).cpu().numpy()
    
    n_test = len(X_test)
    predictions = torch.zeros(n_test, device=device)
    variances = torch.zeros(n_test, device=device)
    
    for i in range(n_test):
        model.zero_grad()
        output_i = model(X_test[i])
        
        grads_i = torch.autograd.grad(
            output_i, model.parameters(),
            create_graph=False, retain_graph=False
        )
        grads_i_flat = torch.cat([g.flatten() for g in grads_i])
        
        # Compute kernel vector k(x_test, X_train)
        k_vec = torch.zeros(n_train, device=device)
        for j in range(n_train):
            model.zero_grad()
            output_j = model(X_train[j])
            
            grads_j = torch.autograd.grad(
                output_j, model.parameters(),
                create_graph=False, retain_graph=False
            )
            grads_j_flat = torch.cat([g.flatten() for g in grads_j])
            
            k_vec[j] = torch.dot(grads_i_flat, grads_j_flat)
        
        # Prediction
        predictions[i] = torch.dot(k_vec, alpha)
        
        # Variance: k(x*,x*) - k(x*,X)(K+λI)^{-1}k(X,x*)
        k_star_star = torch.dot(grads_i_flat, grads_i_flat)
        variances[i] = k_star_star - torch.dot(k_vec, torch.linalg.solve(K_reg, k_vec))
        
        if (i + 1) % 100 == 0:
            print(f"    Processed {i+1}/{n_test} test samples")
    
    y_pred = predictions.cpu().numpy()
    uncertainty = torch.sqrt(torch.clamp(variances, min=0)).cpu().numpy()
    
    # Calculate metrics
    metrics = calculate_regression_metrics(y_test, y_pred, logging=True)
    
    # Save results
    model_name = f"ntk_gnn_layers{n_layers}"
    save_results(args.filepath, s, iteration, model_name, 'graph',
                args.sample_size, metrics, 'default', 'ntk_gnn')
    
    # Save uncertainty (NTK variance as uncalibrated std)
    if args.uncertainty:
        save_uncertainty_values(
            y_pred_mean=y_pred,
            y_pred_std=uncertainty,
            y_true_original=y_test_original,
            y_true_noisy=y_test,
            filepath=args.filepath,
            model_name=model_name,
            rep='graph',
            sigma_noise=s,
            iteration=iteration,
            file_no=file_no,
        )
    
    return metrics[3]

def train_conformal_heteroscedastic(
    x_train, y_train, x_test, y_test, x_val, y_val,
    args, s, rep, iteration, iteration_seed, file_no, y_test_original,
    trial=None
, train_noise=None):
    """
    Conformal Prediction + Heteroscedastic Neural Network (Romano et al. 2019 + novel).
    
    Combines:
    1. Learned input-dependent uncertainty (heteroscedastic NLL)
    2. Distribution-free conformal prediction intervals
    
    Provides both model uncertainty and calibrated prediction intervals.
    Can compare learned vs. conformal uncertainty.
    """
    from loss_functions import get_loss_function
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters
    if trial is not None:
        hidden_size1 = trial.suggest_categorical('hidden_size1', [64, 128, 256])
        hidden_size2 = trial.suggest_categorical('hidden_size2', [32, 64, 128])
        alpha = trial.suggest_float('alpha', 0.05, 0.2)
        calibration_method = trial.suggest_categorical('calibration_method', 
                                                       ['split', 'cv', 'jackknife'])
        combine_method = trial.suggest_categorical('combine_method',
                                                   ['separate', 'uncertainty_weighted', 'adaptive'])
    else:
        hidden_size1, hidden_size2 = 128, 64
        alpha = 0.1  # 90% coverage
        calibration_method = 'split'
        combine_method = 'uncertainty_weighted'  # Weight conformal by learned uncertainty
    
    # Prepare data
    x_train_t = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    x_val_t = torch.tensor(x_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
    x_test_t = torch.tensor(x_test, dtype=torch.float32).to(device)
    
    # Split validation into calibration and validation
    # --calibration-size is commented out in process_and_train.py; this is the
    # only function that ever read it, and it is not in the study roster either.
    # 20% was its default (RERUN_PLAN.md 2.13).
    cal_size = int(len(x_val) * (getattr(args, 'calibration_size', 20) / 100.0))
    x_val_cal = x_val_t[:cal_size]
    y_val_cal = y_val_t[:cal_size]
    x_val_proper = x_val_t[cal_size:]
    y_val_proper = y_val_t[cal_size:]
    
    print(f"Train: {len(x_train)}, Cal: {cal_size}, Val: {len(x_val_proper)}, Test: {len(x_test)}")
    
    # Create heteroscedastic model (outputs mean and log variance)
    model = DNNRegressionModel(
        input_size=x_train.shape[1],
        hidden_size1=hidden_size1,
        hidden_size2=hidden_size2
    ).to(device)
    model.fc3 = nn.Linear(hidden_size2, 2)  # mean + log_var
    
    optimizer = torch.optim.Adam(model.parameters(), lr=NEURAL_DEFAULTS['training']['lr'])
    criterion = get_loss_function('heteroscedastic')
    
    # Phase 1: Train heteroscedastic model
    print("Phase 1: Training heteroscedastic model...")
    
    train_losses = []
    val_losses = []
    
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(x_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Validation
        if len(x_val_proper) > 0:
            model.eval()
            with torch.no_grad():
                pred_val = model(x_val_proper)
                val_loss = criterion(pred_val, y_val_proper)
            val_losses.append(val_loss.item())
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: Train Loss={loss.item():.4f}")
    
    # Phase 2: Conformal calibration
    print("Phase 2: Conformal calibration...")
    
    model.eval()
    with torch.no_grad():
        # Get calibration predictions
        pred_cal = model(x_val_cal).cpu().numpy()
        mean_cal = pred_cal[:, 0]
        log_var_cal = pred_cal[:, 1]
        std_cal = np.sqrt(np.exp(log_var_cal))
        
        y_cal_np = y_val_cal.cpu().numpy().flatten()
    
    if calibration_method == 'split':
        # Standard conformal: compute non-conformity scores
        if combine_method == 'separate':
            # Simple absolute residuals
            scores_cal = np.abs(y_cal_np - mean_cal)
        
        elif combine_method == 'uncertainty_weighted':
            # Normalize residuals by learned uncertainty
            # Score = |y - mean| / std
            scores_cal = np.abs(y_cal_np - mean_cal) / (std_cal + 1e-6)
        
        elif combine_method == 'adaptive':
            # Adaptive score that combines both
            # Use CQR-style (Conformalized Quantile Regression) approach
            # Score = max(lower_err, upper_err) where bounds come from learned std
            lower_bound = mean_cal - std_cal
            upper_bound = mean_cal + std_cal
            scores_cal = np.maximum(lower_bound - y_cal_np, y_cal_np - upper_bound)
        
        # Compute quantile
        n_cal = len(scores_cal)
        q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
        q_level = min(q_level, 1.0)
        quantile = np.quantile(scores_cal, q_level)
        
        print(f"  Conformal quantile (α={alpha}): {quantile:.4f}")
    
    # Phase 3: Test predictions with combined uncertainty
    print("Phase 3: Computing predictions and intervals...")
    
    model.eval()
    with torch.no_grad():
        pred_test = model(x_test_t).cpu().numpy()
        mean_test = pred_test[:, 0]
        log_var_test = pred_test[:, 1]
        std_test = np.sqrt(np.exp(log_var_test))
    
    # Construct prediction intervals based on combine method
    if combine_method == 'separate':
        # Conformal intervals (ignore learned uncertainty)
        y_lower_conformal = mean_test - quantile
        y_upper_conformal = mean_test + quantile
        
        # Learned uncertainty intervals (1.645 for 90% coverage under Gaussian)
        z_score = 1.645 if alpha == 0.1 else 1.96
        y_lower_learned = mean_test - z_score * std_test
        y_upper_learned = mean_test + z_score * std_test
        
        # Use conformal for final predictions
        y_lower = y_lower_conformal
        y_upper = y_upper_conformal
    
    elif combine_method == 'uncertainty_weighted':
        # Conformal with uncertainty weighting
        # Interval width adapts to learned uncertainty
        y_lower = mean_test - quantile * std_test
        y_upper = mean_test + quantile * std_test
    
    elif combine_method == 'adaptive':
        # CQR-style adaptive intervals
        y_lower = mean_test - std_test - quantile
        y_upper = mean_test + std_test + quantile
    
    # Calculate coverage and metrics
    coverage = np.mean((y_test >= y_lower) & (y_test <= y_upper))
    avg_interval_width = np.mean(y_upper - y_lower)
    
    print(f"Results:")
    print(f"  Coverage: {coverage:.4f} (target: {1-alpha:.4f})")
    print(f"  Avg interval width: {avg_interval_width:.4f}")
    print(f"  Avg learned std: {std_test.mean():.4f}")
    
    # Point predictions
    y_pred = mean_test
    
    # Calculate metrics
    metrics = calculate_regression_metrics(y_test, y_pred, logging=True)
    
    # Save results
    model_name = f"conformal_hetero_{combine_method}"
    
    save_results(args.filepath, s, iteration, model_name, rep,
                args.sample_size, metrics, 'default', 'conformal_heteroscedastic')
    
    # Save detailed uncertainty information
    if args.uncertainty:
        import pandas as pd
        import os
        
        uncertainty_dir = os.path.dirname(args.filepath.replace('.csv', '_uncertainty/'))
        os.makedirs(uncertainty_dir, exist_ok=True)
        
        uncertainty_df = pd.DataFrame({
            'sample_idx': np.arange(len(y_pred)),
            'y_pred_mean': y_pred,
            'y_pred_std_learned': std_test,
            'y_lower_conformal': y_lower,
            'y_upper_conformal': y_upper,
            'interval_width': y_upper - y_lower,
            'coverage': ((y_test >= y_lower) & (y_test <= y_upper)).astype(int),
            'y_true_noisy': y_test,
            'y_true_original': y_test_original,
        })
        
        # Add separate intervals if using 'separate' method
        if combine_method == 'separate':
            uncertainty_df['y_lower_learned'] = y_lower_learned
            uncertainty_df['y_upper_learned'] = y_upper_learned
            uncertainty_df['interval_width_learned'] = y_upper_learned - y_lower_learned
        
        uncertainty_file = os.path.join(
            uncertainty_dir,
            f"uncertainty_{model_name}_{rep}_sigma{s}_iter{iteration}_file{file_no}.csv"
        )
        uncertainty_df.to_csv(uncertainty_file, index=False)
        
        print(f"Saved combined uncertainty to {uncertainty_file}")
        
        # Additional analysis
        if combine_method == 'separate':
            coverage_learned = np.mean((y_test >= y_lower_learned) & (y_test <= y_upper_learned))
            print(f"  Learned intervals coverage: {coverage_learned:.4f}")
            print(f"  Conformal interval width: {(y_upper_conformal - y_lower_conformal).mean():.4f}")
            print(f"  Learned interval width: {(y_upper_learned - y_lower_learned).mean():.4f}")
    
    # Save conformal intervals in standard format
    if args.uncertainty:
        save_conformal_intervals(
            y_pred=y_pred,
            y_lower=y_lower,
            y_upper=y_upper,
            y_true=y_test,
            filepath=args.filepath,
            model_name=model_name,
            rep=rep,
            sigma_noise=s,
            iteration=iteration,
            file_no=file_no,
            alpha=alpha
        )
    
    # Save per-epoch metrics
    if args.save_per_epoch_metrics:
        save_per_epoch_metrics(
            train_losses=train_losses,
            val_losses=val_losses,
            filepath=args.filepath,
            model_name=model_name,
            rep=rep,
            sigma_noise=s,
            iteration=iteration,
            file_no=file_no
        )
    
    return metrics[3]

def train_mixup(
    x_train, y_train, x_test, y_test, x_val, y_val,
    args, s, rep, iteration, iteration_seed, file_no, y_test_original,
    trial=None
, train_noise=None):
    """
    Mixup for Molecular Regression (Zhang et al. 2018).
    
    Trains on linear interpolations of molecular representations:
        x_mixed = λ*x_i + (1-λ)*x_j
        y_mixed = λ*y_i + (1-λ)*y_j
    
    Modes:
    - 'input': Mix at input level (for fingerprints/descriptors)
    - 'manifold': Mix at hidden layer (Manifold Mixup)
    - 'uncertainty_aware': Use model uncertainty to weight mixing
    """
    from loss_functions import get_loss_function
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    loss_name = args.loss if hasattr(args, 'loss') else 'mse'
    
    # Hyperparameters
    if trial is not None:
        hidden_size1 = trial.suggest_categorical('hidden_size1', [64, 128, 256])
        hidden_size2 = trial.suggest_categorical('hidden_size2', [32, 64, 128])
        alpha = trial.suggest_float('alpha', 0.1, 2.0)  # Beta distribution parameter
        mixup_mode = trial.suggest_categorical('mixup_mode', 
                                               ['input', 'manifold', 'uncertainty_aware'])
        mix_prob = trial.suggest_float('mix_prob', 0.5, 1.0)
    else:
        hidden_size1, hidden_size2 = 128, 64
        alpha = 1.0  # α=1 gives uniform mixing, α>1 prefers edges, α<1 prefers extremes
        mixup_mode = 'input'
        mix_prob = 0.8  # Probability of applying mixup to a batch
    
    # Loss setup
    loss_kwargs = {}
    if hasattr(args, 'loss_params') and args.loss_params:
        import json
        loss_kwargs = json.loads(args.loss_params)
    
    output_size_map = {
        'heteroscedastic': 2, 'evidential': 4, 'evidential_cauchy': 4,
        'evidential_laplace': 4, 'sample_adaptive_barron': 2, 'stratified': 2,
    }
    output_size = output_size_map.get(loss_name, 1)
    
    # Prepare data
    x_train_t = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    x_val_t = torch.tensor(x_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
    x_test_t = torch.tensor(x_test, dtype=torch.float32).to(device)
    
    train_dataset = TensorDataset(x_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=NEURAL_DEFAULTS['training']['batch_size'], shuffle=True)
    
    # Create model
    if mixup_mode == 'manifold':
        # For manifold mixup, we need access to intermediate layers
        class ManifoldMixupModel(nn.Module):
            def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
                super().__init__()
                self.fc1 = nn.Linear(input_size, hidden_size1)
                self.fc2 = nn.Linear(hidden_size1, hidden_size2)
                self.fc3 = nn.Linear(hidden_size2, output_size)
                self.activation = nn.ReLU()
                self.dropout = nn.Dropout(0.2)
            
            def forward(self, x, mixup_layer=None, mixup_lambda=None, mixup_index=None):
                """
                Forward with optional mixup at specified layer
                mixup_layer: None, 0, 1, or 2 (which layer to mix at)
                """
                # Layer 0: after fc1
                x = self.activation(self.fc1(x))
                
                if mixup_layer == 0 and mixup_lambda is not None:
                    x = mixup_lambda * x + (1 - mixup_lambda) * x[mixup_index]
                
                x = self.dropout(x)
                
                # Layer 1: after fc2
                x = self.activation(self.fc2(x))
                
                if mixup_layer == 1 and mixup_lambda is not None:
                    x = mixup_lambda * x + (1 - mixup_lambda) * x[mixup_index]
                
                x = self.dropout(x)
                
                # Output
                x = self.fc3(x)
                return x
        
        model = ManifoldMixupModel(
            input_size=x_train.shape[1],
            hidden_size1=hidden_size1,
            hidden_size2=hidden_size2,
            output_size=output_size
        ).to(device)
    else:
        model = DNNRegressionModel(
            input_size=x_train.shape[1],
            hidden_size1=hidden_size1,
            hidden_size2=hidden_size2
        ).to(device)
        if output_size > 1:
            model.fc3 = nn.Linear(hidden_size2, output_size)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=NEURAL_DEFAULTS['training']['lr'])
    criterion = get_loss_function(loss_name, **loss_kwargs)
    
    # For uncertainty-aware mixup, we need initial model
    if mixup_mode == 'uncertainty_aware':
        print("Pre-training for uncertainty estimates...")
        for epoch in range(20):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
    
    # Training with Mixup
    print(f"Training with Mixup (mode={mixup_mode}, alpha={alpha})...")
    
    train_losses = []
    val_losses = []
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        for X_batch, y_batch in train_loader:
            # Decide whether to apply mixup
            if np.random.rand() > mix_prob:
                # No mixup, regular training
                optimizer.zero_grad()
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                continue
            
            # Sample mixing coefficient
            if alpha > 0:
                lam = np.random.beta(alpha, alpha)
            else:
                lam = 1.0
            
            batch_size = X_batch.size(0)
            index = torch.randperm(batch_size).to(device)
            
            if mixup_mode == 'input':
                # Standard input mixup
                X_mixed = lam * X_batch + (1 - lam) * X_batch[index]
                y_mixed = lam * y_batch + (1 - lam) * y_batch[index]
                
                optimizer.zero_grad()
                pred = model(X_mixed)
                loss = criterion(pred, y_mixed)
            
            elif mixup_mode == 'manifold':
                # Manifold mixup: randomly choose which layer to mix at
                mixup_layer = np.random.choice([None, 0, 1])  # None = input, 0/1 = hidden layers
                
                if mixup_layer is None:
                    # Input mixup
                    X_mixed = lam * X_batch + (1 - lam) * X_batch[index]
                    y_mixed = lam * y_batch + (1 - lam) * y_batch[index]
                    
                    optimizer.zero_grad()
                    pred = model(X_mixed)
                    loss = criterion(pred, y_mixed)
                else:
                    # Hidden layer mixup
                    y_mixed = lam * y_batch + (1 - lam) * y_batch[index]
                    
                    optimizer.zero_grad()
                    pred = model(X_batch, mixup_layer=mixup_layer, 
                               mixup_lambda=lam, mixup_index=index)
                    loss = criterion(pred, y_mixed)
            
            elif mixup_mode == 'uncertainty_aware':
                # Use model uncertainty to weight mixing
                # High uncertainty samples get mixed more
                model.eval()
                with torch.no_grad():
                    pred_unc = model(X_batch)
                    
                    # Extract uncertainty
                    if loss_name == 'heteroscedastic':
                        log_var = pred_unc[:, 1]
                        uncertainty = torch.sqrt(torch.exp(log_var))
                    elif loss_name in ['evidential', 'evidential_cauchy', 'evidential_laplace']:
                        v = F.softplus(pred_unc[:, 1]) + 1.0
                        alpha_param = F.softplus(pred_unc[:, 2]) + 1.0
                        beta = F.softplus(pred_unc[:, 3])
                        epistemic = beta / torch.clamp(alpha_param - 1, min=1e-6)
                        aleatoric = beta / (v * torch.clamp(alpha_param - 1, min=1e-6))
                        uncertainty = torch.sqrt(epistemic + aleatoric)
                    else:
                        uncertainty = torch.ones(batch_size).to(device)
                
                model.train()
                
                # Adjust lambda based on uncertainty
                # High uncertainty pairs get more mixing (lambda closer to 0.5)
                # Low uncertainty pairs use original lambda
                uncertainty_norm = (uncertainty + uncertainty[index]) / 2
                uncertainty_norm = uncertainty_norm / (uncertainty_norm.max() + 1e-8)
                
                # Interpolate lambda towards 0.5 based on uncertainty
                lam_adjusted = lam + (0.5 - lam) * uncertainty_norm
                
                X_mixed = lam_adjusted.view(-1, 1) * X_batch + (1 - lam_adjusted.view(-1, 1)) * X_batch[index]
                y_mixed = lam_adjusted.view(-1, 1) * y_batch + (1 - lam_adjusted.view(-1, 1)) * y_batch[index]
                
                optimizer.zero_grad()
                pred = model(X_mixed)
                loss = criterion(pred, y_mixed)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation (no mixup)
        model.eval()
        with torch.no_grad():
            pred_val = model(x_val_t)
            val_loss = criterion(pred_val, y_val_t)
        val_losses.append(val_loss.item())
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: Train={avg_train_loss:.4f}, Val={val_loss.item():.4f}")
    
    # Test (no mixup)
    model.eval()
    with torch.no_grad():
        pred_test = model(x_test_t).cpu().numpy()
        if output_size > 1:
            y_pred = pred_test[:, 0]
        else:
            y_pred = pred_test.flatten()
    
    # Calculate metrics
    metrics = calculate_regression_metrics(y_test, y_pred, logging=True)
    
    # Save results
    model_name = f"mixup_{mixup_mode}"
    if loss_name != 'mse':
        model_name += f"_{loss_name}"
    
    save_results(args.filepath, s, iteration, model_name, rep,
                args.sample_size, metrics, 'default', loss_name)
    
    # Save per-epoch metrics
    if args.save_per_epoch_metrics:
        save_per_epoch_metrics(
            train_losses=train_losses,
            val_losses=val_losses,
            filepath=args.filepath,
            model_name=model_name,
            rep=rep,
            sigma_noise=s,
            iteration=iteration,
            file_no=file_no
        )
    
    return metrics[3]

def train_sam(
    x_train, y_train, x_test, y_test, x_val, y_val,
    args, s, rep, iteration, iteration_seed, file_no, y_test_original,
    trial=None
, train_noise=None):
    """
    Sharpness-Aware Minimization (Foret et al. 2021).
    
    Seeks parameters in flat minima by minimizing loss in a neighborhood:
        min_w max_{||ε||≤ρ} L(w + ε)
    
    This improves robustness to noisy labels by avoiding sharp minima
    that overfit to individual noisy samples.
    
    Can be combined with any loss function.
    """
    from loss_functions import get_loss_function
    import torch.nn.functional as F
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    loss_name = args.loss if hasattr(args, 'loss') else 'mse'
    
    # Hyperparameters
    if trial is not None:
        hidden_size1 = trial.suggest_categorical('hidden_size1', [64, 128, 256])
        hidden_size2 = trial.suggest_categorical('hidden_size2', [32, 64, 128])
        rho = trial.suggest_float('rho', 0.01, 0.2)  # Neighborhood size
        adaptive = trial.suggest_categorical('adaptive', [True, False])
    else:
        hidden_size1, hidden_size2 = 128, 64
        rho = 0.05  # Standard value from paper
        adaptive = True  # Adaptive SAM (better for varied scales)
    
    # Loss setup
    loss_kwargs = {}
    if hasattr(args, 'loss_params') and args.loss_params:
        import json
        loss_kwargs = json.loads(args.loss_params)
    
    output_size_map = {
        'heteroscedastic': 2, 'evidential': 4, 'evidential_cauchy': 4,
        'evidential_laplace': 4, 'sample_adaptive_barron': 2, 'stratified': 2,
    }
    output_size = output_size_map.get(loss_name, 1)
    
    # Prepare data
    x_train_t = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    x_val_t = torch.tensor(x_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
    x_test_t = torch.tensor(x_test, dtype=torch.float32).to(device)
    
    train_dataset = TensorDataset(x_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=NEURAL_DEFAULTS['training']['batch_size'], shuffle=True)
    
    # Create model
    model = DNNRegressionModel(
        input_size=x_train.shape[1],
        hidden_size1=hidden_size1,
        hidden_size2=hidden_size2
    ).to(device)
    if output_size > 1:
        model.fc3 = nn.Linear(hidden_size2, output_size)
    
    criterion = get_loss_function(loss_name, **loss_kwargs)
    
    # SAM optimizer
    class SAM(torch.optim.Optimizer):
        """Sharpness-Aware Minimization optimizer"""
        def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
            assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
            
            defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
            super(SAM, self).__init__(params, defaults)
            
            self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
            self.param_groups = self.base_optimizer.param_groups
            self.defaults.update(self.base_optimizer.defaults)
        
        @torch.no_grad()
        def first_step(self, zero_grad=False):
            """
            First step: compute and apply adversarial perturbation
            """
            grad_norm = self._grad_norm()
            for group in self.param_groups:
                scale = group["rho"] / (grad_norm + 1e-12)
                
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    
                    # Save current parameters
                    self.state[p]["old_p"] = p.data.clone()
                    
                    # Compute perturbation
                    if group["adaptive"]:
                        # Adaptive SAM: scale by parameter magnitude
                        e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                    else:
                        e_w = p.grad * scale.to(p)
                    
                    # Apply perturbation
                    p.add_(e_w)
            
            if zero_grad:
                self.zero_grad()
        
        @torch.no_grad()
        def second_step(self, zero_grad=False):
            """
            Second step: restore parameters and apply gradient from perturbed point
            """
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    
                    # Restore original parameters
                    p.data = self.state[p]["old_p"]
            
            # Update with base optimizer
            self.base_optimizer.step()
            
            if zero_grad:
                self.zero_grad()
        
        def step(self, closure=None):
            """Not used in SAM - use first_step and second_step instead"""
            raise NotImplementedError("SAM doesn't work like standard optimizers. Use first_step and second_step.")
        
        def _grad_norm(self):
            """Compute gradient norm"""
            shared_device = self.param_groups[0]["params"][0].device
            norm = torch.norm(
                torch.stack([
                    ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                    for group in self.param_groups for p in group["params"]
                    if p.grad is not None
                ]),
                p=2
            )
            return norm
        
        def load_state_dict(self, state_dict):
            super().load_state_dict(state_dict)
            self.base_optimizer.param_groups = self.param_groups
    
    # Create SAM optimizer
    optimizer = SAM(
        model.parameters(),
        base_optimizer=torch.optim.Adam,
        rho=rho,
        adaptive=adaptive,
        lr=0.001
    )
    
    print(f"Training with SAM (rho={rho}, adaptive={adaptive})...")
    
    train_losses = []
    val_losses = []
    sharpness_values = []
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        epoch_sharpness = 0
        
        for X_batch, y_batch in train_loader:
            # First forward-backward pass (compute gradient)
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            
            # First step: apply perturbation
            optimizer.first_step(zero_grad=True)
            
            # Second forward-backward pass at perturbed point
            pred_perturbed = model(X_batch)
            loss_perturbed = criterion(pred_perturbed, y_batch)
            
            # Track sharpness (difference between perturbed and original loss)
            sharpness = (loss_perturbed - loss).item()
            epoch_sharpness += sharpness
            
            loss_perturbed.backward()
            
            # Second step: restore parameters and update with gradient from perturbed point
            optimizer.second_step(zero_grad=True)
            
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        avg_sharpness = epoch_sharpness / len(train_loader)
        
        train_losses.append(avg_train_loss)
        sharpness_values.append(avg_sharpness)
        
        # Validation
        model.eval()
        with torch.no_grad():
            pred_val = model(x_val_t)
            val_loss = criterion(pred_val, y_val_t)
        val_losses.append(val_loss.item())
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: Train={avg_train_loss:.4f}, Val={val_loss.item():.4f}, "
                  f"Sharpness={avg_sharpness:.4f}")
    
    # Test
    model.eval()
    with torch.no_grad():
        pred_test = model(x_test_t).cpu().numpy()
        if output_size > 1:
            y_pred = pred_test[:, 0]
        else:
            y_pred = pred_test.flatten()
    
    # Calculate metrics
    metrics = calculate_regression_metrics(y_test, y_pred, logging=True)
    
    # Save results
    model_name = f"sam_{'adaptive' if adaptive else 'standard'}"
    if loss_name != 'mse':
        model_name += f"_{loss_name}"
    
    save_results(args.filepath, s, iteration, model_name, rep,
                args.sample_size, metrics, 'default', loss_name)
    
    # Save sharpness tracking
    if args.uncertainty:
        import pandas as pd
        import os
        
        sharpness_dir = os.path.dirname(args.filepath.replace('.csv', '_sharpness/'))
        os.makedirs(sharpness_dir, exist_ok=True)
        
        sharpness_df = pd.DataFrame({
            'epoch': np.arange(len(sharpness_values)),
            'train_loss': train_losses,
            'val_loss': val_losses,
            'sharpness': sharpness_values,
        })
        
        sharpness_file = os.path.join(
            sharpness_dir,
            f"sharpness_{model_name}_{rep}_sigma{s}_iter{iteration}_file{file_no}.csv"
        )
        sharpness_df.to_csv(sharpness_file, index=False)
        
        print(f"Saved sharpness tracking to {sharpness_file}")
        print(f"  Initial sharpness: {sharpness_values[0]:.4f}")
        print(f"  Final sharpness: {sharpness_values[-1]:.4f}")
        print(f"  Avg sharpness reduction: {(sharpness_values[0] - sharpness_values[-1]):.4f}")
    
    # Save per-epoch metrics
    if args.save_per_epoch_metrics:
        save_per_epoch_metrics(
            train_losses=train_losses,
            val_losses=val_losses,
            filepath=args.filepath,
            model_name=model_name,
            rep=rep,
            sigma_noise=s,
            iteration=iteration,
            file_no=file_no
        )
    
    return metrics[3]