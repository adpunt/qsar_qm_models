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
from torch_geometric.loader import DataLoader
from torch_geometric.typing import Adj, OptTensor, PairTensor, Size
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
# from torch_sparse import SparseTensor
from torch_geometric.nn.inits import glorot, zeros
import gpytorch
from typing import Union
from sklearn.ensemble import RandomForestRegressor
from quantile_forest import RandomForestQuantileRegressor
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.svm import SVR, SVC
from xgboost import XGBRegressor
from torch.nn.utils import parameters_to_vector as Params2Vec, vector_to_parameters as Vec2Params
import matplotlib.pyplot as plt
import torchbnn as bnn
from torchhk import transform_model, transform_layer
import lightgbm as lgb
from botorch import fit_gpytorch_model
import gauche
from gauche.kernels.fingerprint_kernels import *
from gauche.kernels.graph_kernels import *
from gauche import SIGP, NonTensorialInputs
from gauche.dataloader import MolPropLoader
from gauche.dataloader.data_utils import transform_data
from gauche.kernels.graph_kernels import WeisfeilerLehmanKernel, VertexHistogramKernel
import torchcp
from torchcp.regression.predictor import SplitPredictor, ACIPredictor
from torchcp.classification.score import APS
from torchcp.regression.score import ABS
from sklearn.isotonic import IsotonicRegression

from utils import * 
from loss_functions import *

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

class GCN(torch.nn.Module):
    """Graph Convolutional Network class with 3 convolutional layers and a linear layer"""

    def __init__(self, dim_h, dropout_rate=0.5):
        """init method for GCN

        Args:
            dim_h (int): the dimension of hidden layers
        """
        super().__init__()
        self.conv1 = GCNConv(11, dim_h)
        self.conv2 = GCNConv(dim_h, dim_h)
        self.conv3 = GCNConv(dim_h, dim_h)
        self.lin = torch.nn.Linear(dim_h, 1)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, data):
        e = data.edge_index
        x = data.x

        x = self.conv1(x, e)
        x = x.relu()
        x = self.conv2(x, e)
        x = x.relu()
        x = self.conv3(x, e)
        x = global_mean_pool(x, data.batch)

        x = self.dropout(x)
        # x = Fun.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x

class GIN(torch.nn.Module):
    """Graph Isomorphism Network class with 3 GINConv layers and 2 linear layers"""

    def __init__(self, dim_h, dropout_rate=0.5):
        """Initializing GIN class

        Args:
            dim_h (int): the dimension of hidden layers
        """
        super(GIN, self).__init__()
        self.conv1 = GINConv(
            Sequential(Linear(11, dim_h), BatchNorm1d(dim_h), ReLU(), Linear(dim_h, dim_h), ReLU())
        )
        self.conv2 = GINConv(
            Sequential(
                Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(), Linear(dim_h, dim_h), ReLU()
            )
        )
        self.conv3 = GINConv(
            Sequential(
                Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(), Linear(dim_h, dim_h), ReLU()
            )
        )
        self.lin1 = Linear(dim_h, dim_h)
        self.lin2 = Linear(dim_h, 1)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        # Node embeddings
        h = self.conv1(x, edge_index)
        h = h.relu()
        h = self.conv2(h, edge_index)
        h = h.relu()
        h = self.conv3(h, edge_index)

        # Graph-level readout
        h = global_add_pool(h, batch)

        h = self.lin1(h)
        h = h.relu()
        h = self.dropout(h)
        # h = Fun.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)

        return h

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
            "prior_mu": 0, 
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
            "prior_mu": 0,
            "prior_sigma": 0.1,
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

def apply_bayesian_transformation_last_layer_variational(model):
    """
    Converts the last Linear layer of a PyTorch model to a Bayesian Linear layer
    (VBLL - Variational Bayesian Last Layer) while keeping the rest of the model deterministic.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to be transformed.

    Returns
    -------
    model : nn.Module
        The transformed model with the last layer replaced by a Bayesian layer.
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

    # Transform using torchhk-style util
    bayesian_layer = transform_layer(
        last_linear_module,
        nn.Linear,
        bnn.BayesLinear,
        args={
            "prior_mu": 0,
            "prior_sigma": 0.1,
            "in_features": ".in_features",
            "out_features": ".out_features",
            "bias": ".bias"
        },
        attrs={"weight_mu": ".weight"}
    )

    # Helper for recursive attribute setting
    def set_nested_attr(obj, attr_path, value):
        attrs = attr_path.split(".")
        for a in attrs[:-1]:
            obj = getattr(obj, a)
        setattr(obj, attrs[-1], value)

    # Replace in the model
    set_nested_attr(model, last_linear_name, bayesian_layer)

    return model

def train_rf_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, model_type, file_no, y_test_original, trial=None):
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
            params['max_depth'] = None
            params['max_features'] = 'sqrt'
            params['min_samples_leaf'] = 1
            params['min_samples_split'] = 2
            params['n_estimators'] = 300 if model_type == 'qrf' else 100
            params['bootstrap'] = True
            params_source = 'default'

    if model_type == 'rf':
        model = RandomForestRegressor(random_state=iteration_seed, **params)
    elif model_type == 'qrf':
        quantile = trial.suggest_float('quantile', 0.1, 0.9) if args.tuning else 0.5
        model = RandomForestQuantileRegressor(random_state=iteration_seed, **params)
        if trial is not None:
            trial.set_user_attr("quantile", quantile)

    x_train = np.vstack((x_train, x_val))
    y_train = np.hstack((y_train, y_val))

    model.fit(x_train, y_train)



    if model_type == 'qrf':
        q16, q50, q84 = model.predict(x_test, quantiles=[0.16, 0.5, 0.84]).T
        y_pred = q50
        y_pred_mean = q50
        std_est = (q84 - q16) / 2
        
        if args.uncertainty:
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
            )
    else:
        y_pred = model.predict(x_test)

    metrics = calculate_regression_metrics(y_test, y_pred, logging=True)

    save_results(args.filepath, s, iteration, model_type, rep, args.sample_size, metrics, params_source)

    return metrics[3]

def train_svm_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, trial=None):
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
            params['C'] = 1.0
            params['gamma'] = 'scale'
            params['kernel'] = 'rbf'
            params_source = 'default'

    x_train = np.vstack((x_train, x_val))
    y_train = np.hstack((y_train, y_val))

    model = SVR(**params)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    metrics = calculate_regression_metrics(y_test, y_pred, logging=True)

    save_results(args.filepath, s, iteration, 'svm', rep, args.sample_size, metrics)

    return metrics[3]

def train_ngboost_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, file_no, y_test_original, trial=None):
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
            params['learning_rate'] = 0.01
            params['n_estimators'] = 500
            params['natural_gradient'] = True
            params_source = 'default'
    
    # STEP 1: Split validation for calibration
    if args.uncertainty:
        split_idx = len(x_val) // 2
        x_val_train = np.vstack((x_train, x_val[:split_idx]))
        y_val_train = np.hstack((y_train, y_val[:split_idx]))
        x_val_cal = x_val[split_idx:]
        y_val_cal = y_val[split_idx:]
    else:
        x_val_train = np.vstack((x_train, x_val))
        y_val_train = np.hstack((y_train, y_val))
    
    model = NGBRegressor(
        Dist=Normal,
        Score=MLE,
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
    else:
        temperature = None
        y_pred_std_calibrated = None
    
    metrics = calculate_regression_metrics(y_test, y_pred, logging=True)
    save_results(args.filepath, s, iteration, 'ngboost', rep, args.sample_size, metrics, params_source)
    
    # STEP 3: Save with calibration
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
            temperature=temperature
        )
    
    return metrics[3]

def train_xgboost_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, trial=None):
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
            params['max_depth'] = 6
            params['learning_rate'] = 0.1
            params['subsample'] = 1.0
            params['n_estimators'] = 100
            params['colsample_bytree'] = 1.0
            params['colsample_bylevel'] = 1.0
            params['min_child_weight'] = 1
            params['gamma'] = 0.0
            params['reg_alpha'] = 0.0
            params['reg_lambda'] = 1.0
            params_source = 'default'

    if x_val is not None and y_val is not None:
        x_train = np.vstack((x_train, x_val))
        y_train = np.hstack((y_train, y_val))

    model = XGBRegressor(random_state=iteration_seed, **params)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    metrics = calculate_regression_metrics(y_test, y_pred, logging=True)

    save_results(args.filepath, s, iteration, 'xgboost', rep, args.sample_size, metrics)

    return metrics[3]

def train_gauche_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, file_no, y_test_original, trial=None):
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
            params['kernel_name'] = 'Tanimoto'
            params['outputscale'] = 1.0
            params['likelihood_noise'] = 1e-3
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
    }

    # STEP 1: Split validation for calibration
    if args.uncertainty:
        split_idx = len(x_val) // 2
        x_train_full = np.vstack((x_train, x_val[:split_idx]))
        y_train_full = np.hstack((y_train, y_val[:split_idx]))
        x_val_cal = x_val[split_idx:]
        y_val_cal = y_val[split_idx:]
    else:
        x_train_full = np.vstack((x_train, x_val))
        y_train_full = np.hstack((y_train, y_val))

    x_train_tensor = torch.from_numpy(x_train_full).double()
    x_test_tensor = torch.from_numpy(x_test).double()
    y_train_tensor = torch.from_numpy(y_train_full).double()

    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise=params['likelihood_noise'])
    kernel_class = kernel_map[params['kernel_name']]
    model = Gauche(x_train_tensor, y_train_tensor, likelihood, kernel_class)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    fit_gpytorch_model(mll)

    model.eval()
    likelihood.eval()
    
    # STEP 2: Get predictions and calibrate
    with torch.no_grad():
        # Test predictions
        test_preds = model(x_test_tensor)
        y_pred = test_preds.mean.numpy()
        pred_vars = test_preds.variance.numpy()
        y_pred_std_uncalibrated = np.sqrt(pred_vars)
        
        if args.uncertainty:
            # Calibration predictions
            x_val_cal_tensor = torch.from_numpy(x_val_cal).double()
            cal_preds = model(x_val_cal_tensor)
            y_cal_pred_mean = cal_preds.mean.numpy()
            y_cal_pred_std = np.sqrt(cal_preds.variance.numpy())
            
            # Find temperature
            temperature = calibrate_uncertainty_simple(y_cal_pred_mean, y_cal_pred_std, y_val_cal)
            y_pred_std_calibrated = y_pred_std_uncalibrated * temperature
        else:
            temperature = None
            y_pred_std_calibrated = None

    metrics = calculate_regression_metrics(y_test, y_pred, logging=True)
    save_results(args.filepath, s, iteration, "gauche", rep, args.sample_size, metrics)

    # STEP 3: Save with calibration
    if args.uncertainty:
        save_uncertainty_values(
            y_pred_mean=y_pred,
            y_pred_std=y_pred_std_uncalibrated,
            y_true_original=y_test_original,
            y_true_noisy=y_test,
            filepath=args.filepath,
            model_name="gauche",
            rep=rep,
            sigma_noise=s,
            iteration=iteration,
            file_no=file_no,
            y_pred_std_calibrated=y_pred_std_calibrated,
            temperature=temperature
        )

    return metrics[3]

def train_nn(model, train_loader, val_loader, criterion, optimizer, device, args, s, iteration, file_no, model_name, rep, 
             patience=20, tolerance=0.01, domain_labels_train=None, domain_labels_val=None):
    """
    Added domain_labels parameters for domain-aware losses
    """
    model.to(device)
    best_loss = float('inf')
    epochs_no_improve = 0
    
    train_losses = []
    val_losses = []
    
    # Check if loss needs domain labels
    needs_domains = isinstance(criterion, (DomainWeightedLoss, DomainBalancedLoss, HeteroscedasticPerDomainLoss))

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
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
            batch_idx += len(X_batch)

        # Validation
        model.eval()
        val_loss = 0
        batch_idx = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                val_outputs = model(X_val)
                
                if needs_domains and domain_labels_val is not None:
                    batch_domains = domain_labels_val[batch_idx:batch_idx+len(X_val)]
                    batch_domains = torch.tensor(batch_domains, dtype=torch.long).to(device)
                    loss = criterion(val_outputs, y_val, batch_domains)
                else:
                    loss = criterion(val_outputs, y_val)
                
                val_loss += loss.item()
                batch_idx += len(X_val)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Early stopping check
        if val_loss < best_loss - tolerance:
            best_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
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
                   domain_labels_train=None, domain_labels_val=None, domain_labels_test=None):
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
            params['hidden_size1'], params['hidden_size2'] = 128, 64
            params['activation'] = 'relu'
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

    val_loader = TorchDataLoader(TensorDataset(x_val_tensor, y_val_tensor), batch_size=32, shuffle=False)
    train_loader = TorchDataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=32, shuffle=True)

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
    elif args.bayesian_transformation == "last_layer":
        model = apply_bayesian_transformation_last_layer(model)
        model_name = "bnn_last"
    elif args.bayesian_transformation == "variational":
        model = apply_bayesian_transformation_last_layer_variational(model)
        model_name = "bnn_variational"

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # STEP 5: Train with appropriate domain labels
    train_nn(model, train_loader, val_loader, criterion, optimizer, device, args, s, iteration, file_no, model_name, rep,
             domain_labels_train=domain_labels_train if needs_domains else None, 
             domain_labels_val=domain_labels_val_train if needs_domains else None)
    
    model.eval()

    if is_bayesian:
        torch.manual_seed(iteration_seed)
        np.random.seed(iteration_seed)
        
        num_samples = 100

        # Get calibration predictions
        x_val_cal_tensor = torch.tensor(x_val_cal, dtype=torch.float32).to(device)
        preds_cal = []
        with torch.no_grad():
            for _ in range(num_samples):
                output = model(x_val_cal_tensor).cpu().numpy()
                # For heteroscedastic, only use mean predictions
                if loss_name == 'heteroscedastic':
                    output = output[:, 0:1]
                preds_cal.append(output)
        
        preds_cal = np.stack(preds_cal, axis=0)
        y_cal_pred_mean = preds_cal.mean(axis=0).flatten()
        y_cal_pred_std = preds_cal.std(axis=0).flatten()
        
        # Find optimal temperature
        temperature = calibrate_uncertainty_simple(y_cal_pred_mean, y_cal_pred_std, y_val_cal)
        
        # Get test predictions
        preds = []
        with torch.no_grad():
            for _ in range(num_samples):
                output = model(x_test_tensor).cpu().numpy()
                # For heteroscedastic, only use mean predictions
                if loss_name == 'heteroscedastic':
                    output = output[:, 0:1]
                preds.append(output)

        preds = np.stack(preds, axis=0)
        y_pred_mean = preds.mean(axis=0).flatten()
        y_pred_std_uncalibrated = preds.std(axis=0).flatten()
        y_pred_std_calibrated = y_pred_std_uncalibrated * temperature
        
        y_pred = y_pred_mean

    else:
        # Non-Bayesian prediction
        with torch.no_grad():
            y_pred_tensor = model(x_test_tensor).cpu().numpy()
            # For heteroscedastic, extract mean predictions
            if loss_name == 'heteroscedastic':
                y_pred = y_pred_tensor[:, 0].flatten()
            else:
                y_pred = y_pred_tensor.flatten()
        
        y_pred_std_uncalibrated = None
        y_pred_std_calibrated = None
        temperature = None

    # Calculate metrics normally
    metrics = calculate_regression_metrics(y_test, y_pred, logging=True)
    
    # Create full model name with loss function
    full_model_name = f"{model_name}_{loss_name}" if loss_name != 'mse' else model_name
    
    save_results(args.filepath, s, iteration, full_model_name, rep, args.sample_size, metrics, params_source, loss_name)

    # STEP 7: Save uncertainty with calibration
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

def train_flexible_dnn_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, file_no, y_test_original, trial=None,
                            domain_labels_train=None, domain_labels_val=None, domain_labels_test=None):
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
            params['hidden_sizes'] = [128, 64]
            params['activation'] = 'relu'
            params_source = 'default'
    
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
    
    train_loader = TorchDataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
    val_loader = TorchDataLoader(TensorDataset(x_val_tensor, y_val_tensor), batch_size=32, shuffle=False)
    
    if loss_name == 'heteroscedastic':
        model = FlexibleDNNRegressionModel(input_size=x_train.shape[1], hidden_sizes=params['hidden_sizes'], activation_fn=activation).to(device)
        model.network[-1] = nn.Linear(params['hidden_sizes'][-1], 2)
    elif loss_name == 'evidential':
        model = FlexibleDNNRegressionModel(input_size=x_train.shape[1], hidden_sizes=params['hidden_sizes'], activation_fn=activation).to(device)
        model.network[-1] = nn.Linear(params['hidden_sizes'][-1], 4)
    else:
        model = FlexibleDNNRegressionModel(input_size=x_train.shape[1], hidden_sizes=params['hidden_sizes'], activation_fn=activation).to(device)
    
    model_name = "flexible_dnn"
    if args.bayesian_transformation == "full":
        model = apply_bayesian_transformation(model)
        model_name = "flexible_bnn_full"
    elif args.bayesian_transformation == "last_layer":
        model = apply_bayesian_transformation_last_layer(model)
        model_name = "flexible_bnn_last"
    elif args.bayesian_transformation == "variational":
        model = apply_bayesian_transformation_last_layer_variational(model)
        model_name = "flexible_bnn_variational"
    
    from loss_functions import get_loss_function
    criterion = get_loss_function(loss_name, **loss_kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_nn(model, train_loader, val_loader, criterion, optimizer, device, args, s, iteration, file_no, model_name, rep,
             domain_labels_train=domain_labels_train if needs_domains else None,
             domain_labels_val=domain_labels_val_train if needs_domains else None)
    
    model.eval()
    
    if is_bayesian:
        torch.manual_seed(iteration_seed)
        np.random.seed(iteration_seed)
        
        num_samples = 100
        
        if args.uncertainty:
            x_val_cal_tensor = torch.tensor(x_val_cal, dtype=torch.float32).to(device)
            preds_cal = []
            with torch.no_grad():
                for _ in range(num_samples):
                    output = model(x_val_cal_tensor).cpu().numpy()
                    if loss_name == 'heteroscedastic':
                        output = output[:, 0:1]
                    elif loss_name == 'evidential':
                        output = output[:, 0:1]
                    preds_cal.append(output)
            
            preds_cal = np.stack(preds_cal, axis=0)
            y_cal_pred_mean = preds_cal.mean(axis=0).flatten()
            y_cal_pred_std = preds_cal.std(axis=0).flatten()
            
            temperature = calibrate_uncertainty_simple(y_cal_pred_mean, y_cal_pred_std, y_val_cal)
        
        preds = []
        with torch.no_grad():
            for _ in range(num_samples):
                output = model(x_test_tensor).cpu().numpy()
                if loss_name == 'heteroscedastic':
                    output = output[:, 0:1]
                elif loss_name == 'evidential':
                    output = output[:, 0:1]
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

def train_mlp_variant_model(x_train, y_train, x_test, y_test, x_val, y_val, model_type, args, s, rep, iteration, iteration_seed, file_no, y_test_original, trial=None):
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
            params['hidden_size'], params['num_hidden_layers'], params['dropout_rate'], params['lr'] = 128, 2, 0.2, 0.001
            params_source = 'default'

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
        val_loader = TorchDataLoader(TensorDataset(x_val_tensor, y_val_tensor), batch_size=32, shuffle=False)
    elif x_val is not None and y_val is not None:
        x_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
        val_loader = TorchDataLoader(TensorDataset(x_val_tensor, y_val_tensor), batch_size=32, shuffle=False)
    else:
        val_loader = None

    train_loader = TorchDataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=32, shuffle=True)

    if model_type == "mlp":
        model = MLPRegressor(input_size=x_train.shape[1], hidden_size=params['hidden_size'],
                             num_hidden_layers=params['num_hidden_layers'], dropout_rate=params['dropout_rate']) 
        criterion = nn.MSELoss()

    elif model_type == "residual_mlp":
        model = ResidualMLP(input_size=x_train.shape[1], hidden_size=128, num_layers=3)
        criterion = nn.MSELoss()

    elif model_type == "factorization_mlp":
        model = FactorizationMLP(input_size=x_train.shape[1], hidden_size=128, factor_size=16)
        criterion = nn.MSELoss()

    elif model_type == "mtl":
        model = MTLRegressionModel(input_size=x_train.shape[1], hidden_size=128, num_tasks=1)
        criterion = nn.MSELoss()

    if args.bayesian_transformation == "full":
        model = apply_bayesian_transformation(model)
        model_name = f"{model_type}_bnn_full"
    elif args.bayesian_transformation == "last_layer":
        model = apply_bayesian_transformation_last_layer(model)
        model_name = f"{model_type}_bnn_last"
    elif args.bayesian_transformation == "variational":
        model = apply_bayesian_transformation_last_layer_variational(model)
        model_name = f"{model_type}_bnn_variational"
    else:
        model_name = model_type

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    train_nn(model, train_loader, val_loader, criterion, optimizer, device, args, s, iteration, file_no, model_name, rep)
    model.eval()

    # STEP 2: Get predictions and calibrate if Bayesian
    if is_bayesian:
        torch.manual_seed(iteration_seed)
        np.random.seed(iteration_seed)
        
        num_samples = 100
        
        if args.uncertainty:
            # Get calibration predictions
            x_val_cal_tensor = torch.tensor(x_val_cal, dtype=torch.float32).to(device)
            preds_cal = []
            with torch.no_grad():
                for _ in range(num_samples):
                    preds_cal.append(model(x_val_cal_tensor).cpu().numpy())
            
            preds_cal = np.stack(preds_cal, axis=0)
            y_cal_pred_mean = preds_cal.mean(axis=0).flatten()
            y_cal_pred_std = preds_cal.std(axis=0).flatten()
            
            temperature = calibrate_uncertainty_simple(y_cal_pred_mean, y_cal_pred_std, y_val_cal)
        
        # Get test predictions
        preds = []
        with torch.no_grad():
            for _ in range(num_samples):
                preds.append(model(x_test_tensor).cpu().numpy())
        
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
        y_pred = y_pred_tensor.flatten()
        y_pred_std_uncalibrated = None
        y_pred_std_calibrated = None
        temperature = None
     
    metrics = calculate_regression_metrics(y_test, y_pred, logging=True)
    save_results(args.filepath, s, iteration, model_name, rep, args.sample_size, metrics)

    # STEP 3: Save uncertainty with calibration
    if args.uncertainty and is_bayesian:
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
            temperature=temperature
        )
        
    return metrics[3]

def train_rnn_variant_model(x_train, y_train, x_test, y_test, x_val, y_val, model_type, args, s, rep, iteration, iteration_seed, file_no, trial=None):
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
        val_loader = TorchDataLoader(TensorDataset(x_val_tensor, y_val_tensor), batch_size=32, shuffle=False)
    else:
        val_loader = None

    train_loader = TorchDataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
    test_loader = TorchDataLoader(TensorDataset(x_test_tensor, y_test_tensor), batch_size=32, shuffle=False)

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

    save_results(args.filepath, s, iteration, model_type, rep, args.sample_size, metrics)

    return metrics[3]

def train_gnn(model_type, train_loader, test_loader, val_loader, args, s, iteration, file_no, y_test_original, trial=None, 
              y_train_noisy=None, y_test_noisy=None, y_val_noisy=None):
    """
    Note: y_train_noisy, y_test_noisy, y_val_noisy are the noisy+normalized targets from Rust.
    These should be used instead of batch.y from the dataloaders.
    """
    # Hyperparameter suggestions
    if trial is not None:
        dim_h = trial.suggest_int('dim_h', 32, 256, step=32)
        epochs = trial.suggest_int('epochs', 50, 300, step=50)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    else:
        dim_h = 64 if model_type in ["gin", "gin2d", "ginct"] else 128
        epochs = args.epochs
        learning_rate = 0.001
    
    if model_type == "gin" or model_type == "gin2d":
        model = GIN(dim_h=dim_h)
    elif model_type == "gcn":
        model = GCN(dim_h=dim_h)
    elif model_type == "ginct":
        model = GINCoTeaching(dim_h=dim_h)

    print(f"model: {model} and model_type: {model_type}")

    model_name = model_type
    is_bayesian = args.bayesian_transformation is not None
    
    if args.bayesian_transformation == "full":
        model = apply_bayesian_transformation(model)
        model_name = f"{model_type}_bnn_full"
    elif args.bayesian_transformation == "last_layer":
        model = apply_bayesian_transformation_last_layer(model)
        model_name = f"{model_type}_bnn_last"
    elif args.bayesian_transformation == "variational":
        model = apply_bayesian_transformation_last_layer_variational(model)
        model_name = f"{model_type}_bnn_variational"

    print(f"model: {model} and model_type: {model_type}")

    model.to(device)
    
    # STEP 1: Split val loader for calibration if Bayesian
    if is_bayesian and args.uncertainty:
        # Convert val_loader to list and split
        val_data_list = []
        for data in val_loader:
            val_data_list.extend([d for d in data.to_data_list()])
        
        split_idx = len(val_data_list) // 2
        val_train_list = val_data_list[:split_idx]
        val_cal_list = val_data_list[split_idx:]
        
        from torch_geometric.loader import DataLoader as GeometricDataLoader
        val_loader_train = GeometricDataLoader(val_train_list, batch_size=64, shuffle=False)
        val_loader_cal = GeometricDataLoader(val_cal_list, batch_size=64, shuffle=False)
        
        # Split y_val_noisy accordingly
        y_val_noisy_train = y_val_noisy[:split_idx] if y_val_noisy is not None else None
        y_val_noisy_cal = y_val_noisy[split_idx:] if y_val_noisy is not None else None
    else:
        val_loader_train = val_loader
        y_val_noisy_train = y_val_noisy
    
    if model_type != "ginct":
        train_loss, val_loss, train_target, train_y_target, trained_model = train_epochs(
            epochs, model, train_loader, val_loader_train, args, s, iteration, file_no, model_name,
            y_train_noisy=y_train_noisy, y_val_noisy=y_val_noisy_train, learning_rate=learning_rate
        )
        test_loss, test_target, test_y = testing(test_loader, trained_model, y_test_noisy=y_test_noisy)
    
    logging_flag = args.distribution not in ["domain_mpnn", "domain_tanimoto"]
    if not logging_flag:
        calculate_domain_metrics(test_target, test_y, domain_labels_subset, target_domain)
    metrics = calculate_regression_metrics(test_target, test_y, logging=logging_flag)
    print(f"model: {model_type}")
    print("rep: graph")

    # STEP 2: Get predictions and calibrate if Bayesian
    if is_bayesian and args.uncertainty:
        torch.manual_seed(args.random_seed)  # Use appropriate seed
        np.random.seed(args.random_seed)
        
        trained_model.eval()
        num_samples = 100
        
        # Get calibration predictions
        all_cal_preds = []
        with torch.no_grad():
            for _ in range(num_samples):
                batch_preds = []
                for data in val_loader_cal:
                    data = data.to(device)
                    pred = trained_model(data).cpu().numpy().flatten()
                    batch_preds.extend(pred)
                all_cal_preds.append(np.array(batch_preds))
        
        all_cal_preds = np.stack(all_cal_preds, axis=0)
        y_cal_pred_mean = all_cal_preds.mean(axis=0)
        y_cal_pred_std = all_cal_preds.std(axis=0)
        
        # Get true calibration values
        y_cal_true = y_val_noisy_cal.cpu().numpy() if isinstance(y_val_noisy_cal, torch.Tensor) else y_val_noisy_cal
        
        temperature = calibrate_uncertainty_simple(y_cal_pred_mean, y_cal_pred_std, y_cal_true)
        
        # Get test predictions
        all_test_preds = []
        with torch.no_grad():
            for _ in range(num_samples):
                batch_preds = []
                for data in test_loader:
                    data = data.to(device)
                    pred = trained_model(data).cpu().numpy().flatten()
                    batch_preds.extend(pred)
                all_test_preds.append(np.array(batch_preds))
        
        all_test_preds = np.stack(all_test_preds, axis=0)
        y_pred_mean = all_test_preds.mean(axis=0)
        y_pred_std_uncalibrated = all_test_preds.std(axis=0)
        y_pred_std_calibrated = y_pred_std_uncalibrated * temperature
        
        # STEP 3: Save with calibration
        save_uncertainty_values(
            y_pred_mean=y_pred_mean,
            y_pred_std=y_pred_std_uncalibrated,
            y_true_original=y_test_original.cpu().numpy().flatten(),
            y_true_noisy=test_y,
            filepath=args.filepath,
            model_name=model_name,
            rep='graph',
            sigma_noise=s,
            iteration=iteration,
            file_no=file_no,
            y_pred_std_calibrated=y_pred_std_calibrated,
            temperature=temperature
        )
        
        save_results(args.filepath, s, iteration, model_name, 'graph', args.sample_size, metrics)
    else:
        save_results(args.filepath, s, iteration, model_type, 'graph', args.sample_size, metrics)
    
    return metrics[3]

def train_graph_gp(train_graphs, train_y, test_graphs, test_y, val_graphs, val_y, args, s, iteration, file_no, y_test_original, trial=None):
    """
    This function already receives y values as parameters (train_y, test_y, val_y),
    so it's already compatible with the noisy values from Rust. No changes needed.
    """
    params = {}
    params_source = 'default'
    if hasattr(args, 'use_best_params') and args.use_best_params and not args.tuning:
        best_params = load_best_hyperparameters('graph_gp', 'graph')
        if best_params is not None:
            params = best_params
            params_source = 'tuned'
            print(f"Using tuned hyperparameters for graph_gp-graph")
    if not params:
        if args.tuning and trial is not None:
            params['kernel_name'] = trial.suggest_categorical('kernel', [
                'WeisfeilerLehman', 'VertexHistogram', 'EdgeHistogram', 'NeighborhoodHash'
            ])
            params['outputscale'] = trial.suggest_float('outputscale', 0.1, 10.0, log=True)
            params['likelihood_noise'] = trial.suggest_float('likelihood_noise', 1e-4, 0.1, log=True)
            params_source = 'tuning_trial'
        else:
            params['kernel_name'] = 'WeisfeilerLehman'
            params['outputscale'] = 1.0
            params['likelihood_noise'] = 1e-3
            params_source = 'default'
    
    kernel_map = {
        'WeisfeilerLehman': WeisfeilerLehmanKernel,
        'VertexHistogram': VertexHistogramKernel,
        'EdgeHistogram': EdgeHistogramKernel,
        'NeighborhoodHash': NeighborhoodHashKernel
    }
    
    # STEP 1: Split validation for calibration
    if args.uncertainty and val_graphs is not None and val_y is not None:
        split_idx = len(val_graphs) // 2
        train_graphs_full = train_graphs + val_graphs[:split_idx]
        train_y_full = torch.cat((train_y, val_y[:split_idx]), dim=0)
        val_graphs_cal = val_graphs[split_idx:]
        val_y_cal = val_y[split_idx:]
    elif val_graphs is not None and val_y is not None:
        train_graphs_full = train_graphs + val_graphs
        train_y_full = torch.cat((train_y, val_y), dim=0)
    else:
        train_graphs_full = train_graphs
        train_y_full = train_y
    
    X_train = NonTensorialInputs(train_graphs_full)
    X_test = NonTensorialInputs(test_graphs)
    y_train = train_y_full.flatten().float()
    y_test = test_y.flatten().float()
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise=params['likelihood_noise'])
    kernel_class = kernel_map[params['kernel_name']]
    kernel = kernel_class(edge_label='label') if params['kernel_name'] == 'EdgeHistogram' else kernel_class(node_label='label')
    model = GraphGP(X_train, y_train, likelihood, kernel)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    fit_gpytorch_model(mll)
    
    model.eval()
    likelihood.eval()
    
    # STEP 2: Get predictions and calibrate
    with torch.no_grad():
        # Test predictions
        test_preds = model(X_test)
        y_pred = test_preds.mean.numpy()
        pred_vars = test_preds.variance.numpy()
        y_pred_std_uncalibrated = np.sqrt(pred_vars)
        
        if args.uncertainty:
            # Calibration predictions
            X_val_cal = NonTensorialInputs(val_graphs_cal)
            cal_preds = model(X_val_cal)
            y_cal_pred_mean = cal_preds.mean.numpy()
            y_cal_pred_std = np.sqrt(cal_preds.variance.numpy())
            y_cal_true = val_y_cal.cpu().numpy()
            
            temperature = calibrate_uncertainty_simple(y_cal_pred_mean, y_cal_pred_std, y_cal_true)
            y_pred_std_calibrated = y_pred_std_uncalibrated * temperature
        else:
            temperature = None
            y_pred_std_calibrated = None
    
    metrics = calculate_regression_metrics(y_test.numpy(), y_pred, logging=True)
    save_results(args.filepath, s, iteration, "graph_gp", "graph", args.sample_size, metrics, params_source)
    
    # STEP 3: Save with calibration
    if args.uncertainty:
        save_uncertainty_values(
            y_pred_mean=y_pred,
            y_pred_std=y_pred_std_uncalibrated,
            y_true_original=y_test_original.cpu().numpy().flatten(),
            y_true_noisy=y_test.numpy(),
            filepath=args.filepath,
            model_name="graph_gp",
            rep="graph",
            sigma_noise=s,
            iteration=iteration,
            file_no=file_no,
            y_pred_std_calibrated=y_pred_std_calibrated,
            temperature=temperature
        )
    
    return metrics[3]

def train_custom_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, trial=None):
    model = load_custom_model(args.model_path)

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)

    if x_val is not None and y_val is not None:
        x_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
        val_loader = TorchDataLoader(TensorDataset(x_val_tensor, y_val_tensor), batch_size=32, shuffle=False)
    else:
        val_loader = None

    hyperparams = get_custom_hyperparameter_bounds(args.metadata_path) if args.metadata_path else {}

    learning_rate = hyperparams.get("learning_rate", [0.001, 0.001])[0]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    model.to(device)
    model.train()

    for _ in range(args.epochs):
        optimizer.zero_grad()
        y_pred_train = model(x_train_tensor).squeeze()
        loss = loss_fn(y_pred_train, y_train_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred = model(x_test_tensor).squeeze().cpu().numpy()

    if args.distribution in ["domain_mpnn", "domain_tanimoto"]:
        calculate_domain_metrics(y_test, y_pred, domain_labels, target_domain, args.dataset)
        logging = False
    else:
        logging = True

    metrics = calculate_regression_metrics(y_test, y_pred, logging=logging)

    save_results(args.filepath, s, iteration, "custom", rep, args.sample_size, metrics)

    return metrics[3]

# Sample hyperparameter file
# {
#     "learning_rate": [0.0001, 0.01],
#     "batch_size": [8, 64],
#     "dropout": [0.1, 0.5]
# }
def get_custom_hyperparameter_bounds(metadata_path):
    """
    Reads hyperparameter tuning bounds from a JSON file.
    Assumes the JSON file contains a dictionary with parameter names and their bounds.
    """
    try:
        with open(metadata_path, 'r') as f:
            hyperparams = json.load(f)
        return hyperparams
    except FileNotFoundError:
        raise ValueError("Metadata file not found. Please specify a valid path.")
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format in metadata file.")

def train_conformal_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, file_no, base_model_type, calibration_size, y_test_original, trial=None):
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
            params['alpha'] = alpha
            params['predictor_type'] = predictor_type
            
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
            # Default parameters
            alpha = 0.1
            predictor_type = 'split'
            params['alpha'] = alpha
            params['predictor_type'] = predictor_type
            params_source = 'default'
    
    # Extract conformal parameters
    alpha = params.pop('alpha', 0.1)
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
        
        x_full = np.vstack((x_train, x_val))
        y_full = np.hstack((y_train, y_val))
        
        x_train_tensor = torch.from_numpy(x_full).double()
        x_test_tensor = torch.from_numpy(x_test).double()
        y_train_tensor = torch.from_numpy(y_full).double()
        
        likelihood = gpytorch.likelihoods.GaussianLikelihood(noise=params.get('likelihood_noise', 1e-3))
        kernel_class = kernel_map[params.get('kernel_name', 'Tanimoto')]
        base_model = Gauche(x_train_tensor, y_train_tensor, likelihood, kernel_class)
        
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, base_model)
        fit_gpytorch_model(mll)
        
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
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(base_model.parameters(), lr=learning_rate)
        
        train_nn(base_model, train_loader, val_loader, criterion, optimizer, device, args, s, iteration, file_no, f'conformal_dnn', rep)
    
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
    save_results(args.filepath, s, iteration, model_name, rep, args.sample_size, metrics, params_source)
    
    if args.uncertainty:
        interval_width = y_upper - y_lower
        y_pred_std = interval_width / (2 * 1.645) if alpha == 0.1 else interval_width / (2 * 1.96)
        
        save_uncertainty_values(
            y_pred_mean=y_test_pred,
            y_pred_std=y_pred_std,
            y_true_original=y_test_original,
            y_true_noisy=y_test,
            filepath=args.filepath,
            model_name=model_name,
            rep=rep,
            sigma_noise=s,
            iteration=iteration,
            file_no=file_no,
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
    
    # Create base model (use only dim_h since that's what your GNN classes accept)
    if base_model_type == 'gin':
        base_model = GIN(dim_h=dim_h)
    else:  # gcn
        base_model = GCN(dim_h=dim_h)
    
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
    metrics = calculate_regression_metrics(y_pred, y_test, logging=logging_flag)
    
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
        y_pred_std = interval_width / (2 * 1.645) if alpha == 0.1 else interval_width / (2 * 1.96)
        
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
    """
    master_file = os.path.join(results_dir, 'master_tuned_hyperparameters.json')
    decisions_file = os.path.join(results_dir, 'hyperparameter_decisions.json')
    
    if not os.path.exists(master_file) or not os.path.exists(decisions_file):
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

def calibrate_bayesian_uncertainty(model, cal_loader, device, num_samples=100):
    """
    Calibrate BNN uncertainty estimates using variance scaling.
    Returns optimal scaling factor T.
    """
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
# TODO: testing different loss functions:
# dnn, mlp, mtl, residual_mlp, factorization_mlp, rnn, gru, custom
# You can customize or swap loss functions (e.g., use nn.L1Loss() instead of MSELoss) depending on your use case.
# TODO: add NGBoost and QRF to environment files
