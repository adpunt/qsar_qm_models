import torch
import torch.nn as nn
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
from torch_geometric.nn import GCNConv, GINConv, GATv2Conv, global_mean_pool, global_add_pool
import numpy as np
import math

import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
from torch_geometric.datasets import QM9
from torch_geometric.nn import GCNConv, GINConv, MessagePassing
from torch_geometric.loader import DataLoader
from torch_geometric.typing import Adj, OptTensor, PairTensor, Size
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
# from torch_sparse import SparseTensor
from torch_geometric.nn.inits import glorot, zeros
import gpytorch
from typing import Union
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.svm import SVR, SVC
from xgboost import XGBRegressor, XGBClassifier
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader as TorchDataLoader
from torch.nn.utils import parameters_to_vector as Params2Vec, vector_to_parameters as Vec2Params
import matplotlib.pyplot as plt
import torchbnn as bnn
from torchhk import transform_model
import lightgbm as lgb
from botorch import fit_gpytorch_model
import gauche
from gauche.kernels.fingerprint_kernels import *
from gauche.kernels.graph_kernels import *
from gauche import SIGP, NonTensorialInputs
from gauche.dataloader import MolPropLoader
from gauche.dataloader.data_utils import transform_data
from gauche.kernels.graph_kernels import WeisfeilerLehmanKernel, VertexHistogramKernel


from utils import * 

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

class RNNClassificationModel(nn.Module):
    """Vanilla RNN for classification with one recurrent layer"""

    def __init__(self, input_size, hidden_size=32, num_layers=1, num_classes=2):
        super(RNNClassificationModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)  # Multi-class output
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.rnn(x, h0)
        out = out[:, -1]  # Take last time step
        out = self.dropout(out)
        out = self.fc(out)
        return out  # No activation (CrossEntropyLoss expects raw logits)

class GRUClassificationModel(nn.Module):
    """GRU for classification with one recurrent layer"""

    def __init__(self, input_size, hidden_size=32, num_layers=1, num_classes=2):
        super(GRUClassificationModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)  # Multi-class output
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.gru(x, h0)
        out = out[:, -1]  # Take last time step
        out = self.dropout(out)
        out = self.fc(out)
        return out  # No activation (CrossEntropyLoss expects raw logits)

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

    def train(self, train_loader, val_loader, n_epochs, print_every=10):
        """
        Train the model

        Parameters
        ----------
        train_loader :
            a dataloader with training data
        val_loader :
            a dataloader with training data
        n_epochs :
            number of epochs to train for
        """
        for e in range(n_epochs):
            train_loss, train_loss_batches = self._train_epoch(train_loader)
            val_loss, _, _ = self._eval_epoch(val_loader)
            self.batch_loss += train_loss_batches
            self.train_loss.append(train_loss)
            self.val_loss.append(val_loss)
            if e % print_every == 0:
                print(f"Epoch {e+0:03} | train_loss: {train_loss:.5f} | val_loss: {val_loss:.5f}")

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

class MLPClassifier(nn.Module):
    """Multi-Layer Perceptron for classification on non-sequential data."""

    def __init__(self, input_size, hidden_size=32, num_hidden_layers=2, num_classes=2, dropout_rate=0.2):
        super(MLPClassifier, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        self.output_layer = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # Softmax for classification

    def forward(self, x):
        x = self.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        x = self.dropout(x)
        x = self.output_layer(x)
        return self.softmax(x)  # Final activation for classification

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

def training(loader, model, loss, optimizer):
    """Training one epoch

    Args:
        loader (DataLoader): loader (DataLoader): training data divided into batches
        model (nn.Module): GNN model to train on
        loss (nn.functional): loss function to use during training
        optimizer (torch.optim): optimizer during training

    Returns:
        float: training loss
    """
    model.train()

    current_loss = 0
    for d in loader:
        optimizer.zero_grad()
        d.x = d.x.float()

        out = model(d)

        l = loss(out, torch.reshape(d.y, (len(d.y), 1)))
        current_loss += l / len(loader)
        l.backward()
        optimizer.step()
    return current_loss, model

def validation(loader, model, loss):
    """Validation

    Args:
        loader (DataLoader): validation set in batches
        model (nn.Module): current trained model
        loss (nn.functional): loss function

    Returns:
        float: validation loss
    """
    model.eval()
    val_loss = 0
    for d in loader:
        out = model(d)
        l = loss(out, torch.reshape(d.y, (len(d.y), 1)))
        val_loss += l / len(loader)
    return val_loss

@torch.no_grad()
def testing(loader, model):
    """Testing

    Args:
        loader (DataLoader): test dataset
        model (nn.Module): trained model

    Returns:
        float: test loss
    """
    loss = torch.nn.MSELoss()
    test_loss = 0
    test_target = np.empty((0))
    test_y_target = np.empty((0))
    for d in loader:
        out = model(d)
        # NOTE
        # out = out.view(d.y.size())
        l = loss(out, torch.reshape(d.y, (len(d.y), 1)))
        test_loss += l / len(loader)

        # save prediction vs ground truth values for plotting
        test_target = np.concatenate((test_target, out.detach().numpy()[:, 0]))
        test_y_target = np.concatenate((test_y_target, d.y.detach().numpy()))

    return test_loss, test_target, test_y_target

def train_epochs(epochs, model, train_loader, val_loader, path):
    """Training over all epochs

    Args:
        epochs (int): number of epochs to train for
        model (nn.Module): the current model
        train_loader (DataLoader): training data in batches
        val_loader (DataLoader): validation data in batches
        path (string): path to save the best model

    Returns:
        array: returning train and validation losses over all epochs, prediction and ground truth values for training data in the last epoch
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    loss = torch.nn.MSELoss()

    train_target = np.empty((0))
    train_y_target = np.empty((0))
    train_loss = np.empty(epochs)
    val_loss = np.empty(epochs)
    best_loss = math.inf

    for epoch in range(epochs):
        epoch_loss, model = training(train_loader, model, loss, optimizer)
        v_loss = validation(val_loader, model, loss)
        if v_loss < best_loss:
            torch.save(model.state_dict(), path)
        for d in train_loader:
            out = model(d)
            if epoch == epochs - 1:
                # record truly vs predicted values for training data from last epoch
                train_target = np.concatenate((train_target, out.detach().numpy()[:, 0]))
                train_y_target = np.concatenate((train_y_target, d.y.detach().numpy()))

        train_loss[epoch] = epoch_loss.detach().numpy()
        val_loss[epoch] = v_loss.detach().numpy()

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
    return train_loss, val_loss, train_target, train_y_target, model

class GINCoTeaching:
    def __init__(self, dim_h, learning_rate=0.001, dropout_rate=0.5):
        self.model_f = GIN(dim_h, dropout_rate)
        self.model_g = GIN(dim_h, dropout_rate)
        self.optimizer_f = torch.optim.Adam(self.model_f.parameters(), lr=learning_rate)
        self.optimizer_g = torch.optim.Adam(self.model_g.parameters(), lr=learning_rate)
        self.criterion = torch.nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_f.to(self.device)
        self.model_g.to(self.device)

    def train_epoch(self, loader, epoch, tau, R):
        self.model_f.train()
        self.model_g.train()

        for data in loader:
            data = data.to(self.device)
            self.optimizer_f.zero_grad()
            self.optimizer_g.zero_grad()

            output_f = self.model_f(data)
            output_g = self.model_g(data)

            loss_f = self.criterion(output_f, data.y.view(-1, 1))
            loss_g = self.criterion(output_g, data.y.view(-1, 1))

            # Select small-loss instances
            _, indices_f = torch.topk(loss_f, int(R * len(loss_f)), largest=False)
            _, indices_g = torch.topk(loss_g, int(R * len(loss_g)), largest=False)

            small_loss_data_f = data.x[indices_f], data.edge_index[:, indices_f], data.batch[indices_f], data.y[indices_f]
            small_loss_data_g = data.x[indices_g], data.edge_index[:, indices_g], data.batch[indices_g], data.y[indices_g]

            # Update networks with peer network's small-loss instances
            self.optimizer_f.zero_grad()
            peer_output_f = self.model_f(Data(*small_loss_data_g))
            peer_loss_f = self.criterion(peer_output_f, small_loss_data_g[-1].view(-1, 1))
            peer_loss_f.backward()
            self.optimizer_f.step()

            self.optimizer_g.zero_grad()
            peer_output_g = self.model_g(Data(*small_loss_data_f))
            peer_loss_g = self.criterion(peer_output_g, small_loss_data_f[-1].view(-1, 1))
            peer_loss_g.backward()
            self.optimizer_g.step()

        # Adjust R
        R = 1 - min(epoch / tau, tau)
        return R

    def train(self, loader, epochs, tau):
        R = 1.0
        for epoch in range(epochs):
            R = self.train_epoch(loader, epoch, tau, R)
            print(f'Epoch {epoch+1}/{epochs} completed. R: {R:.4f}')

    def evaluate(self, loader):
        self.model_f.eval()
        self.model_g.eval()
        loss_f = 0
        loss_g = 0

        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                output_f = self.model_f(data)
                output_g = self.model_g(data)
                loss_f += self.criterion(output_f, data.y.view(-1, 1)).item()
                loss_g += self.criterion(output_g, data.y.view(-1, 1)).item()

        loss_f /= len(loader)
        loss_g /= len(loader)

        return loss_f, loss_g

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

class DNNClassificationModel(nn.Module):
    """Densely-connected neural network for classification tasks"""

    def __init__(self, input_size, hidden_size1=32, hidden_size2=32, num_classes=2):
        """
        Fully-connected neural network for classification

        Parameters
        ----------
        input_size : int
            Number of features in the input vector
        hidden_size1 : int
            Number of neurons in the first hidden layer
        hidden_size2 : int
            Number of neurons in the second hidden layer
        num_classes : int
            Number of output classes (2 for binary, >2 for multi-class)
        """
        super(DNNClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_classes)
        self.activation = nn.ReLU()  # Default activation (will be tuned)
        self.dropout = nn.Dropout(p=0.2)
        self.num_classes = num_classes

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # No activation here, handled externally based on task
        return x

# TODO: save plots to appropriate place
# TODO: modify this to work with non-DNN if anything else looks promising, right now it will fail
def loss_landscape(model, model_type, rep, s, x_test_tensor, y_test_tensor, device, iteration_seed, loss_landscape_flag):
    """
    Computes and visualizes 1D and 2D loss landscapes for a given neural network.

    Parameters:
    - model: PyTorch model (DNN or any NN)
    - x_test_tensor: Input test tensor
    - y_test_tensor: Ground truth labels for test set
    - device: Device (CPU/GPU) to run computations
    - iteration_seed: Identifier for saving plots
    - loss_landscape_flag: Boolean flag to enable/disable landscape computation
    """
    if not loss_landscape_flag:
        return  # Exit if landscape computation is disabled

    print("Computing loss landscape...")

    model_save_path = f"trained_dnn_{iteration_seed}.pt"
    torch.save(model.state_dict(), model_save_path)

    # Recreate the model with known architecture parameters
    if isinstance(model, DNNRegressionModel):
        infer_net = DNNRegressionModel(input_size=model.fc1.in_features,
                                       hidden_size1=model.fc1.out_features,
                                       hidden_size2=model.fc2.out_features).to(device)
    elif isinstance(model, DNNClassificationModel):
        infer_net = DNNClassificationModel(input_size=model.fc1.in_features,
                                           hidden_size1=model.fc1.out_features,
                                           hidden_size2=model.fc2.out_features,
                                           num_classes=model.fc3.out_features).to(device)
    else:
        raise ValueError("Unsupported model type for loss landscape analysis")

    # Load trained weights
    infer_net.load_state_dict(torch.load(model_save_path))

    infer_net.load_state_dict(torch.load(model_save_path))

    # Convert parameters to vectors
    theta_ast = Params2Vec(model.parameters()).detach()
    theta = Params2Vec(infer_net.parameters()).detach()

    loss_fn = torch.nn.MSELoss() if isinstance(model, DNNRegressionModel) else torch.nn.CrossEntropyLoss()

    # 1D Loss Landscape
    alphas = torch.linspace(-20, 20, 40)
    losses_1d = []

    for alpha in alphas:
        Vec2Params(alpha * theta_ast + (1 - alpha) * theta, infer_net.parameters())
        infer_net.eval()
        with torch.no_grad():
            y_pred = infer_net(x_test_tensor)
            loss = loss_fn(y_pred, y_test_tensor).item()
            losses_1d.append(loss)

    # 2D Loss Landscape
    x_range = torch.linspace(-20, 20, 20)
    y_range = torch.linspace(-20, 20, 20)
    alpha, beta = torch.meshgrid(x_range, y_range, indexing="ij")

    def tau_2d(alpha, beta, theta_ast):
        return alpha * theta_ast[:, None, None] + beta * alpha * theta_ast[:, None, None]

    space = tau_2d(alpha, beta, theta_ast)
    losses_2d = torch.empty_like(space[0, :, :])

    for a, _ in enumerate(x_range):
        print(f'Processing alpha = {a}')
        for b, _ in enumerate(y_range):
            Vec2Params(space[:, a, b] + theta_ast, infer_net.parameters())
            infer_net.eval()
            with torch.no_grad():
                y_pred = infer_net(x_test_tensor)
                losses_2d[a, b] = loss_fn(y_pred, y_test_tensor).item()

    # Plot 1D loss landscape
    plt.figure(figsize=(8, 6))
    plt.plot(alphas.numpy(), losses_1d)
    plt.xlabel("Alpha")
    plt.ylabel("Loss")
    plt.title("1D Loss Landscape")
    plt.grid()
    plt.savefig(f"../results/loss_landscape_1d_{model_type}_{rep}_{s}.png")
    plt.close()

    # Plot 2D loss contour
    plt.figure(figsize=(8, 6))
    plt.contourf(alpha.numpy(), beta.numpy(), losses_2d.numpy(), levels=50, cmap="viridis")
    plt.colorbar(label="Loss")
    plt.xlabel("Alpha")
    plt.ylabel("Beta")
    plt.title("2D Loss Contour")
    plt.savefig(f"../results/loss_landscape_2d_{model_type}_{rep}_{s}.png")
    plt.close()

    print("Loss landscape computation complete!")

def train_rf_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, trial=None):
    params = {}

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


    if args.dataset == 'QM9':
        model = RandomForestRegressor(random_state=iteration_seed, **params)
    else:
        params['criterion'] = trial.suggest_categorical('criterion', ['gini', 'entropy'])  # Classification
        model = RandomForestClassifier(random_state=iteration_seed, **params)

    x_train = np.vstack((x_train, x_val))
    y_train = np.hstack((y_train, y_val))

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    metrics = calculate_regression_metrics(y_test, y_pred, logging=True)

    save_results(args.filepath, s, iteration, 'rf', rep, args.sample_size, metrics[3], metrics[0], metrics[4])

    return metrics[3] if args.dataset == 'QM9' else metrics[0]

def train_svm_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, trial=None):
    params = {}

    if args.tuning:
        params['C'] = trial.suggest_int('C', 0, 100)

        params['gamma'] = trial.suggest_categorical('gamma', ['scale', 'auto'])
        params['kernel'] = trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid'])

        if params['kernel'] == 'poly':
            params['degree'] = trial.suggest_int('degree', 2, 5)
            params['coef0'] = trial.suggest_float('coef0', 0.0, 10.0)

        if params['kernel'] == 'sigmoid':
            params['coef0'] = trial.suggest_float('coef0', 0.0, 10.0)


    x_train = np.vstack((x_train, x_val))
    y_train = np.hstack((y_train, y_val))

    model = SVR(**params) if args.dataset == 'QM9' else SVC(**params)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    metrics = calculate_regression_metrics(y_test, y_pred, logging=True)

    save_results(args.filepath, s, iteration, 'svm', rep, args.sample_size, metrics[3], metrics[0], metrics[4])

    return metrics[3] if args.dataset == 'QM9' else metrics[0]

def train_xgboost_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, trial=None):
    params = {}

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

    if x_val is not None and y_val is not None:
        x_train = np.vstack((x_train, x_val))
        y_train = np.hstack((y_train, y_val))

    model = XGBRegressor(random_state=iteration_seed, **params) if args.dataset == 'QM9' else XGBClassifier(random_state=iteration_seed, **params)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    metrics = calculate_regression_metrics(y_test, y_pred, logging=True)

    save_results(args.filepath, s, iteration, 'xgboost', rep, args.sample_size, metrics[3], metrics[0], metrics[4])

    return metrics[3] if args.dataset == 'QM9' else metrics[0]

# TODO: classification
def train_gauche_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, trial=None):
    params = {}

    if args.tuning:
        params['kernel_name'] = trial.suggest_categorical('kernel', [
            'Tanimoto', 'BraunBlanquet', 'Dice', 'Faith', 'Forbes',
            'InnerProduct', 'Intersection', 'MinMax', 'Otsuka',
            'Rand', 'RogersTanimoto', 'Sorgenfrei', 'SokalSneath'
        ])
        params['outputscale'] = trial.suggest_float('outputscale', 0.1, 10.0, log=True)
        params['likelihood_noise'] = trial.suggest_float('likelihood_noise', 1e-4, 0.1, log=True)
    else:
        params['kernel_name'] = 'Tanimoto'
        params['outputscale'] = 1.0
        params['likelihood_noise'] = 1e-3

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
        'RogersTanimoto': gauche.kernels.fingerprint_kernels.rogers_tanimoto_kernel.RogersTanimotoKernel,
        'RussellRao': gauche.kernels.fingerprint_kernels.russell_rao_kernel.RussellRaoKernel,
        'SokalSneath': gauche.kernels.fingerprint_kernels.sokal_sneath_kernel.SokalSneathKernel
    }

    if x_val is not None and y_val is not None:
        x_train = np.vstack((x_train, x_val))
        y_train = np.hstack((y_train, y_val))

    x_train_tensor = torch.from_numpy(x_train).double()
    x_test_tensor = torch.from_numpy(x_test).double()
    y_train_tensor = torch.from_numpy(y_train).double()

    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise=params['likelihood_noise'])

    kernel_class = kernel_map[params['kernel_name']]
    model = Gauche(x_train_tensor, y_train_tensor, likelihood, kernel_class)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    fit_gpytorch_model(mll)

    model.eval()
    likelihood.eval()
    with torch.no_grad():
        preds = model(x_test_tensor)
        y_pred = preds.mean.numpy()
        pred_vars = preds.variance.numpy()

    metrics = calculate_regression_metrics(y_test, y_pred, logging=True) if args.dataset == 'QM9' else calculate_classification_metrics(y_test, y_pred, logging=True)

    save_results(args.filepath, s, iteration, "gauche", rep, args.sample_size, metrics[3], metrics[0], metrics[4])

    return metrics[3]

def train_nn(model, train_loader, val_loader, criterion, optimizer, device, epochs, patience=20, tolerance=0.01):
    model.to(device)
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(100):  # Max epochs
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                val_outputs = model(X_val)
                loss = criterion(val_outputs, y_val)
                val_loss += loss.item()

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

def train_dnn_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, trial=None):
    params = {}

    if args.tuning:
        params['hidden_size1'] = trial.suggest_categorical('hidden_size1', [32, 64, 128, 256, 512, 1024, 2048, 4096])
        params['hidden_size2'] = trial.suggest_categorical('hidden_size2', [32, 64, 128, 256, 512, 1024, 2048, 4096])
        params['activation'] = trial.suggest_categorical('activation', ['relu', 'tanh', 'softmax'])
    else:
        params['hidden_size1'], params['hidden_size2'] = 128, 64
        params['activation'] = 'relu'

    activation_map = {'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'softmax': nn.Softmax(dim=1)}
    activation = activation_map[params['activation']]

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

    x_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
    val_loader = TorchDataLoader(TensorDataset(x_val_tensor, y_val_tensor), batch_size=32, shuffle=False)

    train_loader = TorchDataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=32, shuffle=True)

    model = DNNRegressionModel(input_size=x_train.shape[1], hidden_size1=params['hidden_size1'], hidden_size2=params['hidden_size2']) if args.dataset == 'QM9' else DNNClassificationModel(input_size=x_train.shape[1], hidden_size1=params['hidden_size1'], hidden_size2=params['hidden_size2'])

    model.activation = activation
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_nn(model, train_loader, val_loader, criterion, optimizer, device, args.epochs)

    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(x_test_tensor).cpu().numpy()
    y_pred = y_pred_tensor.flatten()

    if args.loss_landscape:
        loss_landscape(model, "dnn", rep, s, x_test_tensor, y_test_tensor, device, iteration_seed, args.loss_landscape)

    metrics = calculate_regression_metrics(y_test, y_pred, logging=True)

    save_results(args.filepath, s, iteration, "dnn", rep, args.sample_size, metrics[3], metrics[0], metrics[4])

    return metrics[3] if args.dataset == 'QM9' else metrics[0]


# TODO: add bayesian here
def train_flexible_dnn_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, trial=None):
    params = {}

    if args.tuning:
        num_layers = trial.suggest_int("num_layers", 1, 4)
        hidden_sizes = []
        for i in range(num_layers):
            hidden_size = trial.suggest_categorical(f"hidden_size_{i}", [32, 64, 128, 256, 512, 1024])
            hidden_sizes.append(hidden_size)
        params['hidden_sizes'] = hidden_sizes
        params['activation'] = trial.suggest_categorical('activation', ['relu', 'tanh'])
    else:
        params['hidden_sizes'] = [128, 64]
        params['activation'] = 'relu'

    activation_map = {'relu': nn.ReLU(), 'tanh': nn.Tanh()}
    activation = activation_map[params['activation']]

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)

    train_loader = TorchDataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
    val_loader = TorchDataLoader(TensorDataset(x_val_tensor, y_val_tensor), batch_size=32, shuffle=False)

    model = FlexibleDNNRegressionModel(input_size=x_train.shape[1], hidden_sizes=params['hidden_sizes'], activation_fn=activation).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_nn(model, train_loader, val_loader, criterion, optimizer, device, args.epochs)

    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(x_test_tensor).cpu().numpy()
    y_pred = y_pred_tensor.flatten()

    metrics = calculate_regression_metrics(y_test, y_pred, logging=True)
    save_results(args.filepath, s, iteration, "dnn", rep, args.sample_size, metrics[3], metrics[0], metrics[4])

    return metrics[3] if args.dataset == 'QM9' else metrics[0]


def train_lgb_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, trial=None):
    params = {}

    if args.tuning:
        params['num_leaves'] = trial.suggest_int('num_leaves', 10, 200)

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
        params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.5, 1.0)
        params['n_estimators'] = trial.suggest_int('n_estimators', 10, 2000)
        params['min_child_samples'] = trial.suggest_int('min_child_samples', 1, 50)

    param_dict = {
        'objective': 'regression' if args.dataset == 'QM9' else 'binary',
        'metric': 'r2' if args.dataset == 'QM9' else 'binary_logloss',
        'random_state': iteration_seed
    }
    param_dict.update(params)

    if x_val is not None and y_val is not None:
        x_train = np.vstack((x_train, x_val))
        y_train = np.hstack((y_train, y_val))

    train_data = lgb.Dataset(x_train, label=y_train)

    model = lgb.train(param_dict, train_data, num_boost_round=100)

    y_pred = model.predict(x_test)

    metrics = calculate_regression_metrics(y_test, y_pred, logging=True)

    save_results(args.filepath, s, iteration, "lgb", rep, args.sample_size, metrics[3], metrics[0], metrics[4])

    return metrics[3] if args.dataset == 'QM9' else metrics[0]

def train_mlp_variant_model(x_train, y_train, x_test, y_test, x_val, y_val, model_type, args, s, rep, iteration, iteration_seed, trial=None):
    params = {}

    if args.tuning:
        params['hidden_size'] = trial.suggest_categorical('hidden_size', [32, 64, 128, 256, 512, 1024])
        params['num_hidden_layers'] = trial.suggest_int('num_hidden_layers', 1, 5)
        params['dropout_rate'] = trial.suggest_float('dropout_rate', 0.1, 0.5)
        params['lr'] = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    else:
        params['hidden_size'], params['num_hidden_layers'], params['dropout_rate'], params['lr'] = 128, 2, 0.2, 0.001

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

    if x_val is not None and y_val is not None:
        x_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
        val_loader = TorchDataLoader(TensorDataset(x_val_tensor, y_val_tensor), batch_size=32, shuffle=False)
    else:
        val_loader = None

    train_loader = TorchDataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=32, shuffle=True)

    if model_type == "mlp":
        model = MLPRegressor(input_size=x_train.shape[1], hidden_size=params['hidden_size'],
                             num_hidden_layers=params['num_hidden_layers'], dropout_rate=params['dropout_rate']) \
            if args.dataset == 'QM9' else \
            MLPClassifier(input_size=x_train.shape[1], hidden_size=params['hidden_size'],
                          num_hidden_layers=params['num_hidden_layers'], num_classes=len(set(y_train)),
                          dropout_rate=params['dropout_rate'])
        criterion = nn.MSELoss() if args.dataset == 'QM9' else nn.CrossEntropyLoss()

    elif model_type == "residual_mlp":
        model = ResidualMLP(input_size=x_train.shape[1], hidden_size=128, num_layers=3)
        criterion = nn.MSELoss()

    elif model_type == "factorization_mlp":
        model = FactorizationMLP(input_size=x_train.shape[1], hidden_size=128, factor_size=16)
        criterion = nn.MSELoss()

    elif model_type == "mtl":
        model = MTLRegressionModel(input_size=x_train.shape[1], hidden_size=128, num_tasks=1)
        criterion = nn.MSELoss()

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    train_nn(model, train_loader, val_loader or train_loader, criterion, optimizer, device, args.epochs)

    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(x_test_tensor).cpu().numpy()
    y_pred = y_pred_tensor.flatten() if args.dataset == 'QM9' else np.argmax(y_pred_tensor, axis=1)

    if args.loss_landscape:
        loss_landscape(model, model_type, rep, s, x_test_tensor, y_test_tensor, device, iteration_seed, args.loss_landscape)

    metrics = calculate_regression_metrics(y_test, y_pred, logging=True)

    save_results(args.filepath, s, iteration, model_type, rep, args.sample_size, metrics[3], metrics[0], metrics[4])

    return metrics[3] if args.dataset == 'QM9' else metrics[0]

def train_rnn_variant_model(x_train, y_train, x_test, y_test, x_val, y_val, model_type, args, s, rep, iteration, iteration_seed, trial=None):
    if model_type not in ["rnn", "gru"] or rep not in ['smiles', 'randomized_smiles']:
        raise ValueError("Invalid model type or representation for RNN/GRU training")

    params = {}

    if args.tuning:
        params['hidden_size'] = trial.suggest_categorical('hidden_size', [32, 64, 128, 256])
        params['num_layers'] = trial.suggest_int('num_layers', 1, 3)
        params['dropout_rate'] = trial.suggest_float('dropout_rate', 0.1, 0.5)
        params['lr'] = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    else:
        params['hidden_size'], params['num_layers'], params['dropout_rate'], params['lr'] = 128, 1, 0.2, 0.001

    if args.dataset == 'QM9':
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)
    else:
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1).to(device)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1).to(device)

    if x_val is not None and y_val is not None:
        x_val_tensor = torch.tensor(x_val, dtype=torch.float32).unsqueeze(1).to(device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32 if args.dataset == 'QM9' else torch.long).view(-1, 1).to(device)
        val_loader = TorchDataLoader(TensorDataset(x_val_tensor, y_val_tensor), batch_size=32, shuffle=False)
    else:
        val_loader = None

    train_loader = TorchDataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
    test_loader = TorchDataLoader(TensorDataset(x_test_tensor, y_test_tensor), batch_size=32, shuffle=False)

    if args.dataset == 'QM9':
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
    else:
        model = RNNClassificationModel(
            input_size=x_train.shape[1],
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            num_classes=len(set(y_train))
        ) if model_type == "rnn" else GRUClassificationModel(
            input_size=x_train.shape[1],
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            num_classes=len(set(y_train))
        )
        criterion = nn.CrossEntropyLoss()

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    trainer = ModelTrainer(model, lr=params['lr'])
    trainer.train(train_loader, val_loader or train_loader, n_epochs=args.epochs)

    y_pred_tensor = trainer.validate(test_loader)[1] 

    y_pred = np.argmax(y_pred_tensor, axis=1) if args.dataset != 'QM9' else np.array(y_pred_tensor).flatten()

    if args.loss_landscape:
        loss_landscape(model, model_type, rep, s, x_test_tensor, y_test_tensor, device, iteration_seed, args.loss_landscape)

    metrics = calculate_regression_metrics(y_test, y_pred, logging=True)

    save_results(args.filepath, s, iteration, model_type, rep, args.sample_size, metrics[3], metrics[0], metrics[4])

    return metrics[3] if args.dataset == 'QM9' else metrics[0]

def train_graph_gp(train_graphs, train_y, test_graphs, test_y, val_graphs, val_y, args, s, iteration, trial=None):
    params = {}

    if args.tuning and trial is not None:
        params['kernel_name'] = trial.suggest_categorical('kernel', [
            'WeisfeilerLehman', 'VertexHistogram', 'EdgeHistogram', 'NeighborhoodHash'
        ])
        params['outputscale'] = trial.suggest_float('outputscale', 0.1, 10.0, log=True)
        params['likelihood_noise'] = trial.suggest_float('likelihood_noise', 1e-4, 0.1, log=True)
    else:
        params['kernel_name'] = 'WeisfeilerLehman'
        params['outputscale'] = 1.0
        params['likelihood_noise'] = 1e-3

    kernel_map = {
        'WeisfeilerLehman': WeisfeilerLehmanKernel,
        'VertexHistogram': VertexHistogramKernel,
        'EdgeHistogram': EdgeHistogramKernel,
        'NeighborhoodHash': NeighborhoodHashKernel
    }

    if val_graphs is not None and val_y is not None:
        train_graphs = train_graphs + val_graphs
        train_y = torch.cat((train_y, val_y), dim=0)

    # 1. Wrap graphs into NonTensorialInputs
    X_train = NonTensorialInputs(train_graphs)
    X_test = NonTensorialInputs(test_graphs)

    # 2. Setup labels
    y_train = train_y.flatten().float()
    y_test = test_y.flatten().float()

    # 3. Setup Likelihood and Kernel
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise=params['likelihood_noise'])

    kernel_class = kernel_map[params['kernel_name']]
    kernel = kernel_class(node_label='label')  # 'label' is what qm9_to_networkx() puts on nodes

    # 4. Define GraphGP model
    model = GraphGP(X_train, y_train, likelihood, kernel)

    # 5. Fit GP model
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    fit_gpytorch_model(mll)

    # 6. Evaluate
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        preds = model(X_test)
        y_pred = preds.mean.numpy()

    # 7. Metrics
    metrics = calculate_regression_metrics(y_test.numpy(), y_pred, logging=True)

    # 8. Save
    save_results(args.filepath, s, iteration, "graph_gp", "graph", args.sample_size, metrics[3], metrics[0], metrics[4])

    return metrics[3]  # Return R^2 for Optuna if tuning


# TODO: actually need to call
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
    loss_fn = torch.nn.MSELoss() if args.dataset == 'QM9' else torch.nn.CrossEntropyLoss()

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

    metrics = calculate_regression_metrics(y_test, y_pred, logging=logging) if args.dataset == 'QM9' else calculate_classification_metrics(y_test, y_pred, logging=logging)

    save_results(args.filepath, s, iteration, "custom", rep, args.sample_size, metrics[3], metrics[0], metrics[4])

    return metrics[3] if args.dataset == 'QM9' else metrics[0]

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

# TODO: test different loss functions
# dnn, mlp, mtl, residual_mlp, factorization_mlp, rnn, gru, custom
# You can customize or swap loss functions (e.g., use nn.L1Loss() instead of MSELoss) depending on your use case.
