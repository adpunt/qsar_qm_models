import torch
import torch.nn as nn
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
from torch_geometric.nn import GCNConv, GINConv, GATv2Conv, global_mean_pool, global_add_pool
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
from torch_geometric.datasets import QM9
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.loader import DataLoader
from bayes_opt import BayesianOptimization, UtilityFunction
import gpytorch
# from gauche.kernels.fingerprint_kernels.tanimoto_kernel import TanimotoKernel
# from gauche.kernels.fingerprint_kernels.braun_blanquet_kernel import BraunBlanquetKernel
# from gauche.kernels.fingerprint_kernels.dice_kernel import DiceKernel
# from gauche.kernels.fingerprint_kernels.faith_kernel import FaithKernel
# from gauche.kernels.fingerprint_kernels.forbes_kernel import ForbesKernel
# from gauche.kernels.fingerprint_kernels.inner_product_kernel import InnerProductKernel
# from gauche.kernels.fingerprint_kernels.intersection_kernel import IntersectionKernel
# from gauche.kernels.fingerprint_kernels.minmax_kernel import MinMaxKernel
# from gauche.kernels.fingerprint_kernels.otsuka_kernel import OtsukaKernel
# from gauche.kernels.fingerprint_kernels.rand_kernel import RandKernel
# from gauche.kernels.fingerprint_kernels.rogers_tanimoto_kernel import RogersTanimotoKernel
# from gauche.kernels.fingerprint_kernels.russell_rao_kernel import RussellRaoKernel
# from gauche.kernels.fingerprint_kernels.sogenfrei_kernel import SogenfreiKernel
# from gauche.kernels.fingerprint_kernels.sokal_sneath_kernel import SokalSneathKernel
from gauce.kernels.fingerprint_kernels import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class RNNRegressionModel(nn.Module):
#     """Vanilla RNN with one recurrent layer"""

#     def __init__(self, input_size, hidden_size=32, num_layers=1, dropout_rate=0.2):
#         """
#         Vanilla RNN

#         Parameters
#         ----------
#         input_size : int
#             The number of expected features in the input vector
#         hidden_size : int
#             The number of features in the hidden state

#         """
#         super(RNNRegressionModel, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, 1)
#         self.dropout_rate = dropout_rate
#         self.dropout = nn.Dropout(p=self.dropout_rate)

#     def forward(self, x):
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
#         out, hn = self.rnn(x, h0)
#         out = out[:, -1]
#         out = self.dropout(out)
#         out = self.fc(out)
#         return out


# class GRURegressionModel(nn.Module):
#     """GRU network with one recurrent layer"""

#     def __init__(self, input_size, hidden_size=32, num_layers=1, dropout_rate=0.2):
#         """
#         GRU network

#         Parameters
#         ----------
#         input_size : int
#             The number of expected features in the input vector
#         hidden_size : int
#             The number of features in the hidden state

#         """
#         super(GRURegressionModel, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
#         self.fc = nn.Linear(hidden_size, 1)
#         self.dropout_rate = dropout_rate
#         self.dropout = nn.Dropout(p=self.dropout_rate)

#     def forward(self, x):
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
#         out, hn = self.gru(x, h0)
#         out = out[:, -1]
#         out = self.dropout(out)
#         out = self.fc(out)
#         return out

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

    def train(self, train_loader, val_loader, test_loader, n_epochs, print_every=10, logging=False):
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
            val_loss, _, _ = self._eval_epoch(test_loader)
            self.batch_loss += train_loss_batches
            self.train_loss.append(train_loss)
            self.val_loss.append(val_loss)
            if logging:
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

# define GP model from Gauche library
class Gauche(gpytorch.models.ExactGP):
  def __init__(self, train_x, train_y, likelihood, kernel):
    super(GaucheGP, self).__init__(train_x, train_y, likelihood)
    self.mean_module = gpytorch.means.ConstantMean()
    if kernel == 'tanimoto':
        self.covar_module = gpytorch.kernels.ScaleKernel(TanimotoKernel())
    elif kernel == 'braun_blanquet':
        self.covar_module = gpytorch.kernels.ScaleKernel(BraunBlanquetKernel())
    elif kernel == 'dice':
        self.covar_module = gpytorch.kernels.ScaleKernel(DiceKernel())
    elif kernel == 'faith':
        self.covar_module = gpytorch.kernels.ScaleKernel(FaithKernel())
    elif kernel == 'forbes':
        self.covar_module = gpytorch.kernels.ScaleKernel(ForbesKernel())
    elif kernel == 'inner_product':
        self.covar_module = gpytorch.kernels.ScaleKernel(InnerProductKernel())
    elif kernel == 'intersection':
        self.covar_module = gpytorch.kernels.ScaleKernel(IntersectionKernel())
    elif kernel == 'minmax':
        self.covar_module = gpytorch.kernels.ScaleKernel(MinMaxKernel())
    elif kernel == 'otsuka':
        self.covar_module = gpytorch.kernels.ScaleKernel(OtsukaKernel())
    elif kernel == 'rand':
        self.covar_module = gpytorch.kernels.ScaleKernel(RandKernel())
    elif kernel == 'rogers_tanimoto':
        self.covar_module = gpytorch.kernels.ScaleKernel(RogersTanimotoKernel())
    elif kernel == 'russell_rao':
        self.covar_module = gpytorch.kernels.ScaleKernel(RussellRaoKernel())
    elif kernel == 'sogenfrei':
        self.covar_module = gpytorch.kernels.ScaleKernel(SogenfreiKernel())
    elif kernel == 'sokal_sneath':
        self.covar_module = gpytorch.kernels.ScaleKernel(SokalSneathKernel())
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    # More kernels here
    # Batch versions of Tanimoto, Braun-Blanquet, Dice, Faith, Forbes, Inner Product, Intersection, MinMax, Otsuka, Rand, Rogers Tanimoto, Russel Rao, Sogenfrei, Sokal Sneath
  
  def forward(self, x):
    mean_x = self.mean_module(x)
    covar_x = self.covar_module(x)
    return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train_mlp(model, train_loader, val_loader, epochs, lr=0.001, weight_decay=0, print_every=10, logging=False):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(data).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data).squeeze()
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        if logging and epoch % print_every == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train loss: {avg_train_loss:.4f}, Validation loss: {avg_val_loss:.4f}")

    return model  # Returning the trained model

def predict_mlp(model, test_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            data, target = batch
            data = data.to(device)
            outputs = model(data).squeeze()  # Squeeze the output
            predictions.extend(outputs.cpu().numpy())
    return predictions


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
