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
from bayes_opt import BayesianOptimization, UtilityFunction
import gpytorch
from typing import Union

# TODO: potentially have to specify different DataLoader

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
from gauche.kernels.fingerprint_kernels import *

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

# TODO: create a class GTAT like GCN/GIN using the GTATConv

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


# define GP model from Gauche library
class Gauche(gpytorch.models.ExactGP):
  def __init__(self, train_x, train_y, likelihood, kernel):
    super(Gauche, self).__init__(train_x, train_y, likelihood)
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

# TODO: fix imports
# GTAT from https://github.com/kouzheng
# class GTATConv(MessagePassing):
#     _alpha: OptTensor

#     def __init__(self, in_channels: int, out_channels: int, heads: int,
#                  topology_channels:int = 15,
#                  concat: bool = True, negative_slope: float = 0.2,
#                  dropout: float = 0., add_self_loops: bool = True,
#                  bias: bool = True, share_weights: bool = False, **kwargs):
#         super(GTATConv, self).__init__(node_dim=0, **kwargs)

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.topology_channels = topology_channels
#         self.heads = heads
#         self.concat = concat
#         self.negative_slope = negative_slope
#         self.dropout = dropout
#         self.add_self_loops = add_self_loops
#         self.share_weights = share_weights
#         self.lin_l = Linear(in_channels, heads * out_channels, bias=bias,
#                             weight_initializer='glorot')
        
#         if share_weights:
#             self.lin_r = self.lin_l
#         else:
#             self.lin_r = Linear(in_channels, heads * out_channels, bias=bias,
#                                 weight_initializer='glorot')
        
        

#         self.att = Parameter(torch.Tensor(1, heads, out_channels))

#         self.att2 = Parameter(torch.Tensor(1, heads, self.topology_channels))

#         if bias and concat:
#             self.bias = Parameter(torch.Tensor(heads * out_channels))
#         elif bias and not concat:
#             self.bias = Parameter(torch.Tensor(out_channels))
#         else:
#             self.register_parameter('bias', None)

#         self._alpha1 = None
#         self._alpha2 = None

#         self.bias2 =  Parameter(torch.Tensor(self.topology_channels))

#         self.reset_parameters()

#     def reset_parameters(self):
#         self.lin_l.reset_parameters()
#         self.lin_r.reset_parameters()
#         glorot(self.att)
#         glorot(self.att2)
#         zeros(self.bias)
#         zeros(self.bias2)

#     def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
#                 topology: Tensor,
#                 size: Size = None, return_attention_weights: bool = None):
#         # type: (Union[Tensor, PairTensor], Tensor , Tensor, Size, NoneType) -> Tensor  # noqa
#         # type: (Union[Tensor, PairTensor], SparseTensor, Size, NoneType) -> Tensor  # noqa
#         # type: (Union[Tensor, PairTensor], Tensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
#         # type: (Union[Tensor, PairTensor], SparseTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
#         r"""
#         Args:
#             return_attention_weights (bool, optional): If set to :obj:`True`,
#                 will additionally return the tuple
#                 :obj:`(edge_index, attention_weights)`, holding the computed
#                 attention weights for each edge. (default: :obj:`None`)
#         """
#         H, C = self.heads, self.out_channels

#         x_l: OptTensor = None
#         x_r: OptTensor = None
#         if isinstance(x, Tensor):
#             assert x.dim() == 2
#             x_l = self.lin_l(x).view(-1, H, C)  #(N , heads, features)
#             if self.share_weights:
#                 x_r = x_l
#             else:
#                 x_r = self.lin_r(x).view(-1, H, C)


#         assert x_l is not None
#         assert x_r is not None
#         topology = topology.unsqueeze(dim = 1)
#         topology = topology.repeat(1, self.heads, 1)
#         x_l = torch.cat((x_l,topology), dim = -1)
#         x_r = torch.cat((x_r,topology), dim = -1)

#         if self.add_self_loops:
#             if isinstance(edge_index, Tensor):
#                 num_nodes = x_l.size(0)
#                 if x_r is not None:
#                     num_nodes = min(num_nodes, x_r.size(0))
#                 if size is not None:
#                     num_nodes = min(size[0], size[1])
#                 edge_index, _ = remove_self_loops(edge_index)
#                 edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
#             # elif isinstance(edge_index, SparseTensor):
#             #     edge_index = set_diag(edge_index)

#         out_all = self.propagate(edge_index, x=(x_l, x_r), size=size)
#         out = out_all[ : , : , :self.out_channels ]
#         out2 = out_all[ : , : , self.out_channels:]
#         alpha1 = self._alpha1
#         self._alpha1 = None
#         alpha2 = self._alpha2
#         self._alpha2 = None

#         if self.concat:
#             out = out.reshape(-1, self.heads * self.out_channels)
#         else:
#             out = out.mean(dim=1)

#         if self.bias is not None:
#             out += self.bias

#         out2 = out2.mean(dim=1)
#         out2 += self.bias2

#         if isinstance(return_attention_weights, bool):
#             assert alpha is not None
#             if isinstance(edge_index, Tensor):
#                 return out, (edge_index, alpha)
#             # elif isinstance(edge_index, SparseTensor):
#             #     return out, edge_index.set_value(alpha, layout='coo')
#         else:
#             return out , out2

#     def message(self, x_j: Tensor, x_i: Tensor, index: Tensor, ptr: OptTensor,
#                 size_i: Optional[int]) -> Tensor:
#         x = x_i + x_j
#         alpha1 = (x[:, :, :self.out_channels] * self.att).sum(dim=-1)
#         alpha2 = (x[:, :, self.out_channels:] * self.att2).sum(dim=-1)
#         alpha1 = F.leaky_relu(alpha1 ,self.negative_slope )
#         alpha2 = F.leaky_relu(alpha2 ,self.negative_slope )
#         alpha1 = softmax(alpha1, index, ptr, size_i)
#         alpha2 = softmax(alpha2, index, ptr, size_i)
#         self._alpha1 = alpha1
#         self._alpha2 = alpha2
#         alpha1= F.dropout(alpha1, p=self.dropout, training=self.training)
#         alpha2= F.dropout(alpha2, p=self.dropout, training=self.training)
#         return torch.cat((x_j[:, :, :self.out_channels]* alpha2.unsqueeze(-1), x_j[:, :, self.out_channels: ]* alpha1.unsqueeze(-1)) ,dim = -1)

#     def __repr__(self):
#         return '{}({}, {}, heads={})'.format(self.__class__.__name__,
#                                              self.in_channels,
#                                              self.out_channels, self.heads)

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

# Define the DNN model
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

def train_dnn(model, train_loader, val_loader, criterion, optimizer, device, epochs, patience=20, tolerance=0.01):
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
