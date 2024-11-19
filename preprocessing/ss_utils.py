"""
This module contains some helper functions for our main notebook
"""

import collections
import itertools
import time
import typing
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from rdkit import Chem
from rdkit.Chem import AllChem

import torch
from torch.utils import data
from torch import nn
from torch import optim
from torch.nn import functional as F
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage

from ignite.contrib.handlers import ProgressBar


@dataclass
class TrainParams:
    batch_size: int = 64
    val_batch_size: int = 64
    learning_rate: float = 1e-3
    num_epochs: int = 10
    device: typing.Optional[str] = 'cpu'  # <-- I have not run this on the GPU yet, so that may need some debugging.


class SmilesRegressionDataset(data.Dataset):
    """
    Dataset that holds SMILES molecule data along with an associated single
    regression target.
    """

    def __init__(self, smiles_list: typing.List[str],
                 regression_target_list: typing.List[float],
                 transform: typing.Optional[typing.Callable] = None):
        """
        :param smiles_list: list of SMILES strings represnting the molecules
        we are regressing on.
        :param regression_target_list: list of targets
        :param transform: an optional transform which will be applied to the
        SMILES string before it is returned.
        """
        self.smiles_list = smiles_list
        self.regression_target_list = regression_target_list
        self.transform = transform

        assert len(self.smiles_list) == len(self.regression_target_list), \
            "Dataset and targets should be the same length!"

    def __getitem__(self, index):
        x, y = self.smiles_list[index], self.regression_target_list[index]
        if self.transform is not None:
            x = self.transform(x)
        y = torch.tensor([y], dtype=torch.float32)
        return x, y

    def __len__(self):
        return len(self.smiles_list)

    @classmethod
    def create_from_df(cls, df: pd.DataFrame, smiles_column: str = 'smiles',
                       regression_column: str = 'y', transform=None):
        """
        convenience method that takes in a Pandas dataframe and turns it
        into an   instance of this class.
        :param df: Dataframe containing the data.
        :param smiles_column: name of column that contains the x data
        :param regression_column: name of the column which contains the
        y data (i.e. targets)
        :param transform: a transform to pass to class's constructor
        """
        # smiles_list = [x.strip() for x in df[smiles_column].tolist()]
        smiles_list = df[smiles_column].tolist()
        # targets = [float(y) for y in df[regression_column].tolist()]
        targets = df[regression_column].tolist()
        return cls(smiles_list, targets, transform)


def train_neural_network(train_df: pd.DataFrame, val_df: pd.DataFrame,
                          smiles_col:str, regression_column:str,
                         transform: typing.Callable,
                         neural_network: nn.Module,
                         corrected_labels: typing.Optional[np.ndarray] = None,
                         params: typing.Optional[TrainParams]=None,
                         collate_func: typing.Optional[typing.Callable]=None):
    """
    Trains a PyTorch NN module on train dataset, validates it each epoch and returns a series of useful metrics
    for further analysis. Note the networks parameters will be changed in place.

    :param train_df: data to use for training.
    :param val_df: data to use for validation.
    :param smiles_col: column name for SMILES data in Dataframe
    :param regression_column: column name for the data we want to regress to.
    :param transform: the transform to apply to the datasets to create new ones suitable for working with neural network
    :param neural_network: the PyTorch nn.Module to train
    :param corrected_labels: an array of corrected labels for training
    :param params: the training params eg number of epochs etc.
    :param collate_func: collate_fn to pass to dataloader constructor. Leave as None to use default.
    """
    if params is None:
        params = TrainParams()

    # Update the train and valid datasets with new parameters
    train_dataset = SmilesRegressionDataset.create_from_df(train_df, smiles_col, regression_column, transform=transform)
    val_dataset = SmilesRegressionDataset.create_from_df(val_df, smiles_col, regression_column, transform=transform)
    print(f"Train dataset is of size {len(train_dataset)} and valid of size {len(val_dataset)}")

    # Put into dataloaders
    train_dataloader = data.DataLoader(train_dataset, params.batch_size, shuffle=True,
                                       collate_fn=collate_func, num_workers=1)
    val_dataloader = data.DataLoader(val_dataset, params.val_batch_size, shuffle=False, collate_fn=collate_func,
                                       num_workers=1)

    # Optimizer
    optimizer = optim.Adam(neural_network.parameters(), lr=params.learning_rate)

    # Work out what device we're going to run on (ie CPU or GPU)
    device = params.device

    # We're going to use PyTorch Ignite to take care of the majority of the training boilerplate for us
    # see https://pytorch.org/ignite/
    # in particular we are going to follow the example
    # https://github.com/pytorch/ignite/blob/53190db227f6dda8980d77fa5351fa3ddcdec6fb/examples/contrib/mnist/mnist_with_tqdm_logger.py
    def prepare_batch(batch, device, non_blocking):
        x, y = batch
        return x.to(device), y.to(device)


    # TODO: what value of alpha should I be using? 
    # Replace the loss function here with your custom loss function
    alpha = 0.5  # You can adjust the alpha value as needed
    if corrected_labels is not None:
        def custom_loss(outputs, targets):
            original_loss = F.binary_cross_entropy_with_logits(outputs, targets)  # You can change this to your specific loss function
            corrected_loss = F.binary_cross_entropy_with_logits(outputs, corrected_labels)
            total_loss = (1 - alpha) * original_loss + alpha * corrected_loss
            return total_loss

        trainer = create_supervised_trainer(neural_network, optimizer, custom_loss, device=device, prepare_batch=prepare_batch)
    else:
        trainer = create_supervised_trainer(neural_network, optimizer, F.binary_cross_entropy_with_logits, device=device, prepare_batch=prepare_batch)

    evaluator = create_supervised_evaluator(neural_network,
                                            metrics={'loss': Loss(F.binary_cross_entropy_with_logits)},
                                            device=device, prepare_batch=prepare_batch)
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names='all')

    train_loss_list = []
    val_lost_list = []
    val_times_list = []

    @trainer.on(Events.EPOCH_COMPLETED | Events.STARTED)
    def log_training_results(engine):
        evaluator.run(train_dataloader)
        metrics = evaluator.state.metrics
        loss = metrics['loss']
        pbar.log_message("Epoch - {}".format(engine.state.epoch))
        pbar.log_message(
            "Training Results - Epoch: {}  Avg loss: {:.2f}"
                .format(engine.state.epoch, loss)
        )
        train_loss_list.append(loss)

    @trainer.on(Events.EPOCH_COMPLETED | Events.STARTED)
    def log_validation_results(engine):
        s_time = time.time()
        evaluator.run(val_dataloader)
        e_time = time.time()
        metrics = evaluator.state.metrics
        loss = metrics['loss']
        pbar.log_message(
            "Validation Results - Epoch: {} Avg loss: {:.2f}"
                .format(engine.state.epoch, loss))

        pbar.n = pbar.last_print_n = 0
        val_lost_list.append(loss)
        val_times_list.append(e_time - s_time)

    # We can now train our network!
    trainer.run(train_dataloader, max_epochs=params.num_epochs)

    # Having trained it wee are now also going to run through the validation set one
    # last time to get the actual predictions
    val_predictions = []
    neural_network.eval()
    for batch in val_dataloader:
        x, _ = batch  # We don't need the original labels
        x = x.to(device)

        print(f"Shape of input tensor (x): {x.shape}")

        if corrected_labels is not None:
            # Use corrected labels if available
            corrected_labels_tensor = torch.tensor(corrected_labels, dtype=torch.float32).to(device)
            y_pred = neural_network(x, corrected_labels_tensor)  # Use corrected labels in the forward pass
        else:
            # Use original labels
            y_pred = neural_network(x)
        print(f"Shape of predicted tensor (y_pred): {y_pred.shape}")
        val_predictions.append(y_pred.cpu().detach().numpy())

    neural_network.train()
    val_predictions = np.concatenate(val_predictions)

    # Create a table of useful metrics (as part of the information we return)
    total_number_params = sum([v.numel() for v in  neural_network.parameters()])
    out_table = [
        ["Num params", f"{total_number_params:.2e}"],
        ["Minimum train loss", f"{np.min(train_loss_list):.3f}"],
        ["Mean validation time", f"{np.mean(val_times_list):.3f}"],
        ["Minimum validation loss", f"{np.min(val_lost_list):.3f}"]
     ]

    # We will create a dictionary of results.
    results = dict(
        train_loss_list=train_loss_list,
        val_lost_list=val_lost_list,
        val_times_list=val_times_list,
        out_table=out_table,
        val_predictions=val_predictions
    )
    return results


def plot_train_and_val_using_mpl(train_loss, val_loss):
    """
    Plots the train and validation loss using Matplotlib
    """
    assert len(train_loss) == len(val_loss)

    f, ax = plt.subplots()
    x = np.arange(len(train_loss))
    ax.plot(x, np.array(train_loss), label='train')
    ax.plot(x, np.array(val_loss), label='val')
    ax.legend()
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    return f, ax

