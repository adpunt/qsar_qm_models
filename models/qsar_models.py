import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import os.path
import time
import random
import pickle
from multiprocessing import Manager, Process
import importlib
import plotly.express as px
import matplotlib.pyplot as plt
from tqdm import tqdm 
from statistics import mean

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, precision_recall_curve

import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, PandasTools
from rdkit.DataStructs import ExplicitBitVect

import torch
from torch.utils import data 
from torch import nn
from torch.nn import functional as F

import ss_utils
import FLuID as fluid

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.metrics import classification_report, balanced_accuracy_score, matthews_corrcoef
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.cluster import KMeans

params_rf = {
    'max_depth':[5,10,20],
    'min_samples_split':[2,8,32],
    'min_samples_leaf':[1,2,5,10],
    'n_estimators':[50,100,200]
}

params_svc = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': [0.01, 0.1, 1]
}

params_svc_best = {
    'C': 10, 
    'kernel': 'linear',
    'gamma': 0.01
}


# Load FluID library
importlib.reload(fluid)

"""
Binary Classification Neural Network class.

Attributes:
    fc1 (nn.Linear): First fully connected layer.
    relu1 (nn.ReLU): First ReLU activation function.
    fc2 (nn.Linear): Second fully connected layer.
    relu2 (nn.ReLU): Second ReLU activation function.
    fc3 (nn.Linear): Third fully connected layer for output.

Methods:
    forward(x): Forward pass of the neural network.
    get_last_layer_activations(x): Get activations from the last layer.

Parameters:
    input_dim (int): Dimension of the input features.
    hidden_dim1 (int): Dimension of the first hidden layer.
    hidden_dim2 (int): Dimension of the second hidden layer.
"""
class BinaryClassificationNN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3): 
        super(BinaryClassificationNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_dim3, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)  # Applying ReLU after the extra layer
        out = self.fc4(out)

        return out

    def get_second_last_layer_activations(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        activations = self.fc3(out)

        return activations

"""
Generate a cross-domain model trained and tested on different teacher clusters

Parameters:
    sample_weights (bool): Flag indicating whether to use sample weights for training.
    cw (str or dict): Class weights for the model, used to handle imbalanced classes.
    hypertuning (bool): Flag indicating whether to perform hyperparameter tuning for the model.
    label_table (pd.DataFrame): DataFrame containing label information.
    model_version (str): Version of the model ('rf' for Random Forest, 'svm' for Support Vector Machine).
    x_train (list): List of training features.
    y_train (list): List of training labels.
    x_test (list): List of test features.
    y_test (list): List of test labels.
    x_val (list): List of validation features, only used for cross-domain analysis.
    y_val (list): List of validation labels, only used for cross-domain analysis.
    agreement_rate_training (list): List of agreement rates for training data (optional).

Returns:
    dict: Dictionary containing various evaluation metrics for the model.
"""
def generate_model(sample_weights, cw, hypertuning, label_table, model_version, x_train, y_train, x_test, y_test, x_val=[], y_val=[], agreement_rate_training=[]):
    # Store metrics in a dictionary
    metrics = {}

    # Parameter hypertuning using built-in cross-validation
    if hypertuning:
        cv = KFold(n_splits=10,shuffle=True)
        if model_version == 'rf':
            model = GridSearchCV(RandomForestClassifier(class_weight=cw), params=params_rf, cv=cv,verbose=1,refit=True)
        else:
            model = GridSearchCV(SVC(class_weight=cw, kernel='linear'), param_grid=params_svc, cv=cv,verbose=1,refit=True)
    # Single model with no parameter hypertuning
    else:
        if model_version == 'rf':
            model = RandomForestClassifier(class_weight=cw)
        else:
            model = SVC(class_weight=cw, kernel='linear', C=10, gamma=0.01)

    # Train the model using a version of agreement rates as sample weights, or not
    if sample_weights:
        model.fit(list(x_train), y_train, sample_weight=agreement_rate_training)
    else:
        model.fit(list(x_train), y_train)

    # Record the best hyperparameters, if applicable
    if hypertuning:
        print('Best score: %0.2f', model.best_score_)
        print('Training set performance using best parameters (%s)', model.best_params_)
        model = model.best_estimator_

    # Make predictions
    prediction = model.predict(x_test)

    # Calculate Balanced Matthew Correlation Coefficient (BMCC)
    bmcc = matthews_corrcoef(y_test, prediction)
    metrics['balanced_matthew_corr_coefficient'] = bmcc
    print("bmcc: ", bmcc)

    # Calculate Balanced Accuracy
    balanced_acc = balanced_accuracy_score(y_test, prediction)
    metrics['balanced_accuracy'] = balanced_acc
    print("balanced accuracy: ", balanced_acc)

    # Print and save metrics
    confusion_matrix = confusion_matrix_summary(y_test, prediction)
    print(confusion_matrix)
    metrics['confusion_matrix_summary'] = confusion_matrix
    report = classification_report(y_test, prediction)
    print(report)
    metrics['classification_report'] = report

    return metrics
