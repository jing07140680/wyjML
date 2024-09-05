import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

class PyTorchMLPRegressor(BaseEstimator, RegressorMixin, nn.Module):
    def __init__(self, hidden_layer_sizes=(10,), activation='relu', alpha=0.0001, epochs=20, batch_size=32, solver='adam'):
        nn.Module.__init__(self)  # Initialize nn.Module
        super(PyTorchMLPRegressor, self).__init__()
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.epochs = epochs
        self.batch_size = batch_size
        self.solver = solver

        # Define the model architecture
        self._build_model()

    def _build_model(self):
        self.activation_func = self._get_activation_function(self.activation)
        layers = []
        input_dim = 10  # Adjust this according to your data features

        for size in self.hidden_layer_sizes:
            layers.append(nn.Linear(input_dim, size))
            layers.append(self.activation_func())
            input_dim = size

        layers.append(nn.Linear(input_dim, 1))  # Output layer for regression
        self.network = nn.Sequential(*layers)

        self.optimizer = self._get_optimizer()
        self.criterion = nn.MSELoss()

    def _get_activation_function(self, activation):
        if activation == 'relu':
            return nn.ReLU
        elif activation == 'tanh':
            return nn.Tanh
        else:
            raise ValueError("Unsupported activation function: {}".format(activation))

    def _get_optimizer(self):
        if self.solver == 'adam':
            return optim.Adam(self.parameters(), lr=self.alpha)
        elif self.solver == 'sgd':
            return optim.SGD(self.parameters(), lr=self.alpha)
        elif self.solver == 'lbfgs':
            return optim.LBFGS(self.parameters(), lr=self.alpha)
        else:
            raise ValueError("Unsupported solver: {}".format(self.solver))

    def forward(self, x):
        return self.network(x)

    def _prepare_data(self, X, y, device):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.to_numpy()

        X = torch.tensor(X, dtype=torch.float32).to(device)
        y = torch.tensor(y, dtype=torch.float32).to(device)
        return X, y

    def fit(self, X, y):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)  # Move model to GPU
        X, y = self._prepare_data(X, y, device)

        for epoch in range(self.epochs):
            self.train()
            for start in range(0, len(X), self.batch_size):
                end = min(start + self.batch_size, len(X))
                batch_X = X[start:end]
                batch_y = y[start:end]

                self.optimizer.zero_grad()
                outputs = self(batch_X)
                loss = self.criterion(outputs.squeeze(), batch_y)
                loss.backward()
                self.optimizer.step()

        return self

    def predict(self, X):
        self.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X = X.to(device)
        with torch.no_grad():
            outputs = self(X)
            return outputs.cpu().numpy()

    def get_params(self, deep=True):
        return {
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'activation': self.activation,
            'alpha': self.alpha,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'solver': self.solver
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        self._build_model()
        return self
