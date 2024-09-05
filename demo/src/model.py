# model.py
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np

class PyTorchMLPRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', alpha=0.001, epochs=10, batch_size=32, solver='adam'):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.epochs = epochs
        self.batch_size = batch_size
        self.solver = solver

    def _create_model(self):
        layers = []
        input_dim = self.input_dim
        activation_fn = {'relu': nn.ReLU(), 'tanh': nn.Tanh()}[self.activation]
        
        for size in self.hidden_layer_sizes:
            layers.append(nn.Linear(input_dim, size))
            layers.append(activation_fn)
            input_dim = size
        layers.append(nn.Linear(input_dim, 1))
        
        self.model = nn.Sequential(*layers)
        
        # Set up the optimizer based on the solver
        if self.solver == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        elif self.solver == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.alpha)
        elif self.solver == 'lbfgs':
            self.optimizer = optim.LBFGS(self.model.parameters(), lr=self.alpha)
        else:
            raise ValueError(f"Unsupported solver: {self.solver}")

        self.criterion = nn.MSELoss()

    def fit(self, X, y):
        # Check if X and y are pandas DataFrame/Series and convert to NumPy arrays
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.to_numpy()

        self.input_dim = X.shape[1]
        self._create_model()
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in dataloader:
                def closure():
                    self.optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    loss.backward()
                    return loss
                
                if self.solver == 'lbfgs':
                    self.optimizer.step(closure)
                else:
                    self.optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    loss.backward()
                    self.optimizer.step()
        
        return self

    def predict(self, X):
        # Check if X is pandas DataFrame and convert to NumPy array
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            predictions = self.model(X_tensor).numpy()
        return predictions.flatten()
