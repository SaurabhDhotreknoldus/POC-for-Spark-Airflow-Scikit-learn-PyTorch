"""PyTorch Deep Learning Model definition for NYC Taxi Fare Prediction.

Implements a Multi-Layer Perceptron (MLP) with batch normalization / dropout,
ReLU activations, and a non-negative output constraint.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FareMLP(nn.Module):
    """Multi-Layer Perceptron Regressor for estimating NYC taxi fare amounts."""

    def __init__(
        self,
        input_dim: int,
        hidden1: int = 128,
        hidden2: int = 64,
        dropout: float = 0.1,
    ):
        """Initializes the FareMLP network layers.

        Args:
            input_dim: Number of input features in transformed feature matrix.
            hidden1: Number of units in first hidden layer.
            hidden2: Number of units in second hidden layer.
            dropout: Dropout probability for regularization.
        """
        super().__init__()
        self.input_dim = input_dim

        self.fc1 = nn.Linear(input_dim, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.dropout2 = nn.Dropout(dropout)

        self.out = nn.Linear(hidden2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with non-negative output constraint.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Predicted fare amounts of shape (batch_size, 1).
        """
        x = self.fc1(x)
        # Avoid BatchNorm1d failure if batch size is 1 during evaluation
        if x.size(0) > 1:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        if x.size(0) > 1:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        fare = self.out(x)
        # Real-world taxi fares cannot be negative: clamp at 0
        return F.relu(fare)

    def count_parameters(self) -> int:
        """Returns the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

