"""Unit tests for PyTorch FareMLP neural network architecture."""

import pytest
import torch
import torch.nn as nn
from src.training.model import FareMLP


def test_fare_mlp_initialization():
    """Tests FareMLP initialization and layer configuration."""
    model = FareMLP(input_dim=25, hidden1=64, hidden2=32, dropout=0.2)
    assert model.input_dim == 25
    assert model.count_parameters() > 0


def test_fare_mlp_forward_pass_shape():
    """Verifies that forward pass produces (batch_size, 1) tensor."""
    batch_size = 16
    input_dim = 30
    model = FareMLP(input_dim=input_dim)
    model.eval()

    dummy_input = torch.randn(batch_size, input_dim)
    output = model(dummy_input)

    assert output.shape == (batch_size, 1)
    assert isinstance(output, torch.Tensor)


def test_fare_mlp_non_negative_fares():
    """Verifies that FareMLP strictly enforces non-negative predictions (fare >= 0)."""
    model = FareMLP(input_dim=10)
    model.eval()

    # Extreme negative input values
    extreme_negative_input = torch.full((32, 10), -100.0)
    output = model(extreme_negative_input)

    assert torch.all(output >= 0.0), "Predicted fares must never be negative!"


def test_fare_mlp_gradient_flow():
    """Verifies gradient backpropagation through all layers during training."""
    model = FareMLP(input_dim=12)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    x = torch.randn(8, 12)
    y = torch.tensor([[15.0], [25.0], [8.5], [42.0], [12.0], [30.0], [18.5], [5.0]])

    initial_weights = model.fc1.weight.clone()

    optimizer.zero_grad()
    preds = model(x)
    loss = criterion(preds, y)
    loss.backward()
    optimizer.step()

    updated_weights = model.fc1.weight
    assert not torch.equal(initial_weights, updated_weights), "Weights should update after optimizer step!"

