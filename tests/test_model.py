"""
Unit tests for GCN model.
"""

import pytest
import torch
from torch_geometric.data import Data, Batch

from model import GCNModel


def test_gcn_model_creation():
    """Test that GCN model can be created."""
    model = GCNModel(input_dim=128, hidden_dim=128, output_dim=1)
    assert model is not None
    assert isinstance(model, torch.nn.Module)


def test_gcn_forward_single_graph():
    """Test forward pass with a single graph."""
    model = GCNModel(input_dim=10, hidden_dim=64, output_dim=1)
    model.eval()
    
    # Create dummy graph
    x = torch.randn(20, 10)
    edge_index = torch.randint(0, 20, (2, 40))
    
    with torch.no_grad():
        output = model(x, edge_index)
    
    assert output.shape == (1, 1)


def test_gcn_forward_batched_graphs():
    """Test forward pass with batched graphs."""
    model = GCNModel(input_dim=10, hidden_dim=64, output_dim=1)
    model.eval()
    
    # Create dummy batched graphs
    batch_size = 4
    graphs = []
    for _ in range(batch_size):
        x = torch.randn(15, 10)
        edge_index = torch.randint(0, 15, (2, 30))
        graphs.append(Data(x=x, edge_index=edge_index))
    
    batch = Batch.from_data_list(graphs)
    
    with torch.no_grad():
        output = model(batch.x, batch.edge_index, batch.batch)
    
    assert output.shape == (batch_size, 1)


def test_gcn_different_dimensions():
    """Test GCN with different input/output dimensions."""
    model = GCNModel(input_dim=50, hidden_dim=100, output_dim=5)
    
    x = torch.randn(25, 50)
    edge_index = torch.randint(0, 25, (2, 50))
    batch = torch.zeros(25, dtype=torch.long)
    
    model.eval()
    with torch.no_grad():
        output = model(x, edge_index, batch)
    
    assert output.shape == (1, 5)


def test_gcn_reset_parameters():
    """Test parameter reset functionality."""
    model = GCNModel(input_dim=10, hidden_dim=32, output_dim=1)
    
    # Store initial weights
    initial_weight = model.fc.weight.clone()
    
    # Reset parameters
    model.reset_parameters()
    
    # Check that weights changed
    assert not torch.equal(initial_weight, model.fc.weight)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
