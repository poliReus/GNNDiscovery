"""
Graph Convolutional Network implementation for molecular property prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GCNModel(nn.Module):
    """
    Graph Convolutional Network for molecular property prediction.
    
    This model is optimized for high-throughput inference on molecular graphs.
    Uses batch normalization and dropout for stability and regularization.
    
    Architecture:
        Input Layer: GCNConv(input_dim -> hidden_dim)
        Hidden Layers: GCNConv(hidden_dim -> hidden_dim) x (num_layers - 2)
        Output Layer: GCNConv(hidden_dim -> hidden_dim)
        Pooling: Global Mean Pooling
        Prediction: Linear(hidden_dim -> output_dim)
    
    Args:
        input_dim: Dimension of input node features
        hidden_dim: Dimension of hidden layers
        output_dim: Dimension of output predictions
        num_layers: Number of graph convolution layers
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 128,
        output_dim: int = 1,
        num_layers: int = 3,
        dropout: float = 0.2
    ):
        super(GCNModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Build GCN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Final prediction layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass through the GCN.
        
        Args:
            x: Node feature matrix [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch vector [num_nodes] (None for single graph)
        
        Returns:
            Predictions [batch_size, output_dim]
        """
        # Apply GCN layers with batch norm and dropout
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            
            if i < len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global mean pooling
        if batch is None:
            x = torch.mean(x, dim=0, keepdim=True)
        else:
            x = global_mean_pool(x, batch)
        
        # Final prediction
        out = self.fc(x)
        
        return out
    
    def reset_parameters(self):
        """Reset all learnable parameters."""
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.batch_norms:
            bn.reset_parameters()
        self.fc.reset_parameters()
