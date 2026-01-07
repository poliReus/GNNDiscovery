"""
GNN Model Definitions for Drug Screening
Simple Graph Convolutional Network (GCN) for molecular property prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GCNModel(nn.Module):
    """
    Graph Convolutional Network for molecular property prediction.
    
    Architecture:
        - 3 GCN layers with configurable hidden dimensions
        - ReLU activation and dropout between layers
        - Global mean pooling to aggregate node features
        - Final linear layer for prediction
    
    Args:
        input_dim: Dimension of input node features (default: 128)
        hidden_dim: Dimension of hidden layers (default: 128)
        output_dim: Dimension of output (1 for regression, num_classes for classification)
        num_layers: Number of GCN layers (default: 3)
        dropout: Dropout probability (default: 0.2)
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
            batch: Batch vector [num_nodes] indicating which graph each node belongs to
                   (required for batched inference)
        
        Returns:
            out: Predictions [batch_size, output_dim]
        """
        # Apply GCN layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            
            # Apply dropout except on last layer
            if i < len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling to get graph-level representation
        if batch is None:
            # Single graph: use mean pooling
            x = torch.mean(x, dim=0, keepdim=True)
        else:
            # Batched graphs: use global mean pool
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


class GATModel(nn.Module):
    """
    Graph Attention Network (GAT) for molecular property prediction.
    
    Alternative to GCN that uses attention mechanisms to weight neighbor contributions.
    
    Args:
        input_dim: Dimension of input node features (default: 128)
        hidden_dim: Dimension of hidden layers (default: 128)
        output_dim: Dimension of output (1 for regression)
        num_layers: Number of GAT layers (default: 3)
        heads: Number of attention heads (default: 4)
        dropout: Dropout probability (default: 0.2)
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 128,
        output_dim: int = 1,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.2
    ):
        super(GATModel, self).__init__()
        
        try:
            from torch_geometric.nn import GATConv
        except ImportError:
            raise ImportError(
                "GATConv requires torch_geometric. Install with: "
                "pip install torch-geometric"
            )
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        
        # Build GAT layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim * heads))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout)
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim * heads))
        
        # Output layer (single head)
        self.convs.append(
            GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout)
        )
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Final prediction layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, edge_index, batch=None):
        """Forward pass through the GAT."""
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            
            if i < len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        if batch is None:
            x = torch.mean(x, dim=0, keepdim=True)
        else:
            x = global_mean_pool(x, batch)
        
        # Final prediction
        out = self.fc(x)
        
        return out


def create_model(model_type: str = "gcn", **kwargs) -> nn.Module:
    """
    Factory function to create GNN models.
    
    Args:
        model_type: Type of model ('gcn' or 'gat')
        **kwargs: Additional arguments passed to model constructor
    
    Returns:
        Initialized model
    """
    model_type = model_type.lower()
    
    if model_type == "gcn":
        return GCNModel(**kwargs)
    elif model_type == "gat":
        return GATModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'gcn' or 'gat'.")


if __name__ == "__main__":
    """Quick test of model architecture."""
    print("Testing GCN Model...")
    
    # Create model
    model = GCNModel(input_dim=128, hidden_dim=128, output_dim=1)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy data
    num_nodes = 20
    num_edges = 40
    batch_size = 4
    
    x = torch.randn(num_nodes, 128)  # Node features
    edge_index = torch.randint(0, num_nodes, (2, num_edges))  # Edges
    batch = torch.tensor([i // (num_nodes // batch_size) for i in range(num_nodes)])
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index, batch)
    
    print(f"Input shape: {x.shape}")
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Batch shape: {batch.shape}")
    print(f"Output shape: {out.shape}")
    print("\nModel test successful!")
