"""
Example: Training a GCN model for molecular property prediction.

This script demonstrates how to train the GCN model on molecular data.
In production, you would:
1. Load real molecular datasets
2. Convert SMILES to graphs using RDKit
3. Train with proper validation and checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch_geometric.datasets import MoleculeNet

from hpc_drug_screening.models import GCNModel


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        out = model(batch.x, batch.edge_index, batch.batch)
        
        # Compute loss
        loss = criterion(out, batch.y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            total_loss += loss.item()
    
    return total_loss / len(loader)


def main():
    """Main training loop."""
    print("=" * 70)
    print("GCN Training Example for Drug Discovery")
    print("=" * 70)
    
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset (example using MoleculeNet)
    # In production, replace with your own dataset
    print("\nLoading dataset...")
    try:
        dataset = MoleculeNet(root="./data", name="ESOL")
        print(f"Dataset size: {len(dataset)}")
        print(f"Number of features: {dataset.num_features}")
        print(f"Number of classes: {dataset.num_classes if hasattr(dataset, 'num_classes') else 1}")
    except Exception as e:
        print(f"Could not load MoleculeNet dataset: {e}")
        print("Using dummy data for demonstration...")
        # Create dummy dataset
        from torch_geometric.data import Data
        dataset = [
            Data(
                x=torch.randn(20, 9),
                edge_index=torch.randint(0, 20, (2, 40)),
                y=torch.randn(1)
            )
            for _ in range(100)
        ]
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:]
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create model
    input_dim = dataset[0].x.shape[1] if hasattr(dataset[0], 'x') else 9
    model = GCNModel(
        input_dim=input_dim,
        hidden_dim=128,
        output_dim=1,
        num_layers=3,
        dropout=0.2
    )
    model = model.to(device)
    
    print(f"\nModel architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print("\nStarting training...")
    num_epochs = 50
    best_val_loss = float("inf")
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_gcn_model.pt")
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f} - "
                  f"Val Loss: {val_loss:.4f}")
    
    print("\n" + "=" * 70)
    print(f"Training complete! Best validation loss: {best_val_loss:.4f}")
    print("Model saved as: best_gcn_model.pt")
    print("=" * 70)


if __name__ == "__main__":
    main()
