"""
HPC Drug Screening Package
High-performance distributed GNN inference for drug discovery.
"""

__version__ = "0.1.0"
__author__ = "GNNDiscovery Team"
__description__ = "Distributed GNN inference on billions of molecules for drug discovery"

from .models.gcn import GCNModel

__all__ = ["GCNModel"]
