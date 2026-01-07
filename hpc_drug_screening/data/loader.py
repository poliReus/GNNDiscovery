"""
SMILES Data Loader
Efficient loading and batching of SMILES strings for distributed processing.
"""

import os
from pathlib import Path
from typing import List, Iterator, Optional

import torch
from torch_geometric.data import Data, Batch


class SMILESDataLoader:
    """
    Data loader for SMILES files.
    
    Efficiently reads SMILES strings from files and prepares them for
    distributed GNN inference.
    
    Args:
        filepath: Path to SMILES file (one molecule per line)
        chunk_size: Number of molecules to read at once
        skip_invalid: Skip invalid SMILES strings
    """
    
    def __init__(
        self,
        filepath: str,
        chunk_size: int = 10000,
        skip_invalid: bool = True
    ):
        self.filepath = Path(filepath)
        self.chunk_size = chunk_size
        self.skip_invalid = skip_invalid
        
        if not self.filepath.exists():
            raise FileNotFoundError(f"SMILES file not found: {filepath}")
        
        # Count total lines
        self.total_molecules = self._count_lines()
    
    def _count_lines(self) -> int:
        """Count total number of valid lines in file."""
        count = 0
        with open(self.filepath, "r") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count
    
    def __len__(self) -> int:
        """Return total number of molecules."""
        return self.total_molecules
    
    def __iter__(self) -> Iterator[List[str]]:
        """Iterate over chunks of SMILES strings."""
        chunk = []
        
        with open(self.filepath, "r") as f:
            for line in f:
                smiles = line.strip()
                
                if not smiles:
                    continue
                
                chunk.append(smiles)
                
                if len(chunk) >= self.chunk_size:
                    yield chunk
                    chunk = []
            
            # Yield remaining molecules
            if chunk:
                yield chunk
    
    def load_all(self) -> List[str]:
        """Load all SMILES strings into memory.
        
        Warning: Use only for small datasets.
        
        Returns:
            List of all SMILES strings
        """
        smiles_list = []
        with open(self.filepath, "r") as f:
            for line in f:
                smiles = line.strip()
                if smiles:
                    smiles_list.append(smiles)
        return smiles_list
    
    def get_chunk(self, chunk_id: int) -> List[str]:
        """Get a specific chunk of SMILES by index.
        
        Args:
            chunk_id: Index of the chunk to retrieve
            
        Returns:
            List of SMILES strings in the chunk
        """
        start_idx = chunk_id * self.chunk_size
        end_idx = start_idx + self.chunk_size
        
        smiles_chunk = []
        with open(self.filepath, "r") as f:
            for i, line in enumerate(f):
                if i < start_idx:
                    continue
                if i >= end_idx:
                    break
                
                smiles = line.strip()
                if smiles:
                    smiles_chunk.append(smiles)
        
        return smiles_chunk
    
    @property
    def num_chunks(self) -> int:
        """Get total number of chunks."""
        return (self.total_molecules + self.chunk_size - 1) // self.chunk_size


class GraphBatchLoader:
    """
    Batch loader for PyTorch Geometric graphs.
    
    Takes a list of graphs and creates batches for efficient GPU processing.
    
    Args:
        batch_size: Number of graphs per batch
        shuffle: Whether to shuffle graphs
    """
    
    def __init__(self, batch_size: int = 32, shuffle: bool = False):
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def create_batches(self, graphs: List[Data]) -> List[Batch]:
        """
        Create batches from a list of graphs.
        
        Args:
            graphs: List of PyTorch Geometric Data objects
            
        Returns:
            List of batched graphs
        """
        if self.shuffle:
            import random
            random.shuffle(graphs)
        
        batches = []
        for i in range(0, len(graphs), self.batch_size):
            batch_graphs = graphs[i:i + self.batch_size]
            if batch_graphs:
                batch = Batch.from_data_list(batch_graphs)
                batches.append(batch)
        
        return batches
    
    def __call__(self, graphs: List[Data]) -> List[Batch]:
        """Make the loader callable."""
        return self.create_batches(graphs)


def write_smiles_file(smiles_list: List[str], filepath: str) -> None:
    """
    Write SMILES strings to file.
    
    Args:
        smiles_list: List of SMILES strings
        filepath: Output file path
    """
    with open(filepath, "w") as f:
        for smiles in smiles_list:
            f.write(f"{smiles}\n")


def read_smiles_file(filepath: str) -> List[str]:
    """
    Read SMILES strings from file.
    
    Args:
        filepath: Input file path
        
    Returns:
        List of SMILES strings
    """
    smiles_list = []
    with open(filepath, "r") as f:
        for line in f:
            smiles = line.strip()
            if smiles:
                smiles_list.append(smiles)
    return smiles_list
