"""
MPI Utility Functions
Helper functions for MPI-based distributed computing.
"""

import logging
import sys
from typing import Optional

from mpi4py import MPI


def get_rank() -> int:
    """Get the rank of the current MPI process.
    
    Returns:
        Rank of the current process (0 to size-1)
    """
    return MPI.COMM_WORLD.Get_rank()


def get_size() -> int:
    """Get the total number of MPI processes.
    
    Returns:
        Total number of MPI processes
    """
    return MPI.COMM_WORLD.Get_size()


def is_master(rank: Optional[int] = None) -> bool:
    """Check if the current process is the master (rank 0).
    
    Args:
        rank: MPI rank (if None, gets current rank)
        
    Returns:
        True if this is the master process
    """
    if rank is None:
        rank = get_rank()
    return rank == 0


def setup_logging(
    log_level: str = "INFO",
    rank: Optional[int] = None
) -> logging.Logger:
    """Setup logging for MPI processes.
    
    Each process gets its own logger with rank-specific formatting.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        rank: MPI rank (if None, gets current rank)
        
    Returns:
        Configured logger instance
    """
    if rank is None:
        rank = get_rank()
    
    # Create logger
    logger = logging.getLogger(f"hpc_drug_screening_rank_{rank}")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatter with rank information
    if is_master(rank):
        formatter = logging.Formatter(
            "[MASTER] %(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    else:
        formatter = logging.Formatter(
            f"[WORKER-{rank}] %(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


def barrier():
    """Synchronization barrier for all MPI processes."""
    MPI.COMM_WORLD.Barrier()


def broadcast(data, root: int = 0):
    """Broadcast data from root to all processes.
    
    Args:
        data: Data to broadcast
        root: Root rank (default: 0)
        
    Returns:
        Broadcasted data
    """
    return MPI.COMM_WORLD.bcast(data, root=root)


def scatter(data, root: int = 0):
    """Scatter data from root to all processes.
    
    Args:
        data: List of data to scatter (one element per process)
        root: Root rank (default: 0)
        
    Returns:
        Data for this process
    """
    return MPI.COMM_WORLD.scatter(data, root=root)


def gather(data, root: int = 0):
    """Gather data from all processes to root.
    
    Args:
        data: Data from this process
        root: Root rank (default: 0)
        
    Returns:
        List of data from all processes (only on root)
    """
    return MPI.COMM_WORLD.gather(data, root=root)


def allgather(data):
    """Gather data from all processes to all processes.
    
    Args:
        data: Data from this process
        
    Returns:
        List of data from all processes
    """
    return MPI.COMM_WORLD.allgather(data)


def send(data, dest: int, tag: int = 0):
    """Send data to another process.
    
    Args:
        data: Data to send
        dest: Destination rank
        tag: Message tag
    """
    MPI.COMM_WORLD.send(data, dest=dest, tag=tag)


def recv(source: int = MPI.ANY_SOURCE, tag: int = MPI.ANY_TAG):
    """Receive data from another process.
    
    Args:
        source: Source rank (or MPI.ANY_SOURCE)
        tag: Message tag (or MPI.ANY_TAG)
        
    Returns:
        Received data
    """
    return MPI.COMM_WORLD.recv(source=source, tag=tag)


def abort(errorcode: int = 1):
    """Abort all MPI processes.
    
    Args:
        errorcode: Exit code
    """
    MPI.COMM_WORLD.Abort(errorcode)
