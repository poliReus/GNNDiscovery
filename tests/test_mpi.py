"""
Unit tests for MPI utilities.
"""

import pytest

# Note: These tests require mpi4py and should be run with mpirun
# Example: mpirun -np 4 pytest tests/test_mpi.py


def test_imports():
    """Test that MPI utilities can be imported."""
    try:
        from hpc_drug_screening.utils import (
            is_master,
            get_rank,
            get_size,
            setup_logging,
        )
        assert True
    except ImportError:
        pytest.skip("MPI utilities not available")


def test_get_rank():
    """Test get_rank function."""
    try:
        from hpc_drug_screening.utils import get_rank
        
        rank = get_rank()
        assert isinstance(rank, int)
        assert rank >= 0
    except ImportError:
        pytest.skip("MPI not available")


def test_get_size():
    """Test get_size function."""
    try:
        from hpc_drug_screening.utils import get_size
        
        size = get_size()
        assert isinstance(size, int)
        assert size >= 1
    except ImportError:
        pytest.skip("MPI not available")


def test_is_master():
    """Test is_master function."""
    try:
        from hpc_drug_screening.utils import is_master, get_rank
        
        rank = get_rank()
        master = is_master(rank)
        
        assert isinstance(master, bool)
        assert master == (rank == 0)
    except ImportError:
        pytest.skip("MPI not available")


def test_setup_logging():
    """Test setup_logging function."""
    try:
        from hpc_drug_screening.utils import setup_logging
        import logging
        
        logger = setup_logging("INFO")
        assert isinstance(logger, logging.Logger)
        assert logger.level == logging.INFO
        
    except ImportError:
        pytest.skip("MPI not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
