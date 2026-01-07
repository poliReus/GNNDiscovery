"""
Setup script for HPC Drug Screening package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="hpc-drug-screening",
    version="0.1.0",
    author="GNNDiscovery Team",
    author_email="contact@gnndiscovery.org",
    description="High-Performance Computing for Drug Discovery using Graph Neural Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/poliReus/GNNDiscovery",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "mpi4py>=3.1.0",
        "rdkit>=2023.3.1",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "pylint>=2.17.0",
        ],
        "gpu": [
            "torch-scatter>=2.1.0",
            "torch-sparse>=0.6.17",
            "torch-cluster>=1.6.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "hpc-drug-screen=main:main",
        ],
    },
)
