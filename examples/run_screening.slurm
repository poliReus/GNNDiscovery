#!/bin/bash
# Example SLURM job script for running HPC Drug Screening on a cluster.
#
# Usage:
#     sbatch run_screening.slurm
#
#SBATCH --job-name=drug-screening
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --output=drug_screening_%j.out
#SBATCH --error=drug_screening_%j.err

# Load required modules (adjust for your cluster)
module load python/3.9
module load cuda/11.8
module load openmpi/4.1.0

# Activate virtual environment
source venv/bin/activate

# Print job information
echo "Job started at: $(date)"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Number of tasks: $SLURM_NTASKS"
echo "Working directory: $(pwd)"

# Set MPI variables
export OMPI_MCA_btl_openib_allow_ib=1

# Run the distributed screening
mpirun -np $SLURM_NTASKS python main.py \
    --input /path/to/molecules.smi \
    --output /path/to/predictions.csv \
    --model-path models/pretrained_gcn.pt \
    --batch-size 2048 \
    --chunk-size 50000 \
    --use-gpu \
    --dynamic-balancing \
    --log-level INFO

echo "Job finished at: $(date)"
