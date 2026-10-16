#!/bin/bash
#SBATCH --job-name=prep_capture24
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

source /var/scratch/$USER/miniconda3/etc/profile.d/conda.sh
conda activate /var/scratch/$USER/envs/adaptivity

echo "Running with Python : $(which python)"
echo "Host                : $(hostname)"
echo "Allocated CPU cores : $SLURM_CPUS_PER_TASK"

python prepros-das5.py

echo "Job finished at $(date)"