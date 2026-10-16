#!/bin/bash
#SBATCH --job-name=fit
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpunode
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=./logs/adaptive/fit_%j.out
#SBATCH --error=./logs/adaptive/fit_%j.err

module purge
module load cuda12.6/toolkit/12.6

source /var/scratch/$USER/miniconda3/etc/profile.d/conda.sh
conda activate /var/scratch/$USER/envs/adaptivity

export PYTHONUNBUFFERED=1



echo "=========================================="
echo "Host Node           : $(hostname)"
echo "Active Python       : $(which python)"
echo "CUDA Module Loaded  : $(which nvcc 2>/dev/null)"
echo "NVIDIA SMI Path     : $(which nvidia-smi)"
echo "=========================================="

nvidia-smi

echo "===== PYTORCH CUDA VERIFICATION ====="
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
echo "======================================"

# 5. Run training
python -u tfit_early_exit.py

echo "Job finished at $(date)"





