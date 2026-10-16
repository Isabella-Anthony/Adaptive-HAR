#!/bin/bash
#SBATCH --job-name=tbase
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpunode
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=./logs/baseline/train_%j.out
#SBATCH --error=./logs/baseline/train_%j.err

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
python -u train_baseline.py

echo "Job finished at $(date)"





