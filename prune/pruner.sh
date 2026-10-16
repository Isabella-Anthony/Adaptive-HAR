#!/bin/bash

#SBATCH --job-name=prune
#SBATCH --partition=defq
#SBATCH --constraint=gpunode
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=./logs/prune_%j.out
module load cuda12.6/toolkit/12.6

source /var/scratch/$USER/miniconda3/etc/profile.d/conda.sh
conda activate /var/scratch/$USER/envs/adaptivity


python -u prune.py