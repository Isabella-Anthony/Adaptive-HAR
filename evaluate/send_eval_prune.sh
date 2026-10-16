#!/bin/bash

#SBATCH --job-name=eval_pruned
#SBATCH --partition=defq
#SBATCH --constraint=gpunode
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=./logs/prune/eval_prune_%j.out
module load cuda12.6/toolkit/12.6

source /var/scratch/$USER/miniconda3/etc/profile.d/conda.sh
conda activate /var/scratch/$USER/envs/adaptivity


python -u eval_prune.py