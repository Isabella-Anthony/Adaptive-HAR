#!/bin/bash

#SBATCH --job-name=eval_baseline
#SBATCH --partition=defq
#SBATCH --constraint=gpunode
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=test_logs/baseline/eval_%j.out
module load cuda12.6/toolkit/12.6

source /var/scratch/bde2601/miniconda3/etc/profile.d/conda.sh
conda activate /var/scratch/bde2601/envs/adaptivity


python -u eval_baseline.py