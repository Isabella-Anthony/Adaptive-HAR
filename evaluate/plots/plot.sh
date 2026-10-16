#!/bin/bash

#SBATCH --job-name=plot
#SBATCH --partition=defq
#SBATCH --time=00:10:00
#SBATCH --output=./logs/plot_%j.out
module load cuda12.6/toolkit/12.6

source /var/scratch/bde2601/miniconda3/etc/profile.d/conda.sh
conda activate /var/scratch/bde2601/envs/adap_py310


python -u plot_battery_simulation.py 
python -u plot_pareto.py 