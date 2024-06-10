#!/bin/bash
#SBATCH --job-name=droid2torch
#SBATCH --output=logs/droid2torch.%j.out
#SBATCH --error=logs/droid2torch.%j.err

#SBATCH --mem=180G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH -t 100:00:00

echo ${SLURM_NODELIST}

module load pytorch/py3/2.0.0

python convert2PyTorch_episodes_folders.py
