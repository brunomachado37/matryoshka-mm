#!/bin/bash
#SBATCH --job-name=check_len
#SBATCH --output=logs/check_len.%j.out
#SBATCH --error=logs/check_len.%j.err

#SBATCH --mem=80G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH -t 100:00:00


echo ${SLURM_NODELIST}

module load pytorch/py3/2.0.0

python check_step_lengths.py
