#!/bin/bash
#SBATCH --job-name=write_meta
#SBATCH --output=logs/write_meta.%j.out
#SBATCH --error=logs/write_meta.%j.err

#SBATCH --mem=80G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH -t 100:00:00


echo ${SLURM_NODELIST}

module load pytorch/py3/2.0.0

python get_dataset_stats.py
