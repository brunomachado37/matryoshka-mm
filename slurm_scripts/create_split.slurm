#!/bin/bash
#SBATCH --job-name=create_split
#SBATCH --output=logs/create_split.%j.out
#SBATCH --error=logs/create_split.%j.err

#SBATCH -A uli@cpu
#SBATCH --qos=qos_cpu-dev 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -t 1:00:00


echo ${SLURM_NODELIST}

module load pytorch-gpu/py3/2.0.0

python create_droid_split.py 
