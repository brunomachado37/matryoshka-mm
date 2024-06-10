#!/bin/bash
#SBATCH --job-name=check_len
#SBATCH --output=logs/check_len.%j.out
#SBATCH --error=logs/check_len.%j.err

#SBATCH -A uli@cpu
#SBATCH --qos=qos_cpu-t4 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -t 10:00:00


echo ${SLURM_NODELIST}

module load pytorch/py3/2.0.0

python check_step_lengths.py
