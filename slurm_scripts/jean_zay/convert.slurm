#!/bin/bash
#SBATCH --job-name=droid2torch
#SBATCH --output=logs/droid2torch.%j.out
#SBATCH --error=logs/droid2torch.%j.err

#SBATCH -A uli@cpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -t 40:00:00
#SBATCH --qos=qos_cpu-t4 

echo ${SLURM_NODELIST}

module load pytorch-gpu/py3/2.0.0
python convert2Pickle_velocities.py
