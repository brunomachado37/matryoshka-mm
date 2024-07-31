#!/bin/bash
#SBATCH --job-name=M3_droid_train
#SBATCH --output=results_eval/logs/results_eval_pick_velocities_5000_episodes_bs_1.%j.out
#SBATCH --error=results_eval/logs/results_eval_pick_velocities_5000_episodes_bs_1.%j.err

#SBATCH --partition=gpu
#SBATCH --gres=gpu:quadro_rtx_6000:1
#SBATCH --mem=80G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 4-00:00:00


echo ${SLURM_NODELIST}

source ~/.bashrc
conda activate M3_RLDS

python eval_pick_velocities.py