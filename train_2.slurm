#!/bin/bash
#SBATCH --job-name=M3_droid_train
#SBATCH --output=logs/single_episode/train.%j.out
#SBATCH --error=logs/single_episode/train.%j.err

#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH -t 4-00:00:00


echo ${SLURM_NODELIST}

source ~/.bashrc
conda activate M3

python llava/train/train_mem.py \
    --model_name_or_path mucai/llava-v1.5-7b-m3 \
    --tokenizer_path ./tokenizer \
    --version v1 \
    --data_path ../datasets/droid_torch_dc/train_2 \
    --eval_path ../datasets/droid_torch_dc/eval_2 \
    --tokenizer_path ./tokenizer \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/single_episode/test_last \
    --num_train_epochs 10 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "epoch" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --matryoshka_vis_token_scale 36 \
    --report_to wandb