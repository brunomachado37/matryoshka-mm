#!/bin/bash
#SBATCH --job-name=M3_droid_train
#SBATCH --output=logs/train.%j.out
#SBATCH --error=logs/train.%j.err

#SBATCH -A uli@a100
#SBATCH -C a100
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH -t 00:10:00
#SBATCH --qos=qos_gpu-dev

echo ${SLURM_NODELIST}

module load cpuarch/amd
module load pytorch-gpu/py3/2.1.1

torchrun --nproc_per_node 2 llava/train/train_mem.py \
    --model_name_or_path "/gpfswork/rech/uli/ujf38zl/.cache/huggingface/hub/models--mucai--llava-v1.5-7b-m3/snapshots/629faba2a479cf94616e664c4a81f6892b9a4214/" \
    --tokenizer_path ./tokenizer \
    --version v1 \
    --data_path /gpfsscratch/rech/uli/ujf38zl/DROID/train \
    --eval_path /gpfsscratch/rech/uli/ujf38zl/DROID/eval \
    --vision_tower "/gpfswork/rech/uli/ujf38zl/.cache/huggingface/hub/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1/" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./models/test_dummy \
    --num_train_epochs 10 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 100 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --matryoshka_vis_token_scale 36 \
    --report_to wandb \
    --cache_dir /gpfswork/rech/uli/ujf38zl/.cache/huggingface
