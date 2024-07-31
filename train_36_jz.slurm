#!/bin/bash
#SBATCH --job-name=M3_fractal_train
#SBATCH --output=logs/fractal/train-36.%j.out
#SBATCH --error=logs/fractal/train-36.%j.err

#SBATCH -A uli@a100
#SBATCH -C a100
#SBATCH --gres=gpu:8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH -t 20:00:00
#SBATCH --qos=qos_gpu-t3
#SBATCH --exclusive

echo ${SLURM_NODELIST}

module purge
module load cpuarch/amd
module load pytorch-gpu/py3/2.1.1

torchrun --master_port 28012 --nproc_per_node 8 --nnodes 1 llava/train/train_mem.py \
    --model_name_or_path $WORK/.cache/huggingface/hub/models--mucai--llava-v1.5-7b-m3/snapshots/629faba2a479cf94616e664c4a81f6892b9a4214/ \
    --tokenizer_path ./tokenizer \
    --version v1 \
    --data_path $HOME/datasets/oxe/ \
    --vision_tower $WORK/.cache/huggingface/hub/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1/ \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $WORK/M3/models/fractal/test_dummy-token-scale-36 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 100 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --matryoshka_vis_token_scale 36 \
    --report_to wandb
