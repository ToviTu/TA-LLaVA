#!/bin/bash

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path google/gemma-2-2b-it \
    --version gemma_2 \
    --data_path /ib-scratch/chenguang03/vision_share/datasets/llava/blip_laion_cc_sbu_558k.json \
    --image_folder /ib-scratch/chenguang03/vision_share/datasets/llava/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type linear \
    --tune_mm_mlp_adapter True \
    --num_learnable_tokens 32 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /ib-scratch/chenguang03/vision_share/models/llava-gemma2-pretrain-t \
    --num_train_epochs 2 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb