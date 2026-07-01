#!/usr/bin/env bash

# If using virtualenv, activate it (replace path if needed)
# source E:/ASR/.venv/Scripts/activate

# Dataset name
data='facestar_whisper'

# Absolute paths to your train and val .pt files
train_path='E:\ASR\facestar_whisper\formatted_json\data\mciro_train_whisper_tiny.pt'
test_path='E:\ASR\facestar_whisper\formatted_json\data\mciro_test_whisper_tiny.pt'

# Path to TinyLlama checkpoint
checkpoint_dir='C:/TinyLlama-1.1B-3T'

# Run LipGER fine-tuning
python finetune/lipger.py \
       --data ${data} \
       --train_path ${train_path} \
       --val_path ${val_path} \
       --checkpoint_dir ${checkpoint_dir} \
       --device cuda \
       --batch_size 2 \
       --epochs 2
