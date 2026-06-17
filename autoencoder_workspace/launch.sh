#!/bin/bash

set -e

echo "Starting Autoencoder Hyperparameter Scan..."

LEARNING_RATES=(0.001 0.003 0.0005)
BOTTLENECKS=(2 3)

for LR in "${LEARNING_RATES[@]}"; do
  for BOT in "${BOTTLENECKS[@]}"; do
  
    RUN_NAME="ae_bot${BOT}_lr${LR}"
    
    echo "========================================================="
    echo "[SCAN] Running: $RUN_NAME"
    echo "========================================================="
    
    uv run training.py \
      --outfile $RUN_NAME \
      --learning_rate $LR \
      --bottleneck $BOT \
      --epochs 300 \
      --patience 30 \
      --save
        
  done
done

echo "========================================================="
echo "[SUCCESS] Hyperparameter scan complete!"