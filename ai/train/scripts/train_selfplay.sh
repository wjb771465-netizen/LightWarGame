#!/usr/bin/env bash
# 自博弈训练
# 用法：bash ai/train/train_selfplay.sh [额外参数]
set -euo pipefail

SCENARIO="1v1/selfplay"
EXP_NAME="selfplay_mlp256"

cd "$(dirname "$0")/../../.."

conda run --no-capture-output -n chinese_war_game env PYTHONUNBUFFERED=1 python -m ai.train \
  --scenario "$SCENARIO" \
  --exp-name "$EXP_NAME" \
  --self-play \
  --self-play-pool-size 5 \
  --self-play-initial-opponent random \
  --total-timesteps 5000000 \
  --n-steps 4096 \
  --batch-size 64 \
  --n-epochs 10 \
  --lr 3e-4 \
  --gamma 0.99 \
  --gae-lambda 0.97 \
  --clip-range 0.2 \
  --net-arch 256 256 \
  --checkpoint-freq 50000 \
  --seed 42 \
  --wandb \
  "$@"
