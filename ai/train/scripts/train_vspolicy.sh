#!/usr/bin/env bash
set -euo pipefail

SCENARIO="1v1/vspolicy"
EXP_NAME="vspolicy_500k"

cd "$(dirname "$0")/../../.."

conda run --no-capture-output -n chinese_war_game env PYTHONUNBUFFERED=1 python -m ai.train \
  --scenario "$SCENARIO" \
  --exp-name "$EXP_NAME" \
  --total-timesteps 5000000 \
  --n-steps 4096 \
  --batch-size 64 \
  --n-epochs 5 \
  --lr 1e-4 \
  --gamma 0.99 \
  --gae-lambda 0.97 \
  --clip-range 0.15 \
  --net-arch 256 256 \
  --checkpoint-freq 200000 \
  --eval-freq 100000 \
  --eval-episodes 100 \
  --seed 42 \
  --wandb \
  "$@"
