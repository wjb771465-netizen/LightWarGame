#!/usr/bin/env bash
# 对战随机对手训练
# 用法：bash ai/train/train_vsrandom.sh [额外参数]
set -euo pipefail

SCENARIO="1v1/vsbaseline"
EXP_NAME="vsrandom_mlp256"

cd "$(dirname "$0")/../../.."

CONDA_LIB="$(conda info --base)/envs/chinese_war_game/lib"
conda run --no-capture-output -n chinese_war_game \
  env LD_LIBRARY_PATH="$CONDA_LIB:${LD_LIBRARY_PATH:-}" PYTHONUNBUFFERED=1 \
  python -m ai.train \
  --scenario "$SCENARIO" \
  --exp-name "$EXP_NAME" \
  --total-timesteps 5000000 \
  --n-steps 4096 \
  --batch-size 64 \
  --n-epochs 10 \
  --lr 3e-4 \
  --gamma 0.99 \
  --gae-lambda 0.97 \
  --clip-range 0.2 \
  --net-arch 256 256 \
  --checkpoint-freq 100000 \
  --use-eval \
  --eval-episodes 20 \
  --seed 42 \
  --wandb \
  "$@"
