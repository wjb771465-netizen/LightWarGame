#!/usr/bin/env bash
# 地区自博弈训练（首跑：河北 vs 江西，验证可行性）
# 用法：bash ai/train/scripts/train_region_selfplay.sh [额外参数]
set -euo pipefail

SCENARIO="1v1/region_selfplay"
EXP_NAME="region_selfplay_4_20"

cd "$(dirname "$0")/../../.."

CONDA_LIB="$(conda info --base)/envs/chinese_war_game/lib"
conda run --no-capture-output -n chinese_war_game \
  env LD_LIBRARY_PATH="$CONDA_LIB:${LD_LIBRARY_PATH:-}" PYTHONUNBUFFERED=1 \
  python -m ai.train \
  --scenario               "$SCENARIO" \
  --exp-name               "$EXP_NAME" \
  --region-self-play \
  --region-self-play-regions 4,20 \
  --region-pool-history    3 \
  --total-timesteps        500000 \
  --n-steps                2048 \
  --batch-size             512 \
  --n-epochs               10 \
  --lr                     3e-4 \
  --gamma                  0.99 \
  --gae-lambda             0.97 \
  --clip-range             0.2 \
  --net-arch               256 256 \
  --checkpoint-freq        25000 \
  --win-rate-window        200 \
  --seed                   42 \
  --parallel-regions       2 \
  --n-training-threads     4 \
  --wandb \
  "$@"
