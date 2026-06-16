#!/usr/bin/env bash
# 地区自博弈训练 — PFSP（优先虚构自博弈），progress 优先采样 + ELO 门控入池
# 用法：bash ai/train/scripts/train_region_selfplay.sh [额外参数]
set -euo pipefail

SCENARIO="duel/region_selfplay"
EXP_NAME="region_pfsp_4_20"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/snapshot.sh"

cd "$SCRIPT_DIR/../../.."

TS=$(date +%Y%m%d_%H%M%S)
SAVE_DIR="ai/train/results/$SCENARIO/$EXP_NAME/run_$TS"
git_snapshot "$SCENARIO/$EXP_NAME/run_$TS"

CONDA_LIB="$(conda info --base)/envs/chinese_war_game/lib"
conda run --no-capture-output -n chinese_war_game \
  env LD_LIBRARY_PATH="$CONDA_LIB:${LD_LIBRARY_PATH:-}" PYTHONUNBUFFERED=1 \
  python -m ai.train \
  --scenario               "$SCENARIO" \
  --exp-name               "$EXP_NAME" \
  --save-dir               "$SAVE_DIR" \
  --region-self-play \
  --region-self-play-regions 4,20 \
  --self-play-pool-size    20 \
  --self-play-initial-opponent rule \
  --pool-sampling-strategy progress \
  --use-eval \
  --eval-episodes          20 \
  --eval-opponent          random,rule,fsm \
  --eval-opponent-freq     20 \
  --eval-n-envs             6 \
  --total-timesteps        10000000 \
  --n-steps                2048 \
  --batch-size             512 \
  --n-epochs               10 \
  --lr                     3e-4 \
  --gamma                  0.99 \
  --gae-lambda             0.97 \
  --clip-range             0.2 \
  --net-arch               512 512 256 \
  --checkpoint-freq        50000 \
  --win-rate-window        200 \
  --seed                   42 \
  --parallel-regions       2 \
  --n-training-threads     4 \
  --wandb \
  "$@"
