#!/usr/bin/env bash
# GNN 骨架训练（GraphSAGE，固定首都）
# 用法：bash ai/train/scripts/train_gnn.sh [额外参数]
set -euo pipefail

SCENARIO="duel/vsbaseline_gnn_fixed"
EXP_NAME="gnn128"

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
  --use-gnn \
  --net-arch               128 \
  --gnn-hidden-channels    128 \
  --total-timesteps        5000000 \
  --n-steps                2048 \
  --batch-size             512 \
  --n-epochs               10 \
  --lr                     3e-4 \
  --gamma                  0.99 \
  --gae-lambda             0.97 \
  --clip-range             0.2 \
  --checkpoint-freq        16384 \
  --win-rate-window        200 \
  --use-eval \
  --eval-episodes          20 \
  --seed                   42 \
  --wandb \
  "$@"
