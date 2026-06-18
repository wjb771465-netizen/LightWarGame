#!/usr/bin/env bash
# GNN 骨架训练（GraphSAGE，随机首都）
# 用法：bash ai/train/scripts/train_gnn.sh [额外参数]
set -euo pipefail

SCENARIO="duel/vsbaseline_no_adj"
EXP_NAME="gnn256x128"

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
  --net-arch               256 128 \
  --gnn-hidden-channels    128 \
  --total-timesteps        10000000 \
  --n-steps                2048 \
  --batch-size             256 \
  --n-epochs               10 \
  --lr                     2e-4 \
  --gamma                  0.99 \
  --gae-lambda             0.97 \
  --clip-range             0.2 \
  --checkpoint-freq        16384 \
  --win-rate-window        200 \
  --eval-opponent          random \
  --seed                   42 \
  --wandb \
  "$@"
