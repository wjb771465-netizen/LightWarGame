#!/usr/bin/env bash
# 对战随机对手训练（随机首都 + 全邻接矩阵）
# 用法：bash ai/train/scripts/train_vsrandom.sh [额外参数]
set -euo pipefail

SCENARIO="duel/vsbaseline_randcap"
EXP_NAME="vsrandom_mlp512_randcap"

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
  --scenario "$SCENARIO" \
  --exp-name "$EXP_NAME" \
  --save-dir "$SAVE_DIR" \
  --total-timesteps 10000000 \
  --n-steps 2048 \
  --batch-size 512 \
  --n-epochs 10 \
  --lr 1e-4 \
  --gamma 0.99 \
  --gae-lambda 0.97 \
  --clip-range 0.2 \
  --net-arch 512 512 256 \
  --checkpoint-freq 50000 \
  --use-eval \
  --eval-episodes 20 \
  --seed 42 \
  --wandb \
  "$@"
