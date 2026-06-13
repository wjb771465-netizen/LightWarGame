#!/usr/bin/env bash
# 对战规则对手训练
# 用法：bash ai/train/train_vsrule.sh [额外参数]
set -euo pipefail

SCENARIO="duel/vsrule"
EXP_NAME="vsrule_mlp256"

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
  --total-timesteps 2000000 \
  --n-steps 2048 \
  --batch-size 64 \
  --n-epochs 10 \
  --lr 1e-4 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --clip-range 0.2 \
  --net-arch 256 256 \
  --checkpoint-freq 100000 \
  --use-eval \
  --eval-episodes 20 \
  --seed 42 \
  "$@"
