#!/usr/bin/env bash
# 随机首都自博弈 + 课程学习：先打 rule 到 70% 胜率，再转入 PFSP 自博弈
# 用法：bash ai/train/scripts/train_selfplay.sh [额外参数]
set -euo pipefail

SCENARIO="duel/selfplay"
EXP_NAME="curriculum_rule07_randcap"

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
  --self-play \
  --self-play-pool-size 20 \
  --self-play-initial-opponent rule \
  --curriculum-win-rate 0.75 \
  --pool-sampling-strategy progress \
  --sampling-scale 50 \
  --use-eval \
  --eval-episodes 20 \
  --eval-opponent random,rule,fsm \
  --eval-n-envs 6 \
  --eval-opponent-freq 5 \
  --total-timesteps 20000000 \
  --n-steps 2048 \
  --batch-size 512 \
  --n-epochs 10 \
  --lr 1.5e-4 \
  --gamma 0.99 \
  --gae-lambda 0.97 \
  --clip-range 0.2 \
  --net-arch 512 512 256 \
  --checkpoint-freq 100000 \
  --win-rate-window 200 \
  --seed 42 \
  --wandb \
  "$@"
