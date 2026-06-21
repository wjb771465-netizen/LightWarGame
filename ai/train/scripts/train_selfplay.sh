#!/usr/bin/env bash
# 自博弈训练 — PFSP（优先虚构自博弈），progress 优先采样 + ELO 门控入池
# 随机首都 + 全邻接矩阵 + rule 冷启动
# 用法：bash ai/train/scripts/train_selfplay.sh [额外参数]
set -euo pipefail

SCENARIO="duel/selfplay"
EXP_NAME="pfsp_mlp512_randcap"

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
  --self-play-pool-size 8 \
  --self-play-initial-opponent rule \
  --pool-sampling-strategy progress \
  --use-eval \
  --n-envs 8 \
  --n-opponents 4 \
  --eval-n-envs 8 \
  --eval-n-opponents 4 \
  --eval-episodes 20 \
  --eval-opponent random,rule,fsm \
  --total-timesteps 20000000 \
  --n-steps 2048 \
  --batch-size 512 \
  --n-epochs 8 \
  --lr 1e-4 \
  --gamma 0.99 \
  --gae-lambda 0.97 \
  --clip-range 0.2 \
  --net-arch 512 512 256 \
  --checkpoint-freq 50000 \
  --seed 42 \
  --wandb \
  "$@"
