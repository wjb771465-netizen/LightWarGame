#!/usr/bin/env bash
# 对战规则对手训练
# 用法：bash scripts/train_vsrule.sh [额外参数]
set -euo pipefail

SCENARIO="two_players/vsrule"
EXP_NAME="vsrule_mlp256"

cd "$(dirname "$0")/.."

conda run -n chinese_war_game python -m ai.trainer \
  --scenario "$SCENARIO" \
  --exp-name "$EXP_NAME" \
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
  --eval-freq 50000 \
  --eval-episodes 100 \
  --seed 42 \
  "$@"
