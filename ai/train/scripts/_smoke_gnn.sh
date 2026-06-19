#!/usr/bin/env bash
# GNN 轻量 smoke test：快速验证管线能跑、策略能学
# 用法：bash ai/train/scripts/_smoke_gnn.sh
set -euo pipefail

SCENARIO="${1:-duel/vsbaseline_gnn_fixed}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../../.."

CONDA_LIB="$(conda info --base)/envs/chinese_war_game/lib"
conda run --no-capture-output -n chinese_war_game \
  env LD_LIBRARY_PATH="$CONDA_LIB:${LD_LIBRARY_PATH:-}" PYTHONUNBUFFERED=1 \
  python -m ai.train \
  --scenario               "$SCENARIO" \
  --use-gnn \
  --gnn-hidden-channels    128 \
  --net-arch               256 128 \
  --total-timesteps        32768 \
  --n-steps                2048 \
  --batch-size             512 \
  --n-epochs               40 \
  --lr                     3e-4 \
  --gamma                  0.99 \
  --gae-lambda             0.97 \
  --clip-range             0.2 \
  --checkpoint-freq        16384 \
  --seed                   42 \
  "$@"
