"""临时脚本：加载已知高胜率模型，复用 evaluate() 验证 eval 基础设施是否正常。"""
from __future__ import annotations

import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s")

MODEL_PATH = "/home/wjb/Workspace/LightWarGame/ai/train/results/duel/vsbaseline_randcap/vsrandom_mlp512_randcap/run_20260620_234014/final.zip"
SCENARIO = "duel/vsbaseline_randcap"

# ── 1. 建 VecEnv ──
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

def _make_env():
    import torch
    torch.set_num_threads(1)
    from ai.envs.env import LwgEnv
    return LwgEnv(SCENARIO)

eval_env = VecMonitor(make_vec_env(_make_env, n_envs=4, vec_env_cls=SubprocVecEnv))

# ── 2. 加载模型 ──
from ai.algos.policy import SB3Policy
agent = SB3Policy(path=MODEL_PATH)
print(f"Model loaded: obs_dim={agent.obs_dim}, max_players={agent.config.get('max_players')}")

# ── 3. 设置对手 ──
from ai.train.eval import evaluate, aggregate_win_rate

opponent_id = 2  # agent_id=1, opponent=2

# 对每种对手分别测
for opp_type in ["random", "rule", "fsm"]:
    specs = 4 * [{"type": opp_type, "player_id": opponent_id}]
    for i in range(4):
        eval_env.env_method("set_opponent", specs[i], indices=[i])

    results = evaluate(agent, eval_env, episodes_per_env=50, opponent_specs=specs)
    wr = aggregate_win_rate(results)
    turns = [r.avg_turns for r in results]
    print(f"vs_{opp_type}: win_rate={wr:.1%}  avg_turns={turns}")

eval_env.close()
print("Done.")
