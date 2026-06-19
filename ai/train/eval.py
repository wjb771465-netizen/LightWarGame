"""评估模块：SubprocVecEnv 批量并行对战，与训练推理对齐。

训练：N 个 SubprocVecEnv，每个 env 绑定一个对手 spec
评估：同结构——N 个 env → 1 次 batched GNN forward → N 个 action

用法：
    from ai.train.eval import evaluate

    results = evaluate(
        agent_path=".../ckpt_100000",
        opponent_specs=[{"type":"policy","player_id":2,"path":"..."}, ...],
        scenario="two_players/vsbaseline",
        episodes_per=50,
    )
    win_rate = aggregate_win_rate(results)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from ai.algos.policy import SB3Policy
from ai.envs.env import LwgEnv


@dataclass
class EvalResult:
    """单个 eval env 的对局结果。"""
    wins: int
    losses: int
    draws: int
    episodes: int
    win_rate: float
    avg_turns: float | None
    opponent_spec: dict = field(repr=False)


def evaluate(
    agent_path: str,
    opponent_specs: list[dict],
    scenario: str,
    episodes_per_env: int,
    agent_capital: int | None = None,
) -> list[EvalResult]:
    """SubprocVecEnv batched 评估，一次 GNN forward 处理所有 env。

    Args:
        agent_path: checkpoint 路径（不含 .zip 后缀）
        opponent_specs: 每个 env 的对手描述（长度 = N）
        scenario: env 配置名
        episodes_per_env: 每个 env 跑的局数

    Returns:
        每个 env 一个 EvalResult（长度 = len(opponent_specs)）
    """
    n = len(opponent_specs)
    if n == 0:
        return []

    if n == 1:
        return [_evaluate_one(agent_path, opponent_specs[0], scenario, episodes_per_env, agent_capital)]

    agent = SB3Policy(path=agent_path)

    def _make_env(i: int):
        def _init():
            env = LwgEnv(scenario)
            spec = opponent_specs[i]
            env.set_opponent(spec)
            opp_cap = spec.get("opp_region")
            if agent_capital is not None and opp_cap is not None:
                env.set_capitals(agent_capital, opp_cap)
            return env
        return _init

    venv = VecMonitor(
        SubprocVecEnv([_make_env(i) for i in range(n)]),
        info_keywords=("win", "turn"),
    )

    wins = [0] * n
    losses = [0] * n
    draws = [0] * n
    turn_sums = [0] * n
    episode_counts = [0] * n

    obs = venv.reset()
    masks = venv.env_method("action_masks")

    while min(episode_counts) < episodes_per_env:
        action_masks = np.stack(masks)  # (n_envs, action_dim)
        actions, _ = agent._model.predict(obs, action_masks=action_masks, deterministic=True)

        obs, _rewards, dones, infos = venv.step(actions)
        masks = venv.env_method("action_masks")

        for i in range(n):
            if dones[i] and episode_counts[i] < episodes_per_env:
                episode_counts[i] += 1
                info = infos[i]
                turn_sums[i] += info.get("turn", 0)
                if info.get("win", 0.0) == 1.0:
                    wins[i] += 1
                else:
                    losses[i] += 1

    venv.close()

    results = []
    for i in range(n):
        eps = episode_counts[i]
        results.append(EvalResult(
            wins=wins[i],
            losses=losses[i],
            draws=draws[i],
            episodes=eps,
            win_rate=wins[i] / eps if eps > 0 else 0.0,
            avg_turns=turn_sums[i] / eps if eps > 0 else None,
            opponent_spec=opponent_specs[i],
        ))

    return results


def aggregate_win_rate(results: list[EvalResult]) -> float:
    """总胜率（跨所有 eval env 汇总）。"""
    valid = [r for r in results if r.episodes > 0]
    if not valid:
        return 0.0
    return sum(r.wins for r in valid) / sum(r.episodes for r in valid)


def aggregate_avg_turns(results: list[EvalResult]) -> float | None:
    """总平均回合数（跨所有 eval env 汇总）。"""
    valid = [r for r in results if r.avg_turns is not None and r.episodes > 0]
    if not valid:
        return None
    return sum(r.avg_turns * r.episodes for r in valid) / sum(r.episodes for r in valid)


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _evaluate_one(
    agent_path: str,
    opponent_spec: dict,
    scenario: str,
    episodes: int,
    agent_capital: int | None = None,
) -> EvalResult:
    """单个进程：创建 env → 加载模型 → 跑 episodes 局。"""
    env = LwgEnv(scenario)
    env.set_opponent(opponent_spec)
    opp_cap = opponent_spec.get("opp_region")
    if agent_capital is not None and opp_cap is not None:
        env.set_capitals(agent_capital, opp_cap)
    agent = SB3Policy(path=agent_path)

    wins = 0
    losses = 0
    draws = 0
    total_turns = 0

    for _ in range(episodes):
        obs, _ = env.reset()
        while True:
            mask = env.action_masks()
            action = agent.predict(obs, mask, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                turn = info.get("turn", 0)
                total_turns += int(turn) if turn else 0
                if terminated:
                    if info.get("win", 0.0) == 1.0:
                        wins += 1
                    else:
                        losses += 1
                else:
                    draws += 1
                break

    return EvalResult(
        wins=wins,
        losses=losses,
        draws=draws,
        episodes=episodes,
        win_rate=wins / episodes if episodes > 0 else 0.0,
        avg_turns=total_turns / episodes if episodes > 0 else None,
        opponent_spec=opponent_spec,
    )
