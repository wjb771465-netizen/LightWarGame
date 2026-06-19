"""评估模块：复用已有 VecEnv 做 batched 推理。

调用方在 evaluate() 之前负责设置对手和首都：
    for i, spec in enumerate(specs):
        venv.env_method("set_opponent", spec, indices=[i])
    results = evaluate(ckpt, venv, episodes_per_env, specs)

用法：
    from ai.train.eval import evaluate, aggregate_win_rate

    results = evaluate("path/to/ckpt", venv, 20, specs)
    win_rate = aggregate_win_rate(results)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ai.algos.policy import SB3Policy


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
    venv: object,
    episodes_per_env: int,
    opponent_specs: list[dict],
) -> list[EvalResult]:
    """复用已有 VecEnv 做 batched 评估，一次 GNN forward 处理所有 env。

    调用方负责在评估前通过 env_method 设置每个 env 的对手和首都。

    Args:
        agent_path: checkpoint 路径（不含 .zip 后缀）
        venv: 已配置对手的 VecEnv
        episodes_per_env: 每个 env 跑的局数
        opponent_specs: 每 env 一个 spec，用于 EvalResult 元数据

    Returns:
        每个 env 一个 EvalResult（长度 = venv.num_envs）
    """
    n = venv.num_envs
    if n == 0:
        return []

    agent = SB3Policy(path=agent_path)

    wins = [0] * n
    losses = [0] * n
    turn_sums = [0] * n
    episode_counts = [0] * n

    obs = venv.reset()
    masks = venv.env_method("action_masks")

    while min(episode_counts) < episodes_per_env:
        action_masks = np.stack(masks)
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

    results = []
    for i in range(n):
        eps = episode_counts[i]
        results.append(EvalResult(
            wins=wins[i],
            losses=losses[i],
            draws=0,
            episodes=eps,
            win_rate=wins[i] / eps if eps > 0 else 0.0,
            avg_turns=turn_sums[i] / eps if eps > 0 else None,
            opponent_spec=opponent_specs[i % len(opponent_specs)],
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
