"""评估模块：多进程并行对战，与训练管线对齐。

训练：N 个 SubprocVecEnv，每个 env 绑定一个对手 spec
评估：N 个 ProcessPoolExecutor，每个进程绑定一个对手 spec

用法：
    from ai.train.eval import evaluate

    results = evaluate(
        agent_path=".../ckpt_100000",
        opponent_specs=[{"type":"policy","player_id":2,"path":"..."}, ...],  # 每个 env 一个 spec
        scenario="two_players/vsbaseline",
        episodes_per=50,
    )
    win_rate = aggregate_win_rate(results)
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

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
    episodes_per_opponent: int,
) -> list[EvalResult]:
    """N 进程并行评估，每个对手跑 episodes_per_opponent 局（由分配给它的进程平分）。

    与训练对齐：opponent_specs 是 n_envs 个 spec 的扁平列表，同种对手连续排列。
    每个对手分配给 per = n_envs / n_opponents 个进程，每个进程跑
    episodes_per_opponent / per 局。

    Args:
        agent_path: checkpoint 路径（不含 .zip 后缀）
        opponent_specs: 每个 eval 进程的对手描述（长度 = eval_n_envs，同种对手连续）
        scenario: env 配置名
        episodes_per_opponent: 每种对手的总评估局数

    Returns:
        每个进程一个 EvalResult（长度 = len(opponent_specs)）
    """
    n = len(opponent_specs)
    if n == 0:
        return []

    n_opponents = _count_unique(opponent_specs)
    per = n // n_opponents  # 进程数 / 对手种类
    episodes_per_env = episodes_per_opponent // per

    if n == 1:
        return [_evaluate_one(agent_path, opponent_specs[0], scenario, episodes_per_env)]

    # 用 spawn 上下文避免与训练 SubprocVecEnv 的 forkserver 冲突
    ctx = mp.get_context("spawn")
    results = [None] * n
    with ProcessPoolExecutor(max_workers=n, mp_context=ctx) as executor:
        futures = {
            executor.submit(_evaluate_one, agent_path, spec, scenario, episodes_per_env): i
            for i, spec in enumerate(opponent_specs)
        }
        for f in futures:
            i = futures[f]
            try:
                results[i] = f.result()
            except Exception as e:
                logging.warning("[Eval] env[%d] failed: %s", i, e)
                results[i] = EvalResult(
                    wins=0, losses=0, draws=0, episodes=0,
                    win_rate=0.0, avg_turns=None,
                    opponent_spec=opponent_specs[i],
                )

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

def _count_unique(specs: list[dict]) -> int:
    """统计扁平 spec 列表中不同的对手种类数。"""
    seen: set[tuple] = set()
    for s in specs:
        key = tuple(sorted(s.items()))
        seen.add(key)
    return len(seen)


def _evaluate_one(
    agent_path: str,
    opponent_spec: dict,
    scenario: str,
    episodes: int,
) -> EvalResult:
    """单个进程：创建 env → 加载模型 → 跑 episodes 局。"""
    env = LwgEnv(scenario)
    env.set_opponent(opponent_spec)
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
