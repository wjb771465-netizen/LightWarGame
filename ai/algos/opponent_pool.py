"""策略池：管理历史 checkpoint，支持多种淘汰模式与采样策略。

池负责 entry 的增删查；采样方法提取 entry 字段后委托 ai.algos.sampling 的纯数学函数。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PoolEntry:
    """池中一条策略记录。"""
    path: str                 # checkpoint 文件路径
    step: int                 # 训练步数（同时用于 time 淘汰：step 最小 = 最旧）
    elo: float | None = None  # ELO 评分，后续 PFSP 阶段填充


class OpponentPool:
    """管理历史 checkpoint 的索引，支持多种淘汰模式。"""

    def __init__(self, max_size: int, eviction_mode: str = "time") -> None:
        if eviction_mode not in ("time", "elo"):
            raise ValueError(f"Unknown eviction_mode: {eviction_mode!r}, expected 'time' or 'elo'")
        self._max_size = max_size
        self._mode = eviction_mode
        self._entries: dict[int, PoolEntry] = {}

    # ------------------------------------------------------------------
    def add(self, path: str, step: int, elo: float | None = None,
            outcomes: list | None = None,
            ) -> tuple[PoolEntry | None, float, bool]:
        """注册新 checkpoint；池满时按 eviction_mode 淘汰一个。

        outcomes 为 None → 直接入池（向后兼容，--use-eval 关闭时）。
        outcomes 非空 → 每个元素应有 .opponent_spec(dict) / .wins / .draws / .episodes，
                        内部解析对手 step、算分、_update_elo，ELO 不退化 (>=prev) 才入池。

        Returns:
            (evicted_entry_or_None, final_elo, accepted)
        """
        if outcomes is not None:
            agent_elo = elo if elo is not None else 1200.0
            prev_elo = agent_elo
            for r in outcomes:
                opp_step = self._parse_step_from_spec(r.opponent_spec)
                if opp_step is None:
                    continue
                agent_elo, _ = self._update_elo(
                    opp_step, agent_elo, self._compute_score(r))
            if not self._should_accept(prev_elo, agent_elo):
                return None, agent_elo, False
            elo = agent_elo

        evicted: PoolEntry | None = None
        if len(self._entries) >= self._max_size:
            evicted = self._evict_one()
        self._entries[step] = PoolEntry(path=path, step=step, elo=elo)
        return evicted, elo if elo is not None else 1200.0, True

    def latest(self) -> PoolEntry | None:
        """最新（step 最大）的 entry。"""
        if not self._entries:
            return None
        return max(self._entries.values(), key=lambda e: e.step)

    def get(self, step: int) -> PoolEntry | None:
        """按训练步数取 entry。"""
        return self._entries.get(step)

    # ------------------------------------------------------------------
    # 采样策略
    # ------------------------------------------------------------------

    def sample_uniform(self) -> PoolEntry | None:
        """均匀随机采样。P(i) = 1/N（FSP）。"""
        if not self._entries:
            return None
        from ai.algos.sampling import uniform_probs

        entries = list(self._entries.values())
        probs = uniform_probs(len(entries))
        idx = int(np.random.choice(len(entries), p=probs))
        return entries[idx]

    def sample_progress(
        self, lam: float = 1.0, s: float = 100.0, D: float | None = None,
    ) -> PoolEntry | None:
        """进度优先：对 step 做 Logistic-Softmax，越新权重越高。

        Args:
            lam: 温度系数 λ
            s: logistic 缩放因子
            D: logistic 尺度，None 则自动取 (max_step - min_step) / 4，保底 1.0
        """
        if not self._entries:
            return None
        from ai.algos.sampling import logistic_softmax_probs

        entries = list(self._entries.values())
        steps = np.array([e.step for e in entries], dtype=np.float64)
        if D is None:
            D = float(max(1.0, (steps.max() - steps.min()) / 4.0))
        probs = logistic_softmax_probs(steps, lam=lam, s=s, D=D)
        idx = int(np.random.choice(len(entries), p=probs))
        return entries[idx]

    def sample_elo(self, lam: float = 1.0, s: float = 100.0) -> PoolEntry | None:
        """ELO 优先：对 elo 做 Logistic-Softmax（D=400）。
        ELO 全为 None 时退化为 uniform。
        """
        if not self._entries:
            return None
        from ai.algos.sampling import logistic_softmax_probs, uniform_probs

        entries = list(self._entries.values())
        has_elo = any(e.elo is not None for e in entries)
        if not has_elo:
            probs = uniform_probs(len(entries))
        else:
            elos = np.array([e.elo if e.elo is not None else 0.0 for e in entries], dtype=np.float64)
            probs = logistic_softmax_probs(elos, lam=lam, s=s, D=400.0)
        idx = int(np.random.choice(len(entries), p=probs))
        return entries[idx]

    # ------------------------------------------------------------------
    # ELO
    # ------------------------------------------------------------------

    @staticmethod
    def _should_accept(prev_elo: float, new_elo: float) -> bool:
        """ELO 不退化则接受入池。子类可覆写以替换门控策略。"""
        return new_elo >= prev_elo

    @staticmethod
    def _compute_score(r) -> float:
        """从评估结果计算 agent 实际得分（1=胜, 0.5=平, 0=负）。"""
        return (r.wins + 0.5 * r.draws) / r.episodes if r.episodes > 0 else 0.0

    @staticmethod
    def _parse_step_from_spec(spec: dict) -> int | None:
        """从对手 spec 的 path 中解析训练步数。非 policy 对手返回 None。"""
        if spec.get("type") != "policy":
            return None
        path = spec.get("path", "")
        try:
            return int(path.rsplit("ckpt_", 1)[-1])
        except (ValueError, IndexError):
            return None

    def _update_elo(self, opponent_step: int, agent_elo: float,
                   score: float, K: float = 32.0) -> tuple[float, float]:
        """标准 ELO 两两更新：根据一场对局的实际得分更新双方 ELO。

        1. 从池中查找对手当前 ELO（未设置则默认 1200）
        2. 计算预期得分 E = 1 / (1 + 10^((elo_opp - elo_agent) / 400))
        3. 更新: new_elo = old_elo + K * (score - E)
        4. 将对手新 ELO 写回池

        Args:
            opponent_step: 对手的训练步数，用于查找池中条目
            agent_elo: agent 当前 ELO
            score: agent 实际得分（1=胜, 0=负, 0.5=平）
            K: ELO K-factor（默认 32）

        Returns:
            (new_agent_elo, new_opponent_elo)
        """
        entry = self._entries.get(opponent_step)
        opp_elo = (entry.elo if entry.elo is not None else 1200.0) if entry else 1200.0

        expected = 1.0 / (1.0 + 10.0 ** ((opp_elo - agent_elo) / 400.0))

        new_agent_elo = agent_elo + K * (score - expected)
        new_opp_elo = opp_elo + K * (expected - score)

        if entry is not None:
            entry.elo = new_opp_elo

        return new_agent_elo, new_opp_elo

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries.values())

    def __contains__(self, path: str) -> bool:
        return any(e.path == path for e in self._entries.values())

    # ------------------------------------------------------------------
    def _evict_one(self) -> PoolEntry:
        if self._mode == "time":
            key = min(self._entries, key=lambda k: self._entries[k].step)
        else:  # "elo"
            key = min(self._entries, key=lambda k: self._entries[k].elo or 0.0)
        return self._entries.pop(key)
