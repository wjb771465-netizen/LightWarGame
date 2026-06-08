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
        self._entries: list[PoolEntry] = []

    # ------------------------------------------------------------------
    def add(self, path: str, step: int, elo: float | None = None) -> PoolEntry | None:
        """注册新 checkpoint；池满时按 eviction_mode 淘汰一个，返回被淘汰的 entry。"""
        evicted: PoolEntry | None = None
        if len(self._entries) >= self._max_size:
            evicted = self._evict_one()
        self._entries.append(PoolEntry(path=path, step=step, elo=elo))
        return evicted

    def latest(self) -> PoolEntry | None:
        """最新（step 最大）的 entry。"""
        if not self._entries:
            return None
        return max(self._entries, key=lambda e: e.step)

    def sample(self, index: int) -> PoolEntry | None:
        """按索引取 entry。"""
        if 0 <= index < len(self._entries):
            return self._entries[index]
        return None

    # ------------------------------------------------------------------
    # 采样策略
    # ------------------------------------------------------------------

    def sample_uniform(self) -> PoolEntry | None:
        """均匀随机采样。P(i) = 1/N（FSP）。"""
        if not self._entries:
            return None
        from ai.algos.sampling import uniform_probs

        probs = uniform_probs(len(self._entries))
        idx = int(np.random.choice(len(self._entries), p=probs))
        return self._entries[idx]

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

        steps = np.array([e.step for e in self._entries], dtype=np.float64)
        if D is None:
            D = float(max(1.0, (steps.max() - steps.min()) / 4.0))
        probs = logistic_softmax_probs(steps, lam=lam, s=s, D=D)
        idx = int(np.random.choice(len(self._entries), p=probs))
        return self._entries[idx]

    def sample_elo(self, lam: float = 1.0, s: float = 100.0) -> PoolEntry | None:
        """ELO 优先：对 elo 做 Logistic-Softmax（D=400）。
        ELO 全为 None 时退化为 uniform。
        """
        if not self._entries:
            return None
        from ai.algos.sampling import logistic_softmax_probs, uniform_probs

        has_elo = any(e.elo is not None for e in self._entries)
        if not has_elo:
            probs = uniform_probs(len(self._entries))
        else:
            elos = np.array([e.elo if e.elo is not None else 0.0 for e in self._entries], dtype=np.float64)
            probs = logistic_softmax_probs(elos, lam=lam, s=s, D=400.0)
        idx = int(np.random.choice(len(self._entries), p=probs))
        return self._entries[idx]

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, path: str) -> bool:
        return any(e.path == path for e in self._entries)

    # ------------------------------------------------------------------
    def _evict_one(self) -> PoolEntry:
        if self._mode == "time":
            idx = min(range(len(self._entries)), key=lambda i: self._entries[i].step)
        else:  # "elo"
            idx = min(range(len(self._entries)), key=lambda i: self._entries[i].elo or 0.0)
        return self._entries.pop(idx)
