"""策略池：管理历史 checkpoint，支持多种淘汰模式。

池只管理 entry 索引，不负责构造 PolicyOpponent。
"""

from __future__ import annotations

from dataclasses import dataclass


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
