"""每个地区维护独立的 OpponentPool，支持跨地区随机抽取对手 checkpoint。"""

from __future__ import annotations

import json
import os
import random
import threading

from ai.algos.opponent_pool import OpponentPool, PoolEntry


class RegionPool:
    """31 个地区各持一个 OpponentPool，训练器通过 sample_opponent 随机抽取跨地区对手。

    线程安全：所有公共方法通过内部锁保护，可安全用于多线程并发访问。
    """

    def __init__(self, history: int = 3) -> None:
        self._history = history
        self._pools: dict[int, OpponentPool] = {}
        self._lock = threading.Lock()

    def add(self, region_id: int, path: str, step: int,
            elo: float | None = None,
            outcomes: list | None = None,
            ) -> tuple[PoolEntry | None, float, bool]:
        """注册 checkpoint；outcomes 每个元素应有 .opponent_spec(dict, 含 opp_region)
        / .wins / .draws / .episodes，内部解析对手 step+region、算分、_update_elo。"""
        if outcomes is not None:
            agent_elo = elo if elo is not None else 1200.0
            prev_elo = agent_elo
            for r in outcomes:
                spec = r.opponent_spec
                opp_region = spec.get("opp_region")
                opp_step = OpponentPool._parse_step_from_spec(spec)
                if opp_region is None or opp_step is None:
                    continue
                agent_elo, _ = self._update_elo(
                    opp_region, opp_step, agent_elo, OpponentPool._compute_score(r))
            if not OpponentPool._should_accept(prev_elo, agent_elo):
                return None, agent_elo, False
            elo = agent_elo
            outcomes = None  # 已处理，不往下传

        with self._lock:
            if region_id not in self._pools:
                self._pools[region_id] = OpponentPool(max_size=self._history)
            return self._pools[region_id].add(path, step, elo=elo)

    def sample_opponent(
        self,
        exclude_region: int,
        strategy: str = "latest",
        lam: float = 1.0,
        s: float = 100.0,
        progress_D: float | None = None,
    ) -> tuple[int, PoolEntry] | None:
        """随机抽一个地区（排除 exclude_region），按 strategy 从该地区池中采样。

        strategy: latest | uniform | progress | elo
        """
        with self._lock:
            eligible = [rid for rid in self._pools if rid != exclude_region and len(self._pools[rid]) > 0]
            if not eligible:
                return None
            rid = random.choice(eligible)
            pool = self._pools[rid]
            if strategy == "uniform":
                entry = pool.sample_uniform()
            elif strategy == "progress":
                entry = pool.sample_progress(lam=lam, s=s, D=progress_D)
            elif strategy == "elo":
                entry = pool.sample_elo(lam=lam, s=s)
            else:  # "latest"
                entry = pool.latest()
            if entry is None:
                return None
            return rid, entry

    def latest(self, region_id: int) -> PoolEntry | None:
        with self._lock:
            pool = self._pools.get(region_id)
            return pool.latest() if pool is not None else None

    def _update_elo(self, region_id: int, opponent_step: int,
                   agent_elo: float, score: float, K: float = 32.0
                   ) -> tuple[float, float]:
        """更新指定地区池中某个对手的 ELO。"""
        with self._lock:
            pool = self._pools.get(region_id)
            if pool is None:
                return agent_elo, 1200.0
            return pool._update_elo(opponent_step, agent_elo, score, K=K)

    def available_regions(self) -> list[int]:
        with self._lock:
            return sorted(rid for rid, pool in self._pools.items() if len(pool) > 0)

    def save(self, path: str) -> None:
        with self._lock:
            data = {
                "history": self._history,
                "regions": {
                    str(rid): [
                        {"path": e.path, "step": e.step, "elo": e.elo}
                        for e in pool
                    ]
                    for rid, pool in self._pools.items()
                },
            }
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> RegionPool:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        pool = cls(history=data["history"])
        for rid_str, entries in data["regions"].items():
            rid = int(rid_str)
            for entry in entries:
                pool.add(rid, entry["path"], entry["step"], elo=entry.get("elo"))
        return pool
