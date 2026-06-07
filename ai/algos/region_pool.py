"""每个地区维护独立的 OpponentPool，支持跨地区随机抽取对手 checkpoint。"""

from __future__ import annotations

import json
import os
import random

from ai.algos.opponent_pool import OpponentPool, PoolEntry


class RegionPool:
    """31 个地区各持一个 OpponentPool，训练器通过 sample_opponent 随机抽取跨地区对手。"""

    def __init__(self, history: int = 3) -> None:
        self._history = history
        self._pools: dict[int, OpponentPool] = {}

    def add(self, region_id: int, path: str, step: int) -> None:
        if region_id not in self._pools:
            self._pools[region_id] = OpponentPool(max_size=self._history)
        self._pools[region_id].add(path, step)

    def sample_opponent(self, exclude_region: int) -> tuple[int, PoolEntry] | None:
        """随机抽一个地区（排除 exclude_region），返回该地区最新的 checkpoint。"""
        eligible = [rid for rid in self._pools if rid != exclude_region and len(self._pools[rid]) > 0]
        if not eligible:
            return None
        rid = random.choice(eligible)
        return rid, self._pools[rid].latest()

    def latest(self, region_id: int) -> PoolEntry | None:
        pool = self._pools.get(region_id)
        return pool.latest() if pool is not None else None

    def available_regions(self) -> list[int]:
        return sorted(rid for rid, pool in self._pools.items() if len(pool) > 0)

    def save(self, path: str) -> None:
        data = {
            "history": self._history,
            "regions": {
                str(rid): [
                    {"path": pool.sample(i).path, "step": pool.sample(i).step, "elo": pool.sample(i).elo}
                    for i in range(len(pool))
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
                pool.add(rid, entry["path"], entry["step"])
        return pool
