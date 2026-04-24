from __future__ import annotations

import math
from typing import List, Optional, Tuple

import gymnasium as gym
import numpy as np

from game.datatypes.command import Command
from game.datatypes.game_map import GameMap
from game.datatypes.game_obs import Observation

_TROOP_BUCKETS: Tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)


class ActionEncoder:
    """
    动作空间封装。

    动作编码（Discrete）：
      index 0          = no-op
      index 1..E*B     = (edge_idx, bucket_idx)，k = 1 + edge_idx*B + bucket_idx

    edge_list 从地图邻接关系构建，排序固定化，初始化后不可变。
    bucket 为出兵比例档位，基数 = src.troops - 1。
    """

    def __init__(self, game_map: GameMap) -> None:
        self._game_map = game_map
        self._edges: List[Tuple[int, int]] = self._build_edge_list(game_map)
        self._B = len(_TROOP_BUCKETS)
        self.dim = len(self._edges) * self._B + 1  # +1 for no-op

    @property
    def space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(self.dim)

    def mask(
        self,
        obs: Observation,
        commands_issued: int,
        max_commands: int,
    ) -> np.ndarray:
        """
        计算合法动作的 bool 掩码，长度 = self.dim。

        index 0（no-op）始终合法。
        index k>0 合法条件：尚有配额 AND src 归 viewer AND src.troops > 1。
        """
        result = np.zeros(self.dim, dtype=bool)
        result[0] = True

        if commands_issued >= max_commands:
            return result

        game_map = self._game_map
        viewer_id = obs.viewer_id
        B = self._B
        for edge_idx, (src_id, _tgt_id) in enumerate(self._edges):
            region = game_map.regions[src_id]
            if region is None or region.owner != viewer_id or region.troops <= 1:
                continue
            base = 1 + edge_idx * B
            result[base : base + B] = True

        return result

    def decode(self, action: int, player_id: int) -> Optional[Command]:
        """将动作索引解码为 Command；no-op（index 0）返回 None。"""
        if action == 0:
            return None

        edge_idx, bucket_idx = divmod(action - 1, self._B)
        src_id, tgt_id = self._edges[edge_idx]
        region = self._game_map.regions[src_id]
        assert region is not None

        available = region.troops - 1
        troops = max(1, math.floor(available * _TROOP_BUCKETS[bucket_idx]))
        return Command(source=src_id, target=tgt_id, troops=troops, player=player_id)

    @staticmethod
    def _build_edge_list(game_map: GameMap) -> List[Tuple[int, int]]:
        edges: List[Tuple[int, int]] = []
        for src_id in range(1, len(game_map.regions)):
            region = game_map.regions[src_id]
            if region is None:
                continue
            for tgt_id in region.adjacent:
                edges.append((src_id, tgt_id))
        edges.sort()
        return edges
