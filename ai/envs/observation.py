from __future__ import annotations

from typing import Set

import gymnasium as gym
import numpy as np

from game.datatypes.game_map import GameMap
from game.datatypes.game_obs import Observation

# 默认值，可通过 YAML encoder section 覆盖
_MAX_TROOPS = 500
_MAX_GROWTH = 10
_CMD_MAX = 16


class ObservationEncoder:
    """
    将 Observation 编码为固定长度 float32 向量，并暴露对应的 gymnasium Box space。

    每个 region 编码为 F 维：
      owner_onehot (max_players+1) | troops_norm | is_capital | base_growth_norm
      | is_visible [| is_adj_to_my_territory]

    owner 使用 viewer-relative one-hot：
      index 0 = neutral, index 1 = viewer 自己, index 2+ = 其他玩家（player_id 升序）

    use_adjacency=True 时追加 is_adj_to_my_territory（1 维），表示该地区是否与己方领土相邻。
    """

    def __init__(self, game_map: GameMap, max_players: int,
                 max_troops: int = _MAX_TROOPS,
                 max_growth: int = _MAX_GROWTH,
                 cmd_max: int = _CMD_MAX,
                 use_adjacency: bool = False) -> None:
        self._game_map = game_map
        self._max_players = max_players
        self._num_regions = len(game_map.regions) - 1  # 1-indexed, skip index 0
        self._use_adjacency = use_adjacency
        self._F = max_players + 5  # troops, is_capital, base_growth, is_visible
        if use_adjacency:
            self._F += 1  # is_adj_to_my_territory
        self._G = 2   # global: commands_total, commands_used
        self.dim = self._num_regions * self._F + self._G
        self._max_troops = max_troops
        self._max_growth = max_growth
        self._cmd_max = cmd_max

    @property
    def space(self) -> gym.spaces.Box:
        return gym.spaces.Box(low=0.0, high=1.0, shape=(self.dim,), dtype=np.float32)

    def encode(
        self,
        obs: Observation,
        commands_used: int = 0,
        commands_total: int = 1,
    ) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        game_map = self._game_map
        max_players = self._max_players
        F = self._F
        viewer_id = obs.viewer_id

        other_players = sorted(p for p in range(1, max_players + 1) if p != viewer_id)
        other_rank = {p: 2 + i for i, p in enumerate(other_players)}

        if self._use_adjacency:
            viewer_owned: Set[int] = {
                r.region_id for r in obs.regions[1:] if r is not None and r.owner == viewer_id
            }
            adj_to_mine: Set[int] = set()
            for rid in viewer_owned:
                region = game_map.regions[rid]
                if region is not None:
                    adj_to_mine.update(region.adjacent)
            adj_to_mine -= viewer_owned

        for idx, r_obs in enumerate(obs.regions[1:]):
            if r_obs is None:
                continue
            base = idx * F

            if r_obs.owner == 0:
                onehot_idx = 0
            elif r_obs.owner == viewer_id:
                onehot_idx = 1
            else:
                onehot_idx = other_rank.get(r_obs.owner, max_players)
            vec[base + onehot_idx] = 1.0

            o = max_players + 1
            if r_obs.troops is not None:
                vec[base + o] = min(r_obs.troops / self._max_troops, 1.0)
            o += 1
            if r_obs.is_capital is not None:
                vec[base + o] = float(r_obs.is_capital)
            o += 1
            if r_obs.base_growth is not None:
                vec[base + o] = min(r_obs.base_growth / self._max_growth, 1.0)
            o += 1
            vec[base + o] = 1.0 if r_obs.troops is not None else 0.0
            o += 1
            if self._use_adjacency:
                vec[base + o] = 1.0 if r_obs.region_id in adj_to_mine else 0.0

        # 全局特征（倒数两维）
        vec[-2] = min(commands_total / self._cmd_max, 1.0)
        vec[-1] = min(commands_used / self._cmd_max, 1.0)

        return vec
