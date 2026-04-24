from __future__ import annotations

from typing import Set

import gymnasium as gym
import numpy as np

from game.datatypes.game_map import GameMap
from game.datatypes.game_obs import Observation

_MAX_TROOPS = 500
_MAX_GROWTH = 10


class ObservationEncoder:
    """
    将 Observation 编码为固定长度 float32 向量，并暴露对应的 gymnasium Box space。

    每个 region 编码为 F 维：
      owner_onehot (max_players+1) | troops_norm | is_capital | base_growth_norm
      | is_visible | is_adj_to_my_territory

    owner 使用 viewer-relative one-hot：
      index 0 = neutral, index 1 = viewer 自己, index 2+ = 其他玩家（player_id 升序）
    """

    def __init__(self, game_map: GameMap, max_players: int) -> None:
        self._game_map = game_map
        self._max_players = max_players
        self._num_regions = len(game_map.regions) - 1  # 1-indexed, skip index 0
        self._F = max_players + 6  # owner_onehot(max_players+1) + 5 scalars
        self.dim = self._num_regions * self._F

    @property
    def space(self) -> gym.spaces.Box:
        return gym.spaces.Box(low=0.0, high=1.0, shape=(self.dim,), dtype=np.float32)

    def encode(self, obs: Observation) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        game_map = self._game_map
        max_players = self._max_players
        F = self._F
        viewer_id = obs.viewer_id

        other_players = sorted(p for p in range(1, max_players + 1) if p != viewer_id)
        other_rank = {p: 2 + i for i, p in enumerate(other_players)}

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
                vec[base + o] = min(r_obs.troops / _MAX_TROOPS, 1.0)
            o += 1
            if r_obs.is_capital is not None:
                vec[base + o] = float(r_obs.is_capital)
            o += 1
            if r_obs.base_growth is not None:
                vec[base + o] = min(r_obs.base_growth / _MAX_GROWTH, 1.0)
            o += 1
            vec[base + o] = 1.0 if r_obs.troops is not None else 0.0
            o += 1
            vec[base + o] = 1.0 if r_obs.region_id in adj_to_mine else 0.0

        return vec
