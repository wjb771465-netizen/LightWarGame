from __future__ import annotations

import random
from typing import List

from game.datatypes.command import Command
from game.datatypes.state import GameState
from .base_opponent import BaseOpponent


class RandomOpponent(BaseOpponent):
    """每回合从所有合法 (source, target) 对中均匀随机采样，兵力随机。"""

    def act(self, state: GameState) -> List[Command]:
        regions = state.game_map.regions
        player = self.player_id
        owned = sum(1 for r in regions[1:] if r is not None and r.owner == player)
        max_cmds = max(1, owned // 3)

        candidates: List[tuple] = []
        for rid, r in enumerate(regions[1:], 1):
            if r is None or r.owner != player or r.troops <= 1:
                continue
            for nid in r.adjacent:
                if state.game_map.valid_id(nid):
                    candidates.append((rid, nid, r.troops - 1))

        if not candidates:
            return []

        chosen = random.sample(candidates, min(max_cmds, len(candidates)))
        return [
            Command(src, tgt, random.randint(1, avail), player)
            for src, tgt, avail in chosen
        ]
