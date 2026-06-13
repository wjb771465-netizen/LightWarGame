from __future__ import annotations

from game.datatypes.state import GameState
from .reward_function_base import PotentialBasedReward


class TerritoryReward(PotentialBasedReward):
    def __init__(self, territory_gain: float, territory_loss: float, gamma: float = 1.0) -> None:
        super().__init__(gamma)
        self._gain = territory_gain
        self._loss = territory_loss

    def phi(self, state: GameState, player_id: int) -> float:
        own = sum(1 for r in state.game_map.regions[1:] if r is not None and r.owner == player_id)
        enemy = sum(1 for r in state.game_map.regions[1:] if r is not None and r.owner not in (0, player_id))
        return self._gain * own + self._loss * enemy
