from __future__ import annotations

from game.datatypes.state import GameState
from .reward_function_base import PotentialBasedReward


class CapitalCaptureReward(PotentialBasedReward):
    def __init__(self, capital_capture: float, gamma: float = 1.0) -> None:
        super().__init__(gamma)
        self._capital_capture = capital_capture

    def phi(self, state: GameState, player_id: int) -> float:
        return self._capital_capture * sum(
            1 for r in state.game_map.regions[1:]
            if r is not None and r.is_capital and r.owner == player_id
        )
