from __future__ import annotations

from game.datatypes.state import GameState
from ai.envs.utils import StateSnapshot
from .reward_function_base import BaseRewardFunction


class TerritoryReward(BaseRewardFunction):
    def __init__(self, territory_gain: float, territory_loss: float) -> None:
        self._gain = territory_gain
        self._loss = territory_loss

    def get_reward(
        self,
        prev: StateSnapshot,
        curr_state: GameState,
        player_id: int,
        terminated: bool,
    ) -> float:
        prev_owned = sum(1 for o in prev.owners[1:] if o == player_id)
        curr_owned = sum(
            1 for r in curr_state.game_map.regions[1:] if r is not None and r.owner == player_id
        )
        prev_enemy = sum(1 for o in prev.owners[1:] if o not in (0, player_id))
        curr_enemy = sum(
            1 for r in curr_state.game_map.regions[1:] if r is not None and r.owner not in (0, player_id)
        )
        return (curr_owned - prev_owned) * self._gain + (curr_enemy - prev_enemy) * self._loss
