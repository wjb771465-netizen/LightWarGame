from __future__ import annotations

from game.datatypes.state import GameState
from ai.envs.utils import StateSnapshot
from .reward_function_base import BaseRewardFunction


class CapitalCaptureReward(BaseRewardFunction):
    def __init__(self, capital_capture: float) -> None:
        self._capital_capture = capital_capture

    def get_reward(
        self,
        prev: StateSnapshot,
        curr_state: GameState,
        player_id: int,
        terminated: bool,
    ) -> float:
        reward = 0.0
        for rid in range(1, len(curr_state.game_map.regions)):
            curr_r = curr_state.game_map.regions[rid]
            if curr_r is None:
                continue
            if curr_r.is_capital and curr_r.owner == player_id and prev.owners[rid] != player_id:
                reward += self._capital_capture
        return reward
