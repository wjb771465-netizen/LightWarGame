from __future__ import annotations

from game.datatypes.state import GameState
from ai.envs.utils import StateSnapshot
from .reward_function_base import BaseRewardFunction


class StepPenaltyReward(BaseRewardFunction):
    def __init__(self, step_penalty: float) -> None:
        self._penalty = step_penalty

    def get_reward(
        self,
        prev: StateSnapshot,
        curr_state: GameState,
        player_id: int,
        terminated: bool,
    ) -> float:
        return self._penalty
