from __future__ import annotations

from game.datatypes.state import GameState
from ai.envs.utils import StateSnapshot
from .reward_function_base import BaseRewardFunction


class WinLoseReward(BaseRewardFunction):
    def __init__(self, win: float, lose: float) -> None:
        self._win = win
        self._lose = lose

    def get_reward(
        self,
        prev: StateSnapshot,
        curr_state: GameState,
        player_id: int,
        terminated: bool,
    ) -> float:
        if not terminated:
            return 0.0
        w = curr_state.winner()
        if w == player_id:
            return self._win
        if w is not None:
            return self._lose
        return 0.0  # 平局（超时）
