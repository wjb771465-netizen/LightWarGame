from __future__ import annotations

from abc import ABC, abstractmethod

from game.datatypes.state import GameState
from ai.envs.utils import StateSnapshot


class BaseRewardFunction(ABC):
    @abstractmethod
    def get_reward(
        self,
        prev: StateSnapshot,
        curr_state: GameState,
        player_id: int,
        terminated: bool,
    ) -> float: ...

    def reset(self) -> None:
        """episode 开始时重置内部状态，子类按需覆写。"""
