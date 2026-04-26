from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

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

    def reset(self, initial_state: Optional[GameState] = None, player_id: Optional[int] = None) -> None:
        """episode 开始时重置内部状态，子类按需覆写。"""


class PotentialBasedReward(BaseRewardFunction, ABC):
    """势函数奖励基类，F(s,s') = γΦ(s') − Φ(s)。

    子类只需实现 phi(state, player_id)，差分与缓存由基类统一处理。
    reset() 接收初始状态以正确初始化 Φ(s0)，避免第一步奖励偏差。
    """

    def __init__(self, gamma: float = 1.0) -> None:
        self._gamma = gamma
        self._prev_phi: float = 0.0

    @abstractmethod
    def phi(self, state: GameState, player_id: int) -> float: ...

    def reset(self, initial_state: Optional[GameState] = None, player_id: Optional[int] = None) -> None:
        self._prev_phi = self.phi(initial_state, player_id) if initial_state is not None and player_id is not None else 0.0

    def get_reward(
        self,
        prev: StateSnapshot,
        curr_state: GameState,
        player_id: int,
        terminated: bool,
    ) -> float:
        curr_phi = self.phi(curr_state, player_id)
        reward = self._gamma * curr_phi - self._prev_phi
        self._prev_phi = curr_phi
        return reward
