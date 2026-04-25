from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from game.datatypes.command import Command
from game.datatypes.state import GameState


class BaseOpponent(ABC):
    def __init__(self, player_id: int) -> None:
        self.player_id = player_id

    @abstractmethod
    def act(self, state: GameState) -> List[Command]: ...

    def reset(self) -> None:
        """episode 开始时重置内部状态，有状态的对手按需覆写。"""
