from __future__ import annotations

from game.datatypes.state import GameState
from llm.diplomat import LLMDiplomat


class LLMDirector(LLMDiplomat):
    def get_directive(self, state: GameState, player_id: int):
        raise NotImplementedError
