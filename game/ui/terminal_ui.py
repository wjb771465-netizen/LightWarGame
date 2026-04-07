"""终端实现 GameUiPort：组合 display + input_handler。"""

from __future__ import annotations

from typing import Callable, List, Optional, TextIO

from game.datatypes.command import Command
from game.datatypes.state import GameState
from . import display
from . import input_handler


class TerminalGameUi:
    def __init__(
        self,
        *,
        input_fn: Optional[Callable[[str], str]] = None,
        out: Optional[TextIO] = None,
    ) -> None:
        self._input_fn = input_fn
        self._out = out

    def show_game_start(self, state: GameState) -> None:
        display.show_game_start(state, self._out)

    def show_turn_start(self, state: GameState) -> None:
        display.show_turn_start(state, self._out)

    def show_state(self, state: GameState, player_id: int) -> None:
        display.show_full_state(state, player_id, self._out)

    def show_turn_results(self, state: GameState) -> None:
        display.show_turn_results(state, self._out)

    def show_game_result(self, state: GameState) -> None:
        display.show_game_result(state, self._out)

    def collect_commands(self, state: GameState, player_id: int) -> List[Command]:
        return input_handler.collect_commands_for_player(
            state, player_id, input_fn=self._input_fn
        )
