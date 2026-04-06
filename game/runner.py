from __future__ import annotations

from typing import List, Optional

from game.datatypes.command import Command, CommandResult
from game.datatypes.state import GameState
from game.ui_ports import GameUiPort


class GameRunner:
    """主循环：按 UI 约定展示 → 收集指令 → `resolve_turn` → `settle`，直至终局。"""

    def __init__(self, state: GameState, ui: GameUiPort) -> None:
        self.state = state
        self.ui = ui
        self._last_turn_results: Optional[List[CommandResult]] = None

    @property
    def last_turn_results(self) -> Optional[List[CommandResult]]:
        return self._last_turn_results

    def run_single_turn(self) -> bool:
        """
        已终局（active 至多一人）时返回 False 且不结算。
        否则完整一回合 UI + resolve_turn + settle；
        settle 返回 True 表示本局已结束 → 返回 False；否则返回 True 继续游戏。
        """
        state = self.state
        ui = self.ui
        ui.show_turn_start(state)
        commands: List[Command] = []
        for p in state.active_players:
            ui.show_state(state, p)
            commands.extend(ui.collect_commands(state, p))
        self._last_turn_results = state.resolve_turn(commands)
        ui.show_turn_results(state, self._last_turn_results)
        return not state.settle()

    def run(self) -> Optional[List[CommandResult]]:
        """
        展示开局 → `while run_single_turn()` → 展示终局。
        返回最后一回合的 `CommandResult`；从未执行过结算则为 `None`。
        """
        self._last_turn_results = None
        self.ui.show_game_start(self.state)
        while self.run_single_turn():
            pass
        self.ui.show_game_result(self.state)
        return self._last_turn_results
