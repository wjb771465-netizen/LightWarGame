from __future__ import annotations

from typing import List, Optional
import os

from game.datatypes.command import Command
from game.datatypes.state import GameState
from game.save_load import save_game
from game.ui_ports import GameUiPort
from game.ui.map_renderer import render_map


class GameRunner:
    """主循环：按 UI 约定展示 → 收集指令 → `check_cmds` → `apply_cmds` → `settle`，直至终局。"""

    def __init__(self, state: GameState, ui: GameUiPort, save_path: Optional[str] = None, map_dir: Optional[str] = None) -> None:
        self.state = state
        self.ui = ui
        self._save_path = save_path
        self._map_dir = map_dir

    def run_single_turn(self) -> bool:
        """
        已终局（active 至多一人）时返回 False 且不结算。
        否则完整一回合 UI + check_cmds + apply_cmds + settle；
        settle 返回 True 表示本局已结束 → 返回 False；否则返回 True 继续游戏。
        """
        state = self.state
        ui = self.ui
        ui.show_turn_start(state)
        #ui.show_state(state)
        commands: List[Command] = []
        for p in state.active_players:
            ui.show_observation(state.get_observation(p))
            commands.extend(ui.collect_commands(state, p))
        valid_cmds = state.check_cmds(commands)
        state.apply_cmds(valid_cmds)
        ui.show_turn_results(state)
        if self._map_dir:
            path = os.path.join(self._map_dir, f"map_turn_{state.turn:03d}.png")
            render_map(state, path)
            print(f"[地图] 已保存 → {path}")
        return not state.settle()

    def run(self) -> None:
        """展示开局 → `while run_single_turn()` → 展示终局。"""
        self.ui.show_game_start(self.state)
        self.ui.wait_after_welcome()
        while self.run_single_turn():
            if self._save_path:
                save_game(self.state, self._save_path)
        self.ui.show_game_result(self.state)
