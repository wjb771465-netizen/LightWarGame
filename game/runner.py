from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from game.campaign.chat import ChatRoom
from game.datatypes.command import Command
from game.datatypes.state import GameState
from game.campaign.save_load import save_game, save_turn_map, save_turn_obs
from game.ui_ports import GameUiPort


class GameRunner:
    """主循环：按 UI 约定展示 → 收集指令 → `check_cmds` → `apply_cmds` → `settle`，直至终局。"""

    def __init__(
        self,
        state: GameState,
        ui: GameUiPort,
        save_path: Path,
        chat_room: Optional[ChatRoom] = None,
    ) -> None:
        self.state = state
        self.ui = ui
        self._save_path = save_path
        self._chat_room = chat_room
        self._last_battle_report: list[tuple[int, int, int]] = []

    def run_single_turn(self) -> bool:
        """
        已终局（active 至多一人）时返回 False 且不结算。
        否则完整一回合 UI + check_cmds + apply_cmds + settle；
        settle 返回 True 表示本局已结束 → 返回 False；否则返回 True 继续游戏。
        """
        save_game(self.state, str(self._save_path / "save.json"))
        state = self.state
        ui = self.ui
        map_path = save_turn_map(state, self._save_path)
        ui.show_turn_start(state, map_path)
        if self._chat_room is not None:
            ui.run_diplomacy(state, self._chat_room, self._save_path / "chat.json",
                             self._last_battle_report)
        commands: List[Command] = []
        for p in state.active_players:
            obs = state.get_observation(p)
            ui.show_observation(obs)
            save_turn_obs(obs, p, state, self._save_path)
            commands.extend(ui.collect_commands(state, p))
        valid_cmds = state.check_cmds(commands)
        self._last_battle_report = state.apply_cmds(valid_cmds)
        ui.show_turn_results(state, self._last_battle_report)
        return not state.settle()

    def run(self) -> None:
        """展示开局 → `while run_single_turn()` → 展示终局。"""
        self.ui.show_game_start(self.state)
        self.ui.wait_after_welcome()
        while self.run_single_turn():
            pass
        self.ui.show_game_result(self.state)
