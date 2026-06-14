from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional, TextIO

from game.campaign.chat import ChatRoom
from game.datatypes.command import Command
from game.datatypes.game_obs import Observation
from game.datatypes.state import GameState
from game.ui_ports import GameUiPort
from game.utils import get_saves_dir
from . import ai_game_ui, display, input_handler


class TerminalGameUi(GameUiPort):
    def __init__(
        self,
        *,
        input_fn: Optional[Callable[[str], str]] = None,
        out: Optional[TextIO] = None,
    ) -> None:
        self._input_fn = input_fn
        self._out = out
        self._policies: dict[int, Any] = {}
        self._obs_enc: Any = None
        self._act_enc: Any = None
        self._log_path: Optional[Path] = None
        self._diplomats: dict[int, Any] = {}
        self._ai_cfg: dict[int, Any] = {}

    @property
    def has_ai_players(self) -> bool:
        return bool(self._policies)

    def ask_launch(self) -> tuple[Path, bool]:
        o = self._out or sys.stdout
        print("=== LightWarGame ===", file=o)
        print("[1] 新游戏", file=o)
        print("[2] 读档", file=o)
        choice = (self._input_fn or input)("请选择 [1/2]: ").strip()
        if choice == "2":
            session_dir = input_handler.load_session(self._input_fn, self._out)
            is_new = False
        else:
            session_dir = input_handler.pick_or_create_session(self._input_fn, self._out)
            is_new = True
        self._log_path = get_saves_dir(session_dir.name) / "ai_decision.log"
        from game.campaign.init_game import load_session_config
        ai_cfg = {int(k): v for k, v in load_session_config(session_dir).get("ai_players", {}).items()}
        if ai_cfg:
            self._load_ai_setup(ai_cfg)
        return session_dir, is_new

    def _load_ai_setup(self, ai_cfg: dict) -> None:
        from game.campaign.init_game import SESSIONS_DIR
        from ai.algos.policy import SB3Policy
        self._ai_cfg = ai_cfg
        self._policies = {
            pid: SB3Policy(path=str(SESSIONS_DIR.parent / entry["model"]))
            for pid, entry in ai_cfg.items()
        }
        if any(e.get("diplomat") for e in ai_cfg.values()):
            from llm.diplomat import LLMDiplomat
            from llm.prompts import build_diplomat_system_prompt
            self._diplomats = {
                pid: LLMDiplomat(system_prompt=build_diplomat_system_prompt(entry.get("persona", "default")))
                for pid, entry in ai_cfg.items()
                if entry.get("diplomat", False)
            }
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    def show_game_start(self, state: GameState) -> None:
        lines = []
        for pid, entry in self._ai_cfg.items():
            name = entry.get("name", f"玩家{pid}")
            intro = entry.get("intro", "")
            diplomat = entry.get("diplomat", False)
            lines.append(f"对手：{name}（玩家{pid}）{'· 外交官' if diplomat else ''}")
            if intro:
                lines.append(f"  {intro}")
        display.show_game_start(state, self._out, lines or None)

    def wait_after_welcome(self) -> None:
        input_handler.wait_press_to_start(self._input_fn, self._out)

    def show_turn_start(self, state: GameState) -> None:
        display.show_turn_start(state, self._out)

    def show_state(self, state: GameState) -> None:
        display.show_full_state(state, self._out)

    def show_observation(self, obs: Observation) -> None:
        display.show_observation(obs, self._out)

    def show_turn_results(self, state: GameState) -> None:
        display.show_turn_results(state, self._out)

    def show_game_result(self, state: GameState) -> None:
        display.show_game_result(state, self._out)

    def collect_commands(self, state: GameState, player_id: int) -> List[Command]:
        if player_id in self._policies:
            if self._act_enc is None:
                from ai.envs.action import ActionEncoder
                self._act_enc = ActionEncoder(state.game_map)
            if self._obs_enc is None:
                from ai.envs.observation import ObservationEncoder
                num_regions = len(state.game_map.regions) - 1
                max_players = (next(iter(self._policies.values())).obs_dim - 2) // num_regions - 6
                self._obs_enc = ObservationEncoder(state.game_map, max_players)
            return ai_game_ui.collect_ai_commands(
                self._policies, self._obs_enc, self._act_enc, self._log_path, state, player_id,
            )
        return input_handler.collect_commands_for_player(state, player_id, self._input_fn)

    def run_diplomacy(self, state: GameState, chat_room: ChatRoom, save_path=None) -> None:
        if self._diplomats or self._policies:
            ai_game_ui.run_ai_diplomacy(
                self._diplomats, self._policies, state, chat_room, save_path,
            )
