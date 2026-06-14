"""AI 玩家 UI 适配器：继承 TerminalGameUi，仅覆盖 collect_commands。

注入式设计：obs_encoder / act_encoder / policies 由外部构造后传入，
本文件本身不导入 ai/ 或 ML 依赖，保持 game/ 无 ML 依赖。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

from game.campaign.chat import ChatMessage, ChatRoom
from game.constants import max_commands
from game.datatypes.command import Command
from game.datatypes.state import GameState
from game.ui.terminal_ui import TerminalGameUi


class AIGameUi(TerminalGameUi):
    """GameUiPort: AI 玩家走 policy，其余玩家走终端。

    Args:
        policies:    player_id → Policy（实现 predict(obs, mask) -> int）
        obs_encoder: ObservationEncoder，提供 encode(Observation) -> ndarray
        act_encoder: ActionEncoder，提供 mask(...) -> ndarray 和 decode(...) -> Command|None
        log_path:    决策日志路径，None 则不记录
    """

    def __init__(
        self,
        policies: dict[int, Any],
        obs_encoder: Any,
        act_encoder: Any,
        log_path: Optional[str] = None,
        diplomats: Optional[dict[int, Any]] = None,
    ) -> None:
        import logging
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        super().__init__()
        self._policies = policies
        self._obs_enc = obs_encoder
        self._act_enc = act_encoder
        self._log_path = Path(log_path) if log_path else None
        self._diplomats: dict[int, Any] = diplomats or {}

    def collect_commands(self, state: GameState, player_id: int) -> List[Command]:
        if player_id not in self._policies:
            return super().collect_commands(state, player_id)
        obs = state.get_observation(player_id)
        owned = sum(
            1 for r in state.game_map.regions[1:]
            if r is not None and r.owner == player_id
        )
        total = max_commands(owned)

        cmds: List[Command] = []
        for i in range(total):
            obs_arr = self._obs_enc.encode(obs, commands_used=i, commands_total=total)
            mask = self._act_enc.mask(obs, commands_issued=i, max_commands=total,
                                      pending_cmds=cmds if cmds else None)
            action = self._policies[player_id].predict(obs_arr, mask)
            cmd = self._act_enc.decode(action, player_id, state.game_map)
            self._log_decision(state.turn, player_id, i + 1, total, cmd, state, mask)
            if cmd is None:
                break
            cmds.append(cmd)
        return cmds

    def run_diplomacy(self, state: GameState, chat_room: ChatRoom, save_path=None) -> None:
        if save_path is not None:
            chat_room.load(str(save_path))
        for pid, diplomat in self._diplomats.items():
            if pid not in state.active_players:
                continue
            name = f"玩家{pid}"
            msg = diplomat.generate_message(state, chat_room, pid)
            if msg:
                chat_room.add_message(ChatMessage(pid, name, msg, state.turn))
                if save_path is not None:
                    chat_room.save(str(save_path))
                print(f"\n[外交 {name}] {msg}")
        for pid in state.active_players:
            if pid not in self._policies and pid not in self._diplomats:
                resp = input(f"\n[外交 玩家{pid}] 发言（Enter 跳过）: ").strip()
                if resp:
                    chat_room.add_message(ChatMessage(pid, f"玩家{pid}", resp, state.turn))
                    if save_path is not None:
                        chat_room.save(str(save_path))

    def _log_decision(self, turn: int, player_id: int, step: int, quota: int,
                      cmd: Optional[Command], state: GameState, mask) -> None:
        import logging
        valid_n = int(mask.sum())
        if cmd is not None:
            src = state.game_map.regions[cmd.source]
            tgt = state.game_map.regions[cmd.target]
            src_name = src.name if src is not None else "?"
            tgt_name = tgt.name if tgt is not None else "?"
            line = (f"T{turn:03d} P{player_id} #{step}/{quota} "
                    f"{src_name}→{tgt_name} {cmd.troops}兵 valid={valid_n}")
        else:
            line = f"T{turn:03d} P{player_id} #{step}/{quota} no-op valid={valid_n}"

        logging.info(line)
        if self._log_path is not None:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
