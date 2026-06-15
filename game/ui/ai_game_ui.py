from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional

from game.campaign.chat import ChatMessage, ChatRoom
from game.constants import max_commands
from game.datatypes.command import Command
from game.datatypes.state import GameState
from game.ui.display import format_battle_report


def collect_ai_commands(
    policies: dict[int, Any],
    obs_enc: Any,
    act_enc: Any,
    log_path: Optional[Path],
    state: GameState,
    player_id: int,
) -> List[Command]:
    obs = state.get_observation(player_id)
    owned = sum(1 for r in state.game_map.regions[1:] if r is not None and r.owner == player_id)
    total = max_commands(owned)
    cmds: List[Command] = []
    for i in range(total):
        obs_arr = obs_enc.encode(obs, commands_used=i, commands_total=total)
        mask = act_enc.mask(obs, commands_issued=i, max_commands=total,
                            pending_cmds=cmds if cmds else None)
        action = policies[player_id].predict(obs_arr, mask)
        cmd = act_enc.decode(action, player_id, state.game_map)
        _log_decision(state.turn, player_id, i + 1, total, cmd, state, mask, log_path)
        if cmd is None:
            break
        cmds.append(cmd)
    return cmds


def run_ai_diplomacy(
    diplomats: dict[int, Any],
    policies: dict[int, Any],
    state: GameState,
    chat_room: ChatRoom,
    save_path=None,
    battle_report: list[tuple[int, int, int]] | None = None,
) -> None:
    if save_path is not None:
        chat_room.load(str(save_path))
    for pid, diplomat in diplomats.items():
        if pid not in state.active_players:
            continue
        name = f"玩家{pid}"
        battle_lines = format_battle_report(state, battle_report) if battle_report else []
        msg = diplomat.generate_message(state, chat_room, pid, battle_lines)
        if msg:
            chat_room.add_message(ChatMessage(pid, name, msg, state.turn))
            if save_path is not None:
                chat_room.save(str(save_path))
            print(f"\n[外交 {name}] {msg}")
    for pid in state.active_players:
        if pid not in policies and pid not in diplomats:
            resp = input(f"\n[外交 玩家{pid}] 发言（Enter 跳过）: ").strip()
            if resp:
                chat_room.add_message(ChatMessage(pid, f"玩家{pid}", resp, state.turn))
                if save_path is not None:
                    chat_room.save(str(save_path))


def _log_decision(
    turn: int, player_id: int, step: int, quota: int,
    cmd: Optional[Command], state: GameState, mask, log_path: Optional[Path],
) -> None:
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
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
