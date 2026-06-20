from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional

from game.campaign.chat import ChatMessage, ChatRoom
from game.datatypes.command import Command
from game.datatypes.state import GameState
from game.ui.display import format_battle_report


def setup_ai(ai_cfg: dict[int, Any], game_map) -> tuple[dict[int, Any], dict[int, Any]]:
    from game.campaign.init_game import SESSIONS_DIR
    from ai.envs.opponents import FsmOpponent, PolicyOpponent, RandomOpponent, RuleOpponent

    opponents: dict[int, Any] = {}
    for pid, entry in ai_cfg.items():
        opp_type = entry.get("type", "policy")
        if opp_type == "rule":
            opponents[pid] = RuleOpponent(pid)
        elif opp_type == "random":
            opponents[pid] = RandomOpponent(pid)
        elif opp_type == "fsm":
            opponents[pid] = FsmOpponent(pid)
        else:  # policy (default)
            from ai.algos.policy import SB3Policy
            from ai.envs.action import ActionEncoder
            from ai.envs.observation import ObservationEncoder
            model_path = str(SESSIONS_DIR.parent / entry["model"])
            policy = SB3Policy(path=model_path)
            cfg = policy.config
            use_adj = cfg.get("use_adjacency", True)
            mp = cfg.get("max_players") or 6
            num_regions = len(game_map.regions) - 1
            obs_enc = ObservationEncoder(game_map, mp, use_adjacency=use_adj)
            act_enc = ActionEncoder(game_map)
            opponents[pid] = PolicyOpponent(pid, policy, obs_enc, act_enc)
    for opponent in opponents.values():
        opponent.reset()

    diplomats: dict[int, Any] = {}
    if any(e.get("diplomat") for e in ai_cfg.values()):
        from llm.diplomat import LLMDiplomat
        from llm.prompts import build_diplomat_system_prompt
        diplomats = {
            pid: LLMDiplomat(system_prompt=build_diplomat_system_prompt(entry.get("persona", "default")))
            for pid, entry in ai_cfg.items()
            if entry.get("diplomat", False)
        }

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    return opponents, diplomats


def collect_ai_commands(
    opponents: dict[int, Any],
    log_path: Optional[Path],
    state: GameState,
    player_id: int,
) -> List[Command]:
    cmds = opponents[player_id].act(state)
    for i, cmd in enumerate(cmds, 1):
        _log_command(state.turn, player_id, i, len(cmds), cmd, state, log_path)
    return cmds


def run_ai_diplomacy(
    diplomats: dict[int, Any],
    ai_player_ids: set[int],
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
        if pid not in ai_player_ids and pid not in diplomats:
            resp = input(f"\n[外交 玩家{pid}] 发言（Enter 跳过）: ").strip()
            if resp:
                chat_room.add_message(ChatMessage(pid, f"玩家{pid}", resp, state.turn))
                if save_path is not None:
                    chat_room.save(str(save_path))


def _log_command(
    turn: int, player_id: int, step: int, total: int,
    cmd: Command, state: GameState, log_path: Optional[Path],
) -> None:
    src = state.game_map.regions[cmd.source]
    tgt = state.game_map.regions[cmd.target]
    src_name = src.name if src is not None else "?"
    tgt_name = tgt.name if tgt is not None else "?"
    line = (f"T{turn:03d} P{player_id} #{step}/{total} "
            f"{src_name}→{tgt_name} {cmd.troops}兵")
    logging.info(line)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
