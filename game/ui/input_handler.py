from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path
from typing import Callable, List, Optional, TextIO

import yaml

from game.constants import max_commands
from game.datatypes.command import Command
from game.datatypes.state import GameState


def load_session(
    input_fn: Callable[[str], str] | None = None,
    out: Optional[TextIO] = None,
) -> Path:
    from game.campaign.init_game import list_sessions, load_session_config
    o = out or sys.stdout
    inp = input_fn or input
    sessions = [s for s in list_sessions() if (s / "save" / "save.json").exists()]
    if not sessions:
        print("没有找到可用的 session，请先新建。", file=o)
        return pick_or_create_session(input_fn, out)
    for i, s in enumerate(sessions, 1):
        cfg = load_session_config(s)
        turn_info = ""
        save_file = s / "save" / "save.json"
        if save_file.exists():
            turn_info = f"（第 {json.load(open(save_file))['turn']} 回合）"
        print(f"[{i}] {cfg.get('name', s.name)} {turn_info}", file=o)
    raw = inp("请选择: ").strip()
    if raw.isdigit() and 1 <= int(raw) <= len(sessions):
        return sessions[int(raw) - 1]
    return sessions[0]


def pick_or_create_session(
    input_fn: Callable[[str], str] | None = None,
    out: Optional[TextIO] = None,
) -> Path:
    from game.campaign.init_game import list_sessions, load_session_config
    o = out or sys.stdout
    inp = input_fn or input
    ai_sessions = [s for s in list_sessions() if load_session_config(s).get("ai_players")]
    options: list[tuple[str, Path | None]] = [
        (cfg["name"], s) for s in ai_sessions if (cfg := load_session_config(s))
    ]
    options.append(("手动配置", None))
    print("加载场景：", file=o)
    for i, (name, _) in enumerate(options, 1):
        print(f"[{i}] {name}", file=o)
    raw = inp("请选择: ").strip()
    idx = int(raw) - 1 if raw.isdigit() and 1 <= int(raw) <= len(options) else len(options) - 1
    _, session_dir = options[idx]
    if session_dir is None:
        session_dir = _create_manual_session(input_fn, out)
    return session_dir


def _create_manual_session(
    input_fn: Callable[[str], str] | None,
    out: Optional[TextIO],
) -> Path:
    from game.campaign.init_game import SESSIONS_DIR
    o = out or sys.stdout
    inp = input_fn or input
    name = f"manual_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"
    session_dir = SESSIONS_DIR / name
    session_dir.mkdir(parents=True, exist_ok=True)
    num_players = _ask_num_players(input_fn)
    print("[1] 随机首都（默认）", file=o)
    print("[2] 手动选首都", file=o)
    mode = inp("请选择 [1/2]: ").strip()
    capitals: str | list[int] = "random"
    if mode == "2":
        capitals = _ask_capitals(num_players, input_fn, out)
    cfg = {
        "name": f"手动对战 {datetime.datetime.now().strftime('%m/%d %H:%M')}",
        "num_players": num_players,
        "capitals": capitals,
    }
    with open(session_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True)
    return session_dir


def _ask_num_players(input_fn: Callable[[str], str] | None) -> int:
    raw = (input_fn or input)("人数 [2-6，默认2]: ").strip()
    return int(raw) if raw.isdigit() and 2 <= int(raw) <= 6 else 2


def _ask_capitals(
    num_players: int,
    input_fn: Callable[[str], str] | None,
    out: Optional[TextIO],
) -> list[int]:
    from game.campaign.init_game import MAP_CONFIG
    from game.datatypes.game_map import GameMap
    o = out or sys.stdout
    inp = input_fn or input
    m = GameMap(MAP_CONFIG)
    region_list = "  ".join(f"{i}.{m.regions[i].name}" for i in range(1, len(m.regions)))
    print(f"地区列表：{region_list}", file=o)
    capitals: list[int] = []
    for p in range(1, num_players + 1):
        while True:
            raw = inp(f"玩家{p} 首都 ID: ").strip()
            if raw.isdigit():
                rid = int(raw)
                if m.valid_id(rid) and rid not in capitals:
                    capitals.append(rid)
                    break
            print("无效 ID，请重新输入", file=o)
    return capitals


def wait_press_to_start(
    input_fn: Callable[[str], str] | None = None,
    out: Optional[TextIO] = None,
) -> None:
    o = out or sys.stdout
    print("\n按回车键开始游戏…", file=o, flush=True)
    (input_fn or input)("")


def collect_commands_for_player(
    state: GameState,
    player_id: int,
    input_fn: Callable[[str], str] | None = None,
) -> List[Command]:
    """提示并循环读取 `源,目标,兵力`；空行或达到上限时结束。"""
    inp = input_fn or input
    regions = state.game_map.regions
    player_regions = [
        i
        for i in range(1, len(regions))
        if regions[i] is not None and regions[i].owner == player_id
    ]
    if not player_regions:
        print(f"玩家 {player_id} 没有领地，跳过指令输入")
        return []

    max_cmd_count = max_commands(len(player_regions))
    print(f"\n玩家 {player_id} 下达指令（格式: 源,目标,兵力；回车结束，上限 {max_cmd_count} 条）")
    commands: List[Command] = []
    while True:
        if len(commands) >= max_cmd_count:
            print(f"已达上限 {max_cmd_count} 条，自动结束")
            break
        line = ''.join(c for c in inp("> ") if c.isprintable()).strip()
        if not line:
            print(f"玩家 {player_id} 指令结束，共 {len(commands)} 条")
            break
        parts = line.split(",")
        if len(parts) != 3:
            print("格式错误，应为: 源地区,目标地区,兵力")
            continue
        try:
            source_idx, target_idx, troops = map(int, parts)
        except ValueError:
            print("请输入整数")
            continue
        if source_idx not in player_regions:
            print("源地区必须是你的领地")
            continue
        src = regions[source_idx]
        assert src is not None
        if troops >= src.troops:
            print("兵力不足，需至少留 1 兵防守")
            continue
        if target_idx not in src.adjacent:
            print("目标必须与源地区相邻")
            continue
        dst = regions[target_idx] if state.game_map.valid_id(target_idx) else None
        dst_name = dst.name if dst is not None else str(target_idx)
        commands.append(Command(source_idx, target_idx, troops, player_id))
        print(f"  已添加: {src.name} → {dst_name} ({troops} 兵)")

    return commands
