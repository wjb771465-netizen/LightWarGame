"""终端读入并构造 Command 列表。"""

from __future__ import annotations

import sys
from typing import Callable, List, Optional, TextIO

from game.datatypes.command import Command
from game.datatypes.state import GameState


def wait_press_to_start(
    input_fn: Callable[[str], str] | None = None,
    out: Optional[TextIO] = None,
) -> None:
    """输出提示后等待一行输入（通常为回车），用于开局确认。"""
    o = out or sys.stdout
    print("\n按回车键开始游戏…", file=o, flush=True)
    inp = input_fn or input
    inp("")


def collect_commands_for_player(
    state: GameState,
    player_id: int,
    input_fn: Callable[[str], str] | None = None,
) -> List[Command]:
    """
    提示并循环读取 `源,目标,兵力`；空行结束。
    校验基于 game_map 真值（非迷雾）。
    """
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

    print(f"\n玩家 {player_id} 下达指令（格式: 源,目标,兵力；回车结束）")
    commands: List[Command] = []
    while True:
        line = inp("> ").strip()
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
