"""终端输出：仅 print，不读输入。"""

from __future__ import annotations

import sys
from typing import Optional, TextIO

from game.datatypes.state import GameState


def show_game_start(state: GameState, out: Optional[TextIO] = None) -> None:
    o = out or sys.stdout
    print(
        "\n========== 中国地图战棋 ==========\n"
        "简要规则：每回合为所有仍占有领地的玩家收集指令；结算后兵力增长。\n"
        "指令格式：源地区编号,目标地区编号,兵力（回车结束本玩家指令）。\n"
        f"本局人数：{state.num_players}\n",
        file=o,
    )


def show_turn_start(state: GameState, out: Optional[TextIO] = None) -> None:
    o = out or sys.stdout
    print(f"\n----- 第 {state.turn} 回合 -----\n", file=o)


def show_full_state(state: GameState, viewer_id: int, out: Optional[TextIO] = None) -> None:
    """全图信息（非迷雾）；viewer_id 仅用于标题。"""
    o = out or sys.stdout
    print(f"【玩家 {viewer_id} 视角 — 当前全图】", file=o)
    regions = state.game_map.regions
    for i in range(1, len(regions)):
        r = regions[i]
        if r is None:
            continue
        cap = " [首都]" if r.is_capital else ""
        own = "无主" if r.owner == 0 else f"玩家{r.owner}"
        print(
            f"  {i:2d}. {r.name}{cap} | 归属:{own} | 兵:{r.troops} | 增长:{r.base_growth} | 邻:{r.adjacent}",
            file=o,
        )
    print(file=o)


def show_turn_results(state: GameState, out: Optional[TextIO] = None) -> None:
    _ = state
    o = out or sys.stdout
    print("（本回合战报待实现）\n", file=o)


def show_game_result(state: GameState, out: Optional[TextIO] = None) -> None:
    o = out or sys.stdout
    w = state.winner()
    print("\n========== 游戏结束 ==========", file=o)
    if w is None:
        print("无唯一胜者（平局或异常终局）。", file=o)
    else:
        print(f"玩家 {w} 获胜！", file=o)
    print("==============================\n", file=o)
