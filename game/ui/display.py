"""终端输出：仅 print，不读输入。"""

from __future__ import annotations

import sys
from typing import Optional, TextIO

from game.datatypes.game_obs import Observation
from game.datatypes.state import GameState


def show_game_start(state: GameState, out: Optional[TextIO] = None) -> None:
    o = out or sys.stdout
    print(
        "\n========== 中国地图战旗 ==========\n"
        "基本流程：仍占有领地的玩家每回合依次下指令；全员提交后统一结算，再按规则增长兵力。\n"
        "作战要点：\n"
        "  · 只能从己方地区向相邻地区派兵，派出后该地至少留 1 兵。\n"
        "  · 若出发地被敌方领地完全包围，该股途中折半（向下取整）。\n"
        "  · 同一地区同时交战：比较各方总兵力（含原驻守），最强者胜；战后剩余 = 最强值 − 次强值。\n"
        "  · 若多方并列最强：守方在并列中则守方胜；否则该地区变中立。\n"
        "指令格式：源地区编号,目标地区编号,兵力（空行结束本玩家指令）。\n"
        f"本局人数：{state.num_players}\n",
        file=o,
    )


def show_turn_start(state: GameState, out: Optional[TextIO] = None) -> None:
    o = out or sys.stdout
    print(f"\n----- 第 {state.turn} 回合 -----\n", file=o)


def show_full_state(state: GameState, out: Optional[TextIO] = None) -> None:
    """全图信息（非迷雾、上帝视角）。"""
    o = out or sys.stdout
    print("\n──────── 当前全图（上帝视角）────────", file=o)
    print("  编号  地区                    归属    现有兵力  每回增长  邻接", file=o)
    print("  ────  ──────────────────────  ──────  ────────  ────────  ────", file=o)
    regions = state.game_map.regions
    for i in range(1, len(regions)):
        r = regions[i]
        if r is None:
            continue
        cap = "·首都" if r.is_capital else ""
        name_col = f"{r.name}{cap}"
        own = "中立" if r.owner == 0 else f"P{r.owner}"
        print(
            f"  {i:4d}  {name_col:<22}  {own:<6}  {r.troops:8d}  {r.base_growth:8d}  {r.adjacent}",
            file=o,
        )
    print("────────────────────────────────────────\n", file=o)


def show_observation(obs: Observation, out: Optional[TextIO] = None) -> None:
    """观测摘要：仅己方与敌方领地，中立不显示。"""
    o = out or sys.stdout
    vid = obs.viewer_id
    print(f"\n════════ 玩家 {vid} · 第 {obs.turn} 回合 · 观测 ════════", file=o)
    print("  地区   归属            现有兵力  每回增长   首都", file=o)
    print("  ────  ──────────────  ────────  ────────  ────", file=o)
    rows: list[tuple[int, str]] = []
    for i in range(1, len(obs.regions)):
        ro = obs.regions[i]
        if ro is None:
            continue
        if ro.owner == 0:
            continue
        rid = ro.region_id
        if ro.owner == vid:
            if ro.troops is None or ro.base_growth is None or ro.is_capital is None:
                raise ValueError(
                    f"己方地区 {rid} 的 observation 缺少兵力/增长/首都标记"
                )
            cap = "是" if ro.is_capital else "否"
            line = (
                f"  {rid:4d}  {'己方':<14}  {ro.troops:8d}  {ro.base_growth:8d}  {cap:>4}"
            )
        else:
            own = f"敌·P{ro.owner}"
            line = f"  {rid:4d}  {own:<14}  {'—':>8}  {'—':>8}  {'—':>4}"
        rows.append((rid, line))
    for _, line in sorted(rows, key=lambda x: x[0]):
        print(line, file=o)
    print("════════════════════════════════════════\n", file=o)


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
