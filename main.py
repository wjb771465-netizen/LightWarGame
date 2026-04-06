"""最简入口：随机首都开局，终端 UI + Runner，无存档。"""

from __future__ import annotations

import random

from game.datatypes.game_map import GameMap
from game.datatypes.state import GameState
from game.runner import GameRunner
from game.ui.terminal_ui import TerminalGameUi

NUM_PLAYERS = 2
MAP_CONFIG = "cn"


def main() -> None:
    m = GameMap(MAP_CONFIG)
    region_ids = [i for i in range(1, len(m.regions)) if m.valid_id(i)]
    capitals = random.sample(region_ids, NUM_PLAYERS)
    m.assign_capitals(capitals)
    print(f"随机首都：玩家1 → {capitals[0]}，玩家2 → {capitals[1]}")

    state = GameState(m, num_players=NUM_PLAYERS)
    GameRunner(state, TerminalGameUi()).run()


if __name__ == "__main__":
    main()
