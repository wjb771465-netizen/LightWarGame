"""终端入口：新游戏或读取存档，每回合静默保存。"""

from __future__ import annotations

from pathlib import Path

from game.datatypes.game_map import GameMap
from game.runner import GameRunner
from game.ui.terminal_ui import TerminalGameUi
from init_game import MAP_CONFIG, fixed_capitals, from_save, random_capitals

SAVES_DIR = "saves"
SAVE_PATH = f"{SAVES_DIR}/save.json"
MAP_DIR = SAVES_DIR


def _ask_num_players() -> int:
    raw = input("人数 [2-6，默认2]: ").strip()
    return int(raw) if raw.isdigit() and 2 <= int(raw) <= 6 else 2


def _ask_capitals(num_players: int) -> list[int]:
    m = GameMap(MAP_CONFIG)
    region_list = "  ".join(f"{i}.{m.regions[i].name}" for i in range(1, len(m.regions)))
    print(f"地区列表：{region_list}")
    capitals: list[int] = []
    for p in range(1, num_players + 1):
        while True:
            raw = input(f"玩家{p} 首都 ID: ").strip()
            if raw.isdigit():
                rid = int(raw)
                if m.valid_id(rid) and rid not in capitals:
                    capitals.append(rid)
                    break
            print("无效 ID，请重新输入")
    return capitals


def main() -> None:
    Path(SAVES_DIR).mkdir(exist_ok=True)
    print("=== LightWarGame ===")
    print("[1] 新游戏")
    print(f"[2] 读取存档 ({SAVE_PATH})")
    choice = input("请选择 [1/2]: ").strip()

    if choice == "2":
        state = from_save(SAVE_PATH)
        print(f"存档已加载（第 {state.turn} 回合）")
    else:
        num_players = _ask_num_players()
        print("[1] 随机首都（默认）")
        print("[2] 手动选首都")
        mode = input("请选择 [1/2]: ").strip()
        if mode == "2":
            state = fixed_capitals(_ask_capitals(num_players))
        else:
            state = random_capitals(num_players=num_players)
        print("、".join(f"玩家{p + 1} 首都 → {c}" for p, c in enumerate(state.game_map.capitals)))

    GameRunner(state, TerminalGameUi(), save_path=SAVE_PATH, map_dir=MAP_DIR).run()


if __name__ == "__main__":
    main()
