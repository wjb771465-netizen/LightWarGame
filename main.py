"""终端入口：新游戏或读取存档，每回合静默保存。"""

from __future__ import annotations

from pathlib import Path

from game.runner import GameRunner
from game.ui.terminal_ui import TerminalGameUi
from init_game import from_save, random_capitals

SAVES_DIR = "saves"
SAVE_PATH = f"{SAVES_DIR}/save.json"
MAP_DIR = SAVES_DIR


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
        state = random_capitals()

    GameRunner(state, TerminalGameUi(), save_path=SAVE_PATH, map_dir=MAP_DIR).run()


if __name__ == "__main__":
    main()
