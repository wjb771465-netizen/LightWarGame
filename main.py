"""终端入口：新游戏或读取存档，每回合静默保存。"""

from __future__ import annotations

from game.runner import GameRunner
from game.ui.terminal_ui import TerminalGameUi
from init_game import from_save, random_capitals

SAVE_PATH = "save.json"


def main() -> None:
    print("=== LightWarGame ===")
    print("[1] 新游戏")
    print(f"[2] 读取存档 ({SAVE_PATH})")
    choice = input("请选择 [1/2]: ").strip()

    if choice == "2":
        state = from_save(SAVE_PATH)
        print(f"存档已加载（第 {state.turn} 回合）")
    else:
        state = random_capitals()

    GameRunner(state, TerminalGameUi(), save_path=SAVE_PATH).run()


if __name__ == "__main__":
    main()
