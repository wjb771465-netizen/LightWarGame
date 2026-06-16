from __future__ import annotations

from pathlib import Path

from game.campaign.chat import ChatRoom
from game.campaign.init_game import from_session
from game.runner import GameRunner
from game.ui.terminal_ui import TerminalGameUi
from game.utils import get_saves_dir


class GameLauncher:

    def __init__(self, session_dir: Path, is_new: bool, ui: TerminalGameUi) -> None:
        self.session_dir = session_dir
        self._is_new = is_new
        self._ui = ui
        self.save_dir = get_saves_dir(session_dir.name)

    def run(self) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        chat_room = self._load_chat()
        state = from_session(
            self.session_dir,
            save_path=None if self._is_new else self.save_dir / "save.json",
        )
        GameRunner(
            state, self._ui,
            save_path=self.save_dir,
            chat_room=chat_room if self._ui.has_ai_players else None,
        ).run()

    def _load_chat(self) -> ChatRoom:
        chat_path = self.save_dir / "chat.json"
        room = ChatRoom()
        if self._is_new:
            chat_path.unlink(missing_ok=True)
        elif chat_path.exists():
            room.load(str(chat_path))
        return room


def main() -> None:
    ui = TerminalGameUi()
    session_dir, is_new = ui.ask_launch()
    GameLauncher(session_dir, is_new, ui).run()


if __name__ == "__main__":
    main()
