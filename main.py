"""终端入口：启动交互 + GameLauncher。"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import yaml

from game.campaign.chat import ChatRoom
from game.campaign.init_game import (
    MAP_CONFIG, SESSIONS_DIR,
    from_session, list_sessions, load_session_config,
)
from game.datatypes.game_map import GameMap
from game.runner import GameRunner
from game.ui.terminal_ui import TerminalGameUi
from game.utils import get_saves_dir


# ── 启动期交互：session 选择 ──────────────────────────────────────

def _load_session() -> Path:
    sessions = [
        s for s in list_sessions()
        if (s / "save" / "save.json").exists()
    ]
    if not sessions:
        print("没有找到可用的 session，请先新建。")
        return _pick_or_create_session()
    for i, s in enumerate(sessions, 1):
        cfg = load_session_config(s)
        turn_info = ""
        save_file = s / "save" / "save.json"
        if save_file.exists():
            turn_info = f"（第 {json.load(open(save_file))['turn']} 回合）"
        print(f"[{i}] {cfg.get('name', s.name)} {turn_info}")
    raw = input("请选择: ").strip()
    if raw.isdigit() and 1 <= int(raw) <= len(sessions):
        return sessions[int(raw) - 1]
    return sessions[0]


def _pick_or_create_session() -> Path:
    ai_sessions = [s for s in list_sessions() if load_session_config(s).get("ai_players")]
    options: list[tuple[str, Path | None]] = [
        (cfg["name"], s) for s in ai_sessions if (cfg := load_session_config(s))
    ]
    options.append(("手动配置", None))
    print("加载场景：")
    for i, (name, _) in enumerate(options, 1):
        print(f"[{i}] {name}")
    raw = input("请选择: ").strip()
    idx = int(raw) - 1 if raw.isdigit() and 1 <= int(raw) <= len(options) else len(options) - 1
    _, session_dir = options[idx]
    if session_dir is None:
        session_dir = _create_manual_session()
    return session_dir


def _create_manual_session() -> Path:
    name = f"manual_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"
    session_dir = SESSIONS_DIR / name
    session_dir.mkdir(parents=True, exist_ok=True)
    num_players = _ask_num_players()
    print("[1] 随机首都（默认）")
    print("[2] 手动选首都")
    mode = input("请选择 [1/2]: ").strip()
    capitals: str | list[int] = "random"
    if mode == "2":
        capitals = _ask_capitals(num_players)
    cfg = {
        "name": f"手动对战 {datetime.datetime.now().strftime('%m/%d %H:%M')}",
        "num_players": num_players,
        "capitals": capitals,
    }
    with open(session_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True)
    return session_dir


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


# ── GameLauncher ─────────────────────────────────────────────────

class GameLauncher:

    def __init__(self, session_dir: Path, is_new: bool) -> None:
        self.session_dir = session_dir
        self.save_dir = get_saves_dir(session_dir.name)
        cfg = load_session_config(session_dir)
        self.ai_cfg = {int(k): v for k, v in cfg.get("ai_players", {}).items()}
        self.ai_ids = list(self.ai_cfg.keys())
        self._is_new = is_new

    def run(self) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        chat_room = self._load_chat()
        state = from_session(
            self.session_dir,
            save_path=None if self._is_new else self.save_dir / "save.json",
        )
        ui = self._build_ui(state, chat_room)
        GameRunner(
            state, ui,
            save_path=self.save_dir,
            chat_room=chat_room if self.ai_ids else None,
        ).run()

    def _load_chat(self) -> ChatRoom:
        chat_path = self.save_dir / "chat.json"
        room = ChatRoom()
        if self._is_new:
            chat_path.unlink(missing_ok=True)
        elif chat_path.exists():
            room.load(str(chat_path))
        return room

    def _build_ui(self, state, chat_room):
        if not self.ai_ids:
            return TerminalGameUi()
        from ai.algos.policy import SB3Policy
        from ai.envs.action import ActionEncoder
        from ai.envs.observation import ObservationEncoder
        from game.ui.ai_game_ui import AIGameUi
        campaign_dir = SESSIONS_DIR.parent
        policies = {
            pid: SB3Policy(path=str(campaign_dir / self.ai_cfg[pid]["model"]))
            for pid in self.ai_ids
        }
        act_enc = ActionEncoder(state.game_map)
        num_regions = len(state.game_map.regions) - 1
        max_players = (next(iter(policies.values())).obs_dim - 2) // num_regions - 6
        obs_enc = ObservationEncoder(state.game_map, max_players)
        for pid in self.ai_ids:
            entry = self.ai_cfg[pid]
            name = entry.get("name", f"玩家{pid}")
            intro = entry.get("intro", "")
            print(f"对手：{name}（玩家{pid}）")
            if intro:
                print(f"  {intro}")
        return AIGameUi(
            policies, obs_enc, act_enc,
            log_path=str(self.save_dir / "ai_decision.log"),
            diplomats=self._build_diplomats(),
        )

    def _build_diplomats(self) -> dict[int, object]:
        from llm.diplomat import LLMDiplomat
        from llm.prompts import build_diplomat_system_prompt
        diplomats = {}
        for pid in self.ai_ids:
            entry = self.ai_cfg[pid]
            if not entry.get("diplomat", False):
                continue
            persona = entry.get("persona", "default")
            diplomats[pid] = LLMDiplomat(system_prompt=build_diplomat_system_prompt(persona))
            print(f"外交官 玩家{pid}：persona={persona}")
        return diplomats


# ── 入口 ─────────────────────────────────────────────────────────

def main() -> None:
    print("=== LightWarGame ===")
    print("[1] 新游戏")
    print("[2] 读档")
    choice = input("请选择 [1/2]: ").strip()
    if choice == "2":
        session_dir = _load_session()
        GameLauncher(session_dir, is_new=False).run()
    else:
        session_dir = _pick_or_create_session()
        GameLauncher(session_dir, is_new=True).run()


if __name__ == "__main__":
    main()
