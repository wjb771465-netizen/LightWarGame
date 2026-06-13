"""终端入口：GameLauncher 封装游戏初始化与 UI 构造，每回合静默保存。

AI 玩家及外交配置：编辑 ai_players.json，例如
    { "2": { "model": "saves/model.zip", "diplomat": true, "persona": "aggressive" } }
是否启用 AI 及选择哪些玩家由 AI 控制，在启动时交互决定。
"""

from __future__ import annotations

import json
from pathlib import Path

from game.chat import ChatRoom
from game.datatypes.game_map import GameMap
from game.datatypes.state import GameState
from game.init_game import MAP_CONFIG, fixed_capitals, from_save, random_capitals
from game.runner import GameRunner
from game.ui.terminal_ui import TerminalGameUi


class GameLauncher:
    SAVES_DIR = "saves"
    SAVE_PATH = "saves/save.json"
    AI_CONFIG = "ai_players.json"
    CHAT_SAVE_PATH = "saves/chat.json"

    def run(self) -> None:
        Path(self.SAVES_DIR).mkdir(exist_ok=True)
        chat_room = ChatRoom()
        state = self._setup_state(chat_room)
        ai_ids = self._ask_ai_players(state.num_players)
        ui = self._build_ui(state, ai_ids, chat_room)
        runner = GameRunner(state, ui, save_path=self.SAVE_PATH, chat_room=chat_room if ai_ids else None)
        runner.run()

    def _setup_state(self, chat_room: ChatRoom) -> GameState:
        print("=== LightWarGame ===")
        print("[1] 新游戏")
        print(f"[2] 读取存档 ({self.SAVE_PATH})")
        choice = input("请选择 [1/2]: ").strip()

        if choice == "2":
            state = from_save(self.SAVE_PATH)
            chat_room.load(self.CHAT_SAVE_PATH)
            print(f"存档已加载（第 {state.turn} 回合）")
            return state

        num_players = self._ask_num_players()
        print("[1] 随机首都（默认）")
        print("[2] 手动选首都")
        mode = input("请选择 [1/2]: ").strip()
        if mode == "2":
            state = fixed_capitals(self._ask_capitals(num_players))
        else:
            state = random_capitals(num_players=num_players)
        print("、".join(f"玩家{p + 1} 首都 → {c}" for p, c in enumerate(state.game_map.capitals)))
        return state

    def _ask_num_players(self) -> int:
        raw = input("人数 [2-6，默认2]: ").strip()
        return int(raw) if raw.isdigit() and 2 <= int(raw) <= 6 else 2

    def _ask_capitals(self, num_players: int) -> list[int]:
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

    def _ask_ai_players(self, num_players: int) -> list[int]:
        ans = input("是否启用 AI 玩家？[y/N]: ").strip().lower()
        if ans != "y":
            return []
        raw = input(f"哪些玩家由 AI 控制（空格分隔 ID，如 '2' 或 '1 2'，范围 1-{num_players}）: ").strip()
        ids = []
        for token in raw.split():
            if token.isdigit() and 1 <= int(token) <= num_players:
                ids.append(int(token))
        return sorted(set(ids))

    def _load_ai_config(self, ai_ids: list[int]) -> dict[int, dict]:
        cfg_path = Path(self.AI_CONFIG)
        raw: dict = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        cfg = {int(k): v for k, v in raw.items() if k.isdigit()}
        missing = [p for p in ai_ids if p not in cfg]
        if missing:
            raise ValueError(
                f"ai_players.json 中缺少玩家 {missing} 的配置，"
                f"请编辑 {self.AI_CONFIG} 添加对应条目。"
            )
        return cfg

    def _build_ui(self, state: GameState, ai_ids: list[int], chat_room: ChatRoom):
        if not ai_ids:
            return TerminalGameUi()
        cfg = self._load_ai_config(ai_ids)
        from ai.envs.observation import ObservationEncoder
        from ai.envs.action import ActionEncoder
        from ai.algos.policy import SB3Policy
        from game.ui.ai_game_ui import AIGameUi
        act_enc = ActionEncoder(state.game_map)
        policies = {pid: SB3Policy(path=cfg[pid]["model"]) for pid in ai_ids}
        num_regions = len(state.game_map.regions) - 1
        first_policy = next(iter(policies.values()))
        max_players = (first_policy.obs_dim - 2) // num_regions - 6
        obs_enc = ObservationEncoder(state.game_map, max_players)
        print("AI 玩家：" + "、".join(f"玩家{p} ({cfg[p]['model']})" for p in ai_ids))

        diplomats = self._build_diplomats(ai_ids, cfg)
        return AIGameUi(policies, obs_enc, act_enc, log_path="saves/ai_decision.log", diplomats=diplomats)

    def _build_diplomats(self, ai_ids: list[int], cfg: dict[int, dict]) -> dict[int, object]:
        from llm.diplomat import LLMDiplomat
        from llm.prompts import build_diplomat_system_prompt
        diplomats = {}
        for pid in ai_ids:
            entry = cfg[pid]
            if not entry.get("diplomat", False):
                continue
            persona = entry.get("persona", "default")
            system_prompt = build_diplomat_system_prompt(persona)
            diplomats[pid] = LLMDiplomat(system_prompt=system_prompt)
            print(f"外交官 玩家{pid}：persona={persona}")
        return diplomats


def main() -> None:
    GameLauncher().run()


if __name__ == "__main__":
    main()
