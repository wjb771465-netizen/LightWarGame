from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Any, List, Optional

from game.campaign.chat import ChatRoom
from game.constants import max_commands
from game.datatypes.command import Command
from game.datatypes.game_obs import Observation
from game.datatypes.state import GameState
from game.ui_ports import GameUiPort


class WebGameUi(GameUiPort):
    """线程安全的 GameUiPort：Runner 线程写、Flask 线程读，通过 Lock + Queue 协调。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cmd_queue: queue.Queue[List[Command]] = queue.Queue()
        self._launch_ready = threading.Event()

        # ---- 启动信息（由 /start 路由设置） ----
        self._session_dir: Optional[Path] = None
        self._is_new: bool = True

        # ---- 共享状态（_lock 保护） ----
        self._phase: str = "lobby"          # lobby | playing | waiting | result | over
        self._turn: int = 0
        self._num_players: int = 0
        self._current_player: int = 0
        self._observation: Optional[dict] = None
        self._battle_changes: list[str] = []
        self._winner: Optional[int] = None
        self._game_map = None
        self._obs_seq: int = 0               # 每次 show_observation 递增
        self._map_path: Optional[str] = None  # 当前回合地图 PNG
        self._chat_room: Optional[ChatRoom] = None
        self._chat_save_path: Optional[str] = None
        self._session_name: str = ""

        # ---- 指令累积 ----
        self._pending_commands: list[Command] = []

        # ---- AI 支持 ----
        self._opponents: dict[int, Any] = {}
        self._diplomats: dict[int, Any] = {}
        self._ai_cfg: dict[int, Any] = {}
        self._log_path: Optional[Path] = None

        # ---- 错误 ----
        self._error: Optional[str] = None
        self._cmd_error: Optional[str] = None

    # ------------------------------------------------------------------
    # 启动
    # ------------------------------------------------------------------

    def prepare_launch(self, session_dir: Path, is_new: bool) -> None:
        with self._lock:
            self._session_dir = Path(session_dir)
            self._session_name = Path(session_dir).name
            self._is_new = is_new
            self._phase = "lobby"
            self._observation = None
            self._battle_changes = []
            self._winner = None
            self._pending_commands = []
            self._opponents = {}
            self._diplomats = {}
            self._error = None
            self._cmd_error = None
            self._obs_seq = 0
            self._launch_ready.set()
        # 清空残留指令（上一局 Runner 可能已退出但 Queue 还有东西）
        while not self._cmd_queue.empty():
            try:
                self._cmd_queue.get_nowait()
            except queue.Empty:
                break

    def ask_launch(self) -> tuple[Path, bool]:
        """Runner 调用：阻塞直到 prepare_launch 被调用。"""
        self._launch_ready.wait()
        with self._lock:
            return self._session_dir, self._is_new  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # show_* — 非阻塞，只存状态
    # ------------------------------------------------------------------

    def show_game_start(self, state: GameState) -> None:
        from game.campaign.init_game import SESSIONS_DIR
        if self._ai_cfg:
            from game.ui import ai_game_ui
            self._opponents, self._diplomats = ai_game_ui.setup_ai(
                self._ai_cfg, state.game_map,
            )
        with self._lock:
            self._game_map = state.game_map
            self._turn = state.turn
            self._num_players = state.num_players
            self._phase = "playing"

    def wait_after_welcome(self) -> None:
        pass  # Web 版不需要"按 Enter 继续"

    def show_turn_start(self, state: GameState, map_path: Path) -> None:
        with self._lock:
            self._turn = state.turn
            self._map_path = str(map_path)

    def show_state(self, state: GameState) -> None:
        pass  # 终端上帝视角，Web 不需要

    def show_observation(self, obs: Observation) -> None:
        """Runner 调用：存储观测，AI 玩家静默跳过。"""
        if obs.viewer_id in self._ai_cfg:
            return
        with self._lock:
            self._observation = self._obs_to_dict(obs)
            self._current_player = obs.viewer_id
            self._obs_seq += 1
            self._phase = "playing"
            self._pending_commands = []

    def show_turn_results(
        self, state: GameState, battle_report: list[tuple[int, int, int]]
    ) -> None:
        from game.ui.display import format_battle_report
        changes = format_battle_report(state, battle_report)
        with self._lock:
            self._battle_changes = changes
            self._phase = "result"

    def show_game_result(self, state: GameState) -> None:
        with self._lock:
            self._winner = state.winner()
            self._phase = "over"

    def run_diplomacy(
        self, state: GameState, chat_room: ChatRoom,
        save_path=None, battle_report: list[tuple[int, int, int]] | None = None,
    ) -> None:
        """AI 外交官自动发言；人类玩家通过 web 表单发言（不在此阻塞）。"""
        chat_save = str(save_path) if save_path is not None else None
        with self._lock:
            self._chat_room = chat_room
            self._chat_save_path = chat_save
        # 不调 chat_room.load() —— web 单进程，人类发言通过 add_chat_message
        # 实时写入同一实例，reload 反而会覆盖掉未持久化的人类消息
        for pid, diplomat in self._diplomats.items():
            if pid not in state.active_players:
                continue
            from game.ui.display import format_battle_report
            name = f"玩家{pid}"
            battle_lines = format_battle_report(state, battle_report) if battle_report else []
            msg = diplomat.generate_message(state, chat_room, pid, battle_lines)
            if msg:
                from game.campaign.chat import ChatMessage
                chat_room.add_message(ChatMessage(pid, name, msg, state.turn))
                if chat_save is not None:
                    chat_room.save(chat_save)

    # ------------------------------------------------------------------
    # collect_commands — 人类阻塞，AI 瞬返
    # ------------------------------------------------------------------

    def collect_commands(self, state: GameState, player_id: int) -> List[Command]:
        if player_id in self._opponents:
            from game.ui import ai_game_ui
            return ai_game_ui.collect_ai_commands(
                self._opponents, self._log_path, state, player_id,
            )
        return self._cmd_queue.get()

    # ------------------------------------------------------------------
    # Web 路由辅助方法（由 routes.py 调用）
    # ------------------------------------------------------------------

    def load_ai_config(self, config_path: Path) -> None:
        """从 session config 加载 AI 配置。"""
        from game.campaign.init_game import load_session_config
        cfg = load_session_config(config_path)
        self._ai_cfg = {
            int(k): v for k, v in cfg.get("ai_players", {}).items()
        }

    def set_error(self, tb: str) -> None:
        """Runner 线程崩溃时调用，记录 Traceback。"""
        with self._lock:
            self._error = tb
            self._phase = "error"

    def set_log_path(self, path: Path) -> None:
        self._log_path = path

    def add_pending_command(self, cmd: Command, obs_regions: list) -> Optional[str]:
        with self._lock:
            src = next((r for r in obs_regions if r["id"] == cmd.source), None)
            if src is None:
                return f"地区 {cmd.source} 不在观测范围内"
            if src["owner"] != self._current_player:
                return f"地区 {cmd.source} 不归你所有"
            if cmd.target not in src.get("adjacent", []):
                return f"地区 {src['id']} 与 {cmd.target} 不相邻"

            owned = sum(1 for r in obs_regions if r["owner"] == self._current_player)
            limit = max_commands(owned)
            if len(self._pending_commands) >= limit:
                return f"本回合已达指令上限 {limit} 条"

            pending_from_src = sum(
                c.troops for c in self._pending_commands if c.source == cmd.source
            )
            available = (src["troops"] or 0) - 1 - pending_from_src
            if cmd.troops > available:
                return (f"兵力不足（{src['name']} 现有 {src['troops']} 兵，"
                        f"至少留 1 兵，已派 {pending_from_src}，"
                        f"可用 {max(0, available)}）")

            self._pending_commands.append(cmd)
            return None

    def submit_commands(self) -> None:
        """玩家点击"提交全部"：将累积指令送入 Queue 唤醒 Runner。"""
        with self._lock:
            cmds = list(self._pending_commands)
            self._pending_commands = []
            self._phase = "waiting"
            self._observation = None
            self._cmd_queue.put(cmds)

    def add_chat_message(self, sender_id: int, text: str) -> None:
        """人类玩家通过 web 表单发送外交消息。"""
        from game.campaign.chat import ChatMessage
        with self._lock:
            chat_room = self._chat_room
            turn = self._turn
            save_path = self._chat_save_path
        if chat_room is None:
            return
        name = f"玩家{sender_id}"
        chat_room.add_message(ChatMessage(sender_id, name, text, turn))
        if save_path is not None:
            chat_room.save(save_path)

    def snapshot(self) -> dict:
        """Flask 路由获取当前状态的快照（线程安全）。"""
        with self._lock:
            chat_msgs = []
            if self._chat_room is not None:
                for m in self._chat_room.get_history():
                    chat_msgs.append({
                        "sender_id": m.sender_id,
                        "sender_name": m.sender_name,
                        "text": m.text,
                        "turn": m.turn,
                    })
            return {
                "phase": self._phase,
                "turn": self._turn,
                "num_players": self._num_players,
                "current_player": self._current_player,
                "observation": self._observation,
                "battle_changes": self._battle_changes,
                "winner": self._winner,
                "pending_commands": [
                    {"source_name": self._region_name(c.source),
                     "target_name": self._region_name(c.target),
                     "troops": c.troops}
                    for c in self._pending_commands
                ],
                "obs_seq": self._obs_seq,
                "error": self._error,
                "cmd_error": self._cmd_error,
                "map_path": self._map_path,
                "session_name": self._session_name,
                "chat_messages": chat_msgs[-20:],
            }

    @property
    def has_ai_players(self) -> bool:
        return bool(self._ai_cfg)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _obs_to_dict(self, obs: Observation) -> dict:
        """Observation → 模板可用的普通 dict，附带 region name。"""
        regions = []
        for i in range(1, len(obs.regions)):
            ro = obs.regions[i]
            if ro is None or ro.owner == 0:
                continue
            region = self._game_map.regions[i]
            regions.append({
                "id": ro.region_id,
                "name": region.name,
                "owner": ro.owner,
                "troops": ro.troops,
                "is_capital": ro.is_capital,
                "base_growth": ro.base_growth,
                "adjacent": region.adjacent,
            })
        return {
            "viewer_id": obs.viewer_id,
            "turn": obs.turn,
            "regions": regions,
        }

    def _region_name(self, region_id: int) -> str:
        if self._game_map is None:
            return str(region_id)
        r = self._game_map.regions[region_id]
        return r.name if r is not None else str(region_id)
