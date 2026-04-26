"""AI 玩家 UI 适配器：继承 TerminalGameUi，仅覆盖 collect_commands。

注入式设计：obs_encoder / act_encoder / policies 由外部构造后传入，
本文件本身不导入 ai/ 或 ML 依赖，保持 game/ 无 ML 依赖。
"""

from __future__ import annotations

from typing import Any, List

from game.datatypes.command import Command
from game.datatypes.state import GameState
from game.ui.terminal_ui import TerminalGameUi


class AIGameUi(TerminalGameUi):
    """GameUiPort: AI 玩家走 policy，其余玩家走终端。

    Args:
        policies:    player_id → Policy（实现 predict(obs, mask) -> int）
        obs_encoder: ObservationEncoder，提供 encode(Observation) -> ndarray
        act_encoder: ActionEncoder，提供 mask(...) -> ndarray 和 decode(...) -> Command|None
    """

    def __init__(
        self,
        policies: dict[int, Any],
        obs_encoder: Any,
        act_encoder: Any,
    ) -> None:
        super().__init__()
        self._policies = policies
        self._obs_enc = obs_encoder
        self._act_enc = act_encoder

    def collect_commands(self, state: GameState, player_id: int) -> List[Command]:
        if player_id not in self._policies:
            return super().collect_commands(state, player_id)
        obs = state.get_observation(player_id)
        obs_arr = self._obs_enc.encode(obs)
        owned = sum(
            1 for r in state.game_map.regions[1:]
            if r is not None and r.owner == player_id
        )
        mask = self._act_enc.mask(obs, commands_issued=0, max_commands=max(1, owned // 3))
        action = self._policies[player_id].predict(obs_arr, mask)
        cmd = self._act_enc.decode(action, player_id, state.game_map)
        return [cmd] if cmd is not None else []
