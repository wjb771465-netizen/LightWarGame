"""加载 checkpoint 作为对手，实现 BaseOpponent 接口。

内部自回归循环生成多条指令：每步 predict 后更新 mask（按 pending 收缩兵力），
直到策略输出 no-op 或达到配额上限。
"""

from __future__ import annotations

from typing import List

from game.constants import max_commands
from game.datatypes.command import Command
from game.datatypes.state import GameState
from ai.envs.opponents.base_opponent import BaseOpponent


class PolicyOpponent(BaseOpponent):
    """用训练好的 checkpoint 作为对手，每回合自回归生成多条指令。

    ObservationEncoder / ActionEncoder 和 Policy 由调用方注入。
    """

    def __init__(
        self,
        player_id: int,
        policy,          # Policy: predict(obs: ndarray, mask: ndarray) -> int
        obs_encoder,     # ObservationEncoder
        act_encoder,     # ActionEncoder
    ) -> None:
        super().__init__(player_id)
        self._policy = policy
        self._obs_enc = obs_encoder
        self._act_enc = act_encoder

    # ------------------------------------------------------------------
    def act(self, state: GameState) -> List[Command]:
        player = self.player_id
        obs = state.get_observation(player)

        owned = sum(
            1 for r in state.game_map.regions[1:]
            if r is not None and r.owner == player
        )
        total = max_commands(owned)

        commands: List[Command] = []
        for i in range(total):
            obs_arr = self._obs_enc.encode(obs, commands_used=i, commands_total=total)
            mask = self._act_enc.mask(obs, commands_issued=i, max_commands=total,
                                      pending_cmds=commands if commands else None)
            action = self._policy.predict(obs_arr, mask, deterministic=False)
            cmd = self._act_enc.decode(action, player_id=player, game_map=state.game_map)
            if cmd is None:
                break
            commands.append(cmd)

        return commands
