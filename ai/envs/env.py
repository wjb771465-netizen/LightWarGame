from __future__ import annotations

from typing import Any, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from game.constants import max_commands
from game.datatypes.game_map import GameMap
from game.datatypes.state import GameState
from game.ui.map_renderer import render_map
from game.campaign.init_game import fixed_capitals, random_capitals

from ai.algos.policy import SB3Policy
from ai.envs.action import ActionEncoder
from ai.envs.observation import ObservationEncoder
from ai.envs.opponents import FsmOpponent, PolicyOpponent, RandomOpponent, RuleOpponent
from ai.envs.opponents.base_opponent import BaseOpponent
from ai.envs.rewards import build_reward_functions
from ai.envs.rewards.reward_function_base import BaseRewardFunction
from ai.envs.utils import StateSnapshot, parse_config


class LwgEnv(gym.Env):
    """LightWarGame 的 Gymnasium 环境封装。

    用法：
        env = LwgEnv("two_players/vsbaseline")
        obs, info = env.reset()
        obs, reward, terminated, truncated, info = env.step(action)
        mask = env.action_masks()
    """

    def __init__(self, config_name: str, agent_id: int = 1,
                 use_adjacency: bool = False) -> None:
        self.config = parse_config(config_name)
        self.agent_id = agent_id

        game_map = GameMap(self.config.game.map_config)

        self.obs_encoder = ObservationEncoder(
            game_map, self.config.game.max_players,
            max_troops=self.config.observation.max_troops,
            max_growth=self.config.observation.max_growth,
            cmd_max=self.config.observation.cmd_max,
            use_adjacency=use_adjacency,
        )
        self.act_encoder = ActionEncoder(
            game_map,
            troop_buckets=tuple(self.config.action.troop_buckets),
        )
        self.rewards: List[BaseRewardFunction] = build_reward_functions(self.config.reward)
        opponent_id = next(p for p in range(1, self.config.game.num_players + 1) if p != self.agent_id)
        self.opponent: Optional[BaseOpponent] = self._build_opponent(opponent_id)

        self.observation_space = self.obs_encoder.space
        self.action_space = self.act_encoder.space

        self._state: Optional[GameState] = None
        self._episode_steps: int = 0
        self._pending_cmds: List[Command] = []

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._state = self._init_state()
        self._episode_steps = 0
        self._pending_cmds = []
        for rf in self.rewards:
            rf.reset(self._state, self.agent_id)
        if self.opponent is not None:
            self.opponent.reset()
        return self.obs_encoder.encode(
            self._state.get_observation(self.agent_id),
            commands_used=0,
            commands_total=self._max_cmds(),
        ), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        assert self._state is not None, "call reset() before step()"

        cmd = self.act_encoder.decode(action, player_id=self.agent_id, game_map=self._state.game_map)
        max_cmds = self._max_cmds()

        if cmd is not None:
            self._pending_cmds.append(cmd)

        # 未满额且非 no-op → 继续收集
        if cmd is not None and len(self._pending_cmds) < max_cmds:
            obs = self.obs_encoder.encode(
                self._state.get_observation(self.agent_id),
                commands_used=len(self._pending_cmds),
                commands_total=max_cmds,
            )
            return obs, 0.0, False, False, {}

        # === 回合结算 ===
        prev = StateSnapshot.from_state(self._state)
        self._episode_steps += 1

        agent_cmds = list(self._pending_cmds)
        self._pending_cmds = []

        opp_cmds = self.opponent.act(self._state) if self.opponent is not None else []
        valid = self._state.check_cmds(agent_cmds + opp_cmds)
        self._state.apply_cmds(valid)
        terminated = self._state.settle()
        truncated = self._episode_steps >= self.config.game.max_turns

        reward = float(sum(
            rf.get_reward(prev, self._state, self.agent_id, terminated or truncated)
            for rf in self.rewards
        ))

        obs = self.obs_encoder.encode(
            self._state.get_observation(self.agent_id),
            commands_used=0,
            commands_total=self._max_cmds(),
        )
        info: dict = {}
        if terminated or truncated:
            winner = self._state.winner()
            info["win"] = 1.0 if winner == self.agent_id else 0.0
            info["turn"] = self._episode_steps

        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        assert self._state is not None, "call reset() before action_masks()"
        obs = self._state.get_observation(self.agent_id)
        return self.act_encoder.mask(
            obs,
            commands_issued=len(self._pending_cmds),
            max_commands=self._max_cmds(),
            pending_cmds=self._pending_cmds if self._pending_cmds else None,
        )

    def _max_cmds(self) -> int:
        assert self._state is not None
        owned = sum(
            1 for r in self._state.game_map.regions[1:]
            if r is not None and r.owner == self.agent_id
        )
        return max_commands(owned)

    def render(self, path: str) -> None:
        assert self._state is not None, "call reset() before render()"
        render_map(self._state, path)

    def set_opponent(self, spec: dict | None) -> None:
        """替换当前对手（自博弈训练中途换对手用）。

        spec 为轻量对手描述，跨进程只传 dict，模型加载在 env 所在进程内完成：
        - ``{"type": "random", "player_id": 2}``
        - ``{"type": "rule",   "player_id": 2}``
        - ``{"type": "policy", "player_id": 2, "path": "ai/train/.../ckpt_xxx"}``
        - ``None``：清空对手
        """
        if spec is None:
            self.opponent = None
        elif spec["type"] == "random":
            self.opponent = RandomOpponent(player_id=spec["player_id"])
        elif spec["type"] == "rule":
            self.opponent = RuleOpponent(player_id=spec["player_id"])
        elif spec["type"] == "fsm":
            self.opponent = FsmOpponent(player_id=spec["player_id"])
        elif spec["type"] == "policy":
            self.opponent = PolicyOpponent(
                player_id=spec["player_id"],
                policy=SB3Policy(path=spec["path"]),
                obs_encoder=self.obs_encoder,
                act_encoder=self.act_encoder,
            )
        else:
            raise ValueError(f"Unknown opponent spec type: {spec['type']!r}")

    def set_capitals(self, agent_cap: int, opponent_cap: int) -> None:
        """覆盖下一局的首都配置（地区自博弈训练用）。"""
        # TODO: 硬编码 2，仅支持 num_players=2 时正确
        caps = [0, 0]
        caps[self.agent_id - 1] = agent_cap
        caps[2 - self.agent_id] = opponent_cap
        self.config.game.capitals = caps
        self.config.game.capital_mode = "fixed"

    def _build_opponent(self, opponent_id: int) -> Optional[BaseOpponent]:
        opp_type = getattr(self.config.training, "opponent", None)
        if opp_type == "random":
            return RandomOpponent(player_id=opponent_id)
        if opp_type == "rule":
            return RuleOpponent(player_id=opponent_id)
        if opp_type == "fsm":
            return FsmOpponent(player_id=opponent_id)
        if opp_type == "policy":
            return PolicyOpponent(
                player_id=opponent_id,
                policy=SB3Policy(path=self.config.training.policy_opponent_model),
                obs_encoder=self.obs_encoder,
                act_encoder=self.act_encoder,
            )
        if opp_type == "pool":
            return None  # 对手由训练编排层通过 set_opponent() 注入
        return None

    def _init_state(self) -> GameState:
        if self.config.game.capital_mode == "fixed":
            return fixed_capitals(self.config.game.capitals, self.config.game.map_config)
        return random_capitals(self.config.game.num_players, self.config.game.map_config)
