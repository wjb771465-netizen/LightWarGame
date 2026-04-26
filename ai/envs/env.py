from __future__ import annotations

from typing import Any, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from game.datatypes.game_map import GameMap
from game.datatypes.state import GameState
from game.ui.map_renderer import render_map
from init_game import fixed_capitals, random_capitals

from ai.envs.action import ActionEncoder
from ai.envs.observation import ObservationEncoder
from ai.envs.opponents import RandomOpponent, RuleOpponent
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

    def __init__(self, config_name: str, agent_id: int = 1) -> None:
        self.config = parse_config(config_name)
        self.agent_id = agent_id

        game_map = GameMap(self.config.game.map_config)
        self.obs_encoder = ObservationEncoder(game_map, self.config.game.max_players)
        self.act_encoder = ActionEncoder(game_map)
        self.rewards: List[BaseRewardFunction] = build_reward_functions(self.config.reward)
        opponent_id = next(p for p in range(1, self.config.game.num_players + 1) if p != self.agent_id)
        self.opponent: Optional[BaseOpponent] = self._build_opponent(opponent_id)

        self.observation_space = self.obs_encoder.space
        self.action_space = self.act_encoder.space

        self._state: Optional[GameState] = None
        self._episode_steps: int = 0

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._state = self._init_state()
        self._episode_steps = 0
        for rf in self.rewards:
            rf.reset(self._state, self.agent_id)
        if self.opponent is not None:
            self.opponent.reset()
        return self.obs_encoder.encode(self._state.get_observation(self.agent_id)), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        assert self._state is not None, "call reset() before step()"

        prev = StateSnapshot.from_state(self._state)
        self._episode_steps += 1

        agent_cmd = self.act_encoder.decode(action, player_id=self.agent_id, game_map=self._state.game_map)
        agent_cmds = [agent_cmd] if agent_cmd is not None else []
        opp_cmds = self.opponent.act(self._state) if self.opponent is not None else []

        valid = self._state.check_cmds(agent_cmds + opp_cmds)
        self._state.apply_cmds(valid)
        terminated = self._state.settle()
        truncated = self._episode_steps >= self.config.game.max_turns

        reward = float(sum(
            rf.get_reward(prev, self._state, self.agent_id, terminated or truncated)
            for rf in self.rewards
        ))

        obs = self.obs_encoder.encode(self._state.get_observation(self.agent_id))
        info: dict = {}
        if terminated or truncated:
            winner = self._state.winner()
            info["win"] = 1.0 if winner == self.agent_id else 0.0

        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        assert self._state is not None, "call reset() before action_masks()"
        obs = self._state.get_observation(self.agent_id)
        owned = sum(
            1 for r in self._state.game_map.regions[1:]
            if r is not None and r.owner == self.agent_id
        )
        max_commands = max(1, owned // 3)
        return self.act_encoder.mask(obs, commands_issued=0, max_commands=max_commands)

    def render(self, path: str) -> None:
        assert self._state is not None, "call reset() before render()"
        render_map(self._state, path)

    def _build_opponent(self, opponent_id: int) -> Optional[BaseOpponent]:
        opp_type = getattr(self.config.training, "opponent", None)
        if opp_type == "random":
            return RandomOpponent(player_id=opponent_id)
        if opp_type == "rule":
            return RuleOpponent(player_id=opponent_id)
        return None

    def _init_state(self) -> GameState:
        if self.config.game.capital_mode == "fixed":
            return fixed_capitals(self.config.game.capitals, self.config.game.map_config)
        return random_capitals(self.config.game.num_players, self.config.game.map_config)
