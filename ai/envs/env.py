# ai/envs/env.py
from typing import List
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.spaces import Dict, Box
import random

from game import (
    init_game_auto,
    move_troops,
    neutral_regions_growth,
    process_troop_growth,
    check_game_over,
    Command,
    is_command_valid
)

from game.fog_of_war import create_fog_view_for_player
from game.display import display_fog_game_state

from .observation import FogToGraphFeatures, get_edge_index
from .action import ActionSpace
from .reward import RewardCalculator
from .opponent import RandomOpponent

class ChineseWarGameEnv(gym.Env):
    """

    """

    metadata = {"render_modes": []}

    def __init__(self, player_id: int = 1):
        super().__init__()
        self.player_id = player_id
        self.opponent_id = 2 if player_id == 1 else 1

        # ---- Graph observation ----
        self.num_regions = 31
        self.num_node_features = 4

        self.obs_maker = FogToGraphFeatures()
        self.edge_index = get_edge_index()

        # ---- Action ----
        self.action_handler = ActionSpace(max_actions=3200, max_commands_per_turn=8)
        self.action_space = spaces.Discrete(self.action_handler.max_actions)

        # ---- Observation ----
        self.observation_space = Dict({
            "node_features": Box(0, 1, shape=(self.num_regions, self.num_node_features), dtype=np.float32),
            "edge_index": Box(0, self.num_regions - 1, shape=self.edge_index.shape, dtype=np.int64),
        })

        # Reward
        self.reward_calc = RewardCalculator()

        # State
        self.game_state = init_game_auto()
        self.cmds_buffer = []
        self.turn = 0

        self.step_num=0

        # Opponent
        self.opponent = RandomOpponent(max_actions_per_turn=8)

    # ============================================================
    # reset
    # ============================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game_state = init_game_auto()
        self.turn = self.game_state.get("turn", 0)
        self.cmds_buffer = []

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    # ============================================================
    # 核心：step(action)
    # ============================================================
    def step(self, action: int):
        self.step_num += 1

        terminated = False
        truncated = False
        reward = 0.0

        decoded = self.action_handler.decode(action, self.game_state, self.player_id)
        if isinstance(decoded, Command):
            self.cmds_buffer.append(decoded)
            # 此时不执行，只给轻微奖励
            reward = self.reward_calc.calc_step_reward(True)

        else:
            reward = self.reward_calc.calc_step_reward(False)
        # ------------------------
        # 1) EOS: 执行全回合逻辑
        # ------------------------
        if  len(self.cmds_buffer) >= 8 or decoded == "EOS":
            # 生成对手命令
            self.cmds_buffer += self.opponent.sample_commands(self.game_state, self.opponent_id)

            # 执行所有缓存命令（一次性）
            self._process_all_commands()

            # 中立增长
            self.game_state = neutral_regions_growth(self.game_state)


            # 回合数 +1
            self.game_state["turn"] += 1
            self.turn = self.game_state["turn"]

            # 清空 command buffer
            self.cmds_buffer = []

            # 回合奖励
            reward = self.reward_calc.calc_eos_reward(self.game_state, self.player_id)

            # 结束判断
            terminated = check_game_over(self.game_state)
            truncated = self.turn >= 50

            if truncated:
                self.step_num = 0
                fog_view = create_fog_view_for_player(self.game_state, self.player_id)
                #display_fog_game_state(fog_view)
            if terminated:
                reward += self.reward_calc.calc_final_reward(self.game_state, self.player_id)

        # ------------------------
        # 2) 普通命令：加入命令缓存
        # ------------------------


        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    # ============================================================
    # 一次性执行所有命令（符合你的原游戏机制）
    # ============================================================
    def _process_all_commands(self):
        self.game_state = process_troop_growth(self.game_state)
        for idx, cmd in enumerate(self.cmds_buffer, 1):
            ok = is_command_valid(self.game_state, cmd.source, cmd.target, cmd.troops, cmd.player)
            if not ok:
                #要加惩罚
                continue
            self.game_state = move_troops(self.game_state, cmd.source, cmd.target, cmd.troops, cmd.player)

    # ============================================================
    # 动作掩码：考虑当前命令数
    # ============================================================
    def _get_action_mask(self):
        return self.action_handler.get_mask(
            self.game_state,
            self.player_id,
            current_cmd_count=len(self.cmds_buffer)
        )

    def _get_info(self):
        return {"action_masks": self._get_action_mask()}

    # ============================================================
    # 观测
    # ============================================================
    def _get_obs(self):
        node_features = self.obs_maker(self.game_state, self.player_id)
        return {
            "node_features": node_features,
            "edge_index": self.edge_index
        }


