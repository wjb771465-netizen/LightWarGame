# rl/reward.py
from typing import Dict, Optional
from game import Command


class RewardCalculator:
    """
    适配：
      - 多命令/回合
      - EOS 后统一执行
      - 回合差分奖励
    """

    def __init__(self):
        # 权重（你可以调参）
        self.w = {
            "valid_action": 0.0,
            "invalid_action": -0.1,

            "capture_region": 5,
            "lose_region": -5,
            "capture_capital": 15,

            "troop_delta": 0.00,  # 兵力变化的缩放系数

            "surround": 2,

            "win": 200,
            "lose": -200,
        }

        # 用于记录上一回合的状态
        self.last_snapshot = None

    # ============================================================
    # step(action)：只对“下命令的有效性”奖励
    # ============================================================
    def calc_step_reward(self, valid: bool) -> float:
        return self.w["valid_action"] if valid else self.w["invalid_action"]

    # ============================================================
    # EOS：回合差分奖励
    # ============================================================
    def calc_eos_reward(self, game_state, player_id) -> float:
        """
        依据：上一回合 state 与当前 state 的变化。
        """

        reward = 0.0

        regions = game_state["regions"]

        # ---- 如果没有 snapshot，说明是第一回合 —— 不给 EOS 奖励
        if self.last_snapshot is None:
            self.last_snapshot = self._snapshot(game_state)
            return 0.0

        prev = self.last_snapshot
        curr = self._snapshot(game_state)

        # ============================================
        # 1. 领地变化奖励
        # ============================================
        for rid in range(1, 32):
            prev_owner = prev["owner"][rid]
            curr_owner = curr["owner"][rid]

            if prev_owner != curr_owner:
                # 玩家占领
                if curr_owner == player_id:
                    reward += self.w["capture_region"]
                    if prev["capital"][rid]:
                        reward += self.w["capture_capital"]

                # 玩家失去
                elif prev_owner == player_id:
                    reward += self.w["lose_region"]


        # ============================================
        # 3. 战略奖励：包围
        # ============================================
        for rid in range(1, 32):
            if curr["owner"][rid] != player_id:  # 只看敌方地
                if self._is_surrounded(rid, game_state):
                    #reward += self.w["surround"]
                    reward += 0

        # 更新 snapshot
        self.last_snapshot = curr

        return reward

    # ============================================================
    # 终局奖励
    # ============================================================
    def calc_final_reward(self, game_state, player_id) -> float:
        from game.utils import check_game_over
        #if not check_game_over(game_state):
            #return 0.0

        cfg = game_state["config"]
        for p in range(1, cfg.num_players + 1):
            if p != player_id:
                # 如果其他玩家没有领地 → 我赢
                has_land = any(r.owner == p for r in game_state["regions"][1:])
                if not has_land:
                    return self.w["win"]

        # 否则我输
        return self.w["lose"]

    # ============================================================
    # 辅助：拍快照，用于比较 state 变化
    # ============================================================
    def _snapshot(self, game_state):
        regions = game_state["regions"]

        owner = {}
        capital = {}
        troops_player = 0
        troops_enemy = 0

        for rid in range(1, 32):
            r = regions[rid]
            owner[rid] = r.owner
            capital[rid] = getattr(r, "is_capital", False)

            if r.owner == 1:
                troops_player += r.troops
            elif r.owner == 2:
                troops_enemy += r.troops

        return {
            "owner": owner,
            "capital": capital,
            "troops_player": troops_player,
            "troops_enemy": troops_enemy,
        }

    # ============================================================
    # 包围检测
    # ============================================================
    def _is_surrounded(self, rid, game_state):
        region = game_state["regions"][rid]
        enemy_id = region.owner
        for adj in region.adjacent:
            if 1 <= adj <= 31:
                if game_state["regions"][adj].owner == enemy_id:
                    return False
        return True
