# ai/envs/action.py
import numpy as np
from typing import List, Optional
from game import Command


class ActionSpace:
    """
    动作空间：0~999出兵动作 + 1000 = EOS（结束本回合）
    编码方式:
        old_action = src*100 + tgt*3 + ratio_idx
        new_action = old_action
        EOS = 1000
    """
    def __init__(self, max_actions: int = 3200, max_commands_per_turn=8):
        self.old_max = max_actions
        self.EOS = max_actions  # EOS = 1000
        self.max_actions = max_actions + 1
        self.max_commands_per_turn = max_commands_per_turn

        self.ratios = [1 / 3, 1 / 2, 2 / 3]
        self.num_ratios = len(self.ratios)

    # --------------------------
    # 编码 / 解码
    # --------------------------
    def encode(self, src: int, tgt: int, ratio_idx: int) -> int:
        """保持原编码（0–999），不加偏移"""
        return src * 100 + tgt * 3 + ratio_idx

    def decode(self, action_id: int, game_state: dict, player_id: int):
        """解码动作或 EOS"""
        if action_id == self.EOS:
            return "EOS"

        if action_id < 0 or action_id >= self.old_max:
            return None

        raw = action_id
        src = raw // 100
        tgt = (raw % 100) // 3
        ratio_idx = raw % 3

        regions = game_state["regions"]

        if src < 1 or src >= 32 or tgt < 1 or tgt >= 32:
            return None
        if regions[src].owner != player_id:
            return None
        if tgt not in regions[src].adjacent:
            return None

        r = regions[src]
        if r.troops <= 1:
            return None

        ratio = self.ratios[ratio_idx]
        troops = max(1, int(r.troops * ratio))
        if troops >= r.troops:
            return None

        return Command(source=src, target=tgt, troops=troops, player=player_id)

    # --------------------------
    # 合法动作
    # --------------------------
    def get_valid_actions(self, game_state: dict, player_id: int):
        valid = []

        regions = game_state["regions"]
        my_regions = [i for i in range(1, 32) if regions[i].owner == player_id]

        for src in my_regions:
            r = regions[src]
            if r.troops <= 1:
                continue
            for tgt in r.adjacent:
                if not (1 <= tgt < 32):
                    continue
                for ratio_idx in range(self.num_ratios):
                    ratio = self.ratios[ratio_idx]
                    troops = max(1, int(r.troops * ratio))
                    if troops >= r.troops:
                        continue

                    aid = self.encode(src, tgt, ratio_idx)
                    if 0 <= aid < self.old_max:
                        valid.append(aid)

        # EOS 永远合法（用于结束回合）
        valid.append(self.EOS)

        return valid

    # --------------------------
    # 动作 mask
    # --------------------------
    def get_mask(self, game_state: dict, player_id: int, current_cmd_count):
        mask = np.zeros(self.max_actions, dtype=bool)
        #if current_cmd_count >= self.max_commands_per_turn:
            # 超出命令数：只能 EOS
            #return mask

        for aid in self.get_valid_actions(game_state, player_id):
            mask[aid] = True

        return mask

