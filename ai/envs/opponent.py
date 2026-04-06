# ai/envs/opponent.py
import random
from typing import List
from game import Command

class RandomOpponent:
    """
    随机对手：在给定 game_state 的快照上随机挑选 up to n 个命令（只生成，不执行）。
    - 不修改传入的 game_state
    - 生成的命令保证来源于原始拥有者和原始兵力（基于 snapshot）
    """

    def __init__(self, max_actions_per_turn: int = 8):
        self.max_actions = max_actions_per_turn

    def sample_commands(self, game_state: dict, player_id: int) -> List[Command]:
        """
        生成至多 n 条命令（如果 n 为 None 则使用 self.max_actions）。
        返回 Command 列表，但不执行 move_troops。
        """
        n = self.max_actions

        regions = game_state["regions"]
        # 先构建候选源区列表（基于当前 snapshot）
        my_regions = [i for i in range(1, len(regions)) if regions[i].owner == player_id]

        commands = []
        if not my_regions:
            return commands

        # 为避免每次选择都受 earlier choices 影响（source 源自 snapshot），
        # 我们在选择命令时仅参考 snapshot 的 owner/troops，不改变 snapshot。
        for _ in range(n):
            # 重新过滤一次（因为某些 region 可能兵力 <=1）
            valid_srcs = [s for s in my_regions if regions[s].troops > 1]
            if not valid_srcs:
                break

            src = random.choice(valid_srcs)
            r = regions[src]

            # 选 target：只在 snapshot 邻接内选
            valid_targets = [t for t in r.adjacent if 1 <= t < len(regions)]
            if not valid_targets:
                # 此次选择跳过，但不终止整个循环
                continue

            tgt = random.choice(valid_targets)

            # 出兵数基于 snapshot 的 troops，确保至少 1 且 < src.troops
            max_send = max(1, r.troops - 1)
            if max_send <= 0:
                continue

            troops = random.randint(1, max_send)

            cmd = Command(source=src, target=tgt, troops=troops, player=player_id)
            commands.append(cmd)

            # 注意：**不要修改 regions[src].troops** —— 我们只在 snapshot 上采样，
            # 真实结算会在 env 的统一执行阶段完成。

        return commands
