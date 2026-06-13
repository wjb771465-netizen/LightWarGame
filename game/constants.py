"""游戏层全局常量与规则参数。game/ 与 ai/ 统一从此导入，避免魔法数散落。"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# 指令配额
# ---------------------------------------------------------------------------
CMD_PER_TERRITORIES: int = 3   # 每 N 个领地增加 1 条指令配额
MIN_COMMANDS: int = 1          # 最低指令配额


def max_commands(owned_regions: int) -> int:
    """根据领地数计算本回合最大指令数。

    公式: max(MIN_COMMANDS, ceil(owned / CMD_PER_TERRITORIES))

    ceil 语义保证每攒满 CMD_PER_TERRITORIES 个领地立即获得 +1 配额，
    首升仅需 4 地（vs 旧 floor 公式的首升 6 地），更早给扩张正反馈。
    """
    if owned_regions <= 0:
        return 0
    return max(MIN_COMMANDS, math.ceil(owned_regions / CMD_PER_TERRITORIES))


# ---------------------------------------------------------------------------
# 战斗规则
# ---------------------------------------------------------------------------
ISOLATION_PENALTY: float = 0.5  # 围困时到达兵力折半系数
NEUTRAL_GROWTH: int = 1         # 中立地区每回合固定增长兵力
