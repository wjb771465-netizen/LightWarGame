from __future__ import annotations

from typing import Any, List

from .reward_function_base import BaseRewardFunction
from .win_lose_reward import WinLoseReward
from .territory_reward import TerritoryReward
from .capital_capture_reward import CapitalCaptureReward
from .step_penalty_reward import StepPenaltyReward


def build_reward_functions(cfg: Any) -> List[BaseRewardFunction]:
    """根据配置构建奖励函数列表。cfg 为 parse_config() 返回的 reward 节点。"""
    return [
        WinLoseReward(win=cfg.win, lose=cfg.lose),
        TerritoryReward(territory_gain=cfg.shaped.territory_gain, territory_loss=cfg.shaped.territory_loss),
        CapitalCaptureReward(capital_capture=cfg.shaped.capital_capture),
        StepPenaltyReward(step_penalty=cfg.shaped.step_penalty),
    ]
