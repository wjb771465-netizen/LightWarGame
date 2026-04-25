from __future__ import annotations

import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Set

import yaml

from game.datatypes.state import GameState


@dataclass
class StateSnapshot:
    """GameState 的轻量快照，供奖励函数计算 prev→curr 差值。

    owners[i]  — region i 的 owner（1-indexed，owners[0] 不使用）
    capitals   — 所有 is_capital=True 的 region id 集合
    """

    owners: List[int]
    capitals: Set[int]

    @classmethod
    def from_state(cls, state: GameState) -> StateSnapshot:
        owners = [0]
        capitals: Set[int] = set()
        for i, r in enumerate(state.game_map.regions[1:], 1):
            owners.append(r.owner if r is not None else 0)
            if r is not None and r.is_capital:
                capitals.add(i)
        return cls(owners=owners, capitals=capitals)


def parse_config(config_name: str) -> SimpleNamespace:
    """按路径名加载 YAML 配置，返回递归 SimpleNamespace（支持 cfg.reward.win 访问）。

    Args:
        config_name: 相对于 ai/envs/config/ 的路径，不含 .yaml 后缀。
                     例如 "two_players/vsbaseline"

    Returns:
        SimpleNamespace，嵌套 dict 均递归转换为属性访问对象。
    """
    config_dir = os.path.join(os.path.dirname(__file__), "configs")
    filepath = os.path.join(config_dir, f"{config_name}.yaml")
    assert os.path.exists(filepath), f"config not found: {filepath}"
    with open(filepath, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return _to_namespace(data)


def _to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj
