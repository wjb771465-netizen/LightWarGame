from __future__ import annotations

from typing import Protocol

import numpy as np
from sb3_contrib import MaskablePPO


class Policy(Protocol):
    def predict(self, obs: np.ndarray, mask: np.ndarray) -> int: ...


class SB3Policy:
    """将 MaskablePPO checkpoint 包装成统一 Policy 接口。"""

    def __init__(self, model_path: str) -> None:
        self._model = MaskablePPO.load(model_path)

    @property
    def obs_dim(self) -> int:
        """模型期望的观测向量长度，用于反推 max_players。"""
        return int(self._model.observation_space.shape[0])

    def predict(self, obs: np.ndarray, mask: np.ndarray) -> int:
        action, _ = self._model.predict(obs, action_masks=mask, deterministic=True)
        return int(action)
