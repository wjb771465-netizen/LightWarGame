from __future__ import annotations

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from ai.algos.gnn import GNNBackbone


class SpatialExtractor(BaseFeaturesExtractor):
    """flat obs → node_feats + global_feats，子类挂 backbone。

    region 数量和特征维度由 trainer 从 env 动态获取后传入，不硬编码。
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int,
        num_regions: int,
        feat_dim: int,
        global_dim: int,
    ) -> None:
        super().__init__(observation_space, features_dim)
        self.num_regions = num_regions
        self.feat_dim = feat_dim
        self.global_dim = global_dim
        self.backbone: nn.Module | None = None  # 子类赋值

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        node_feats = obs[:, : -self.global_dim].reshape(
            -1, self.num_regions, self.feat_dim
        )
        global_feats = obs[:, -self.global_dim :]
        return self._forward_backbone(node_feats, global_feats)

    def _forward_backbone(
        self, node_feats: torch.Tensor, global_feats: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError("子类实现")


class GNNExtractor(SpatialExtractor):
    """GNN 骨架接入 SB3。edge_index 存为 buffer，全程不变。"""

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
        num_regions: int = 31,
        feat_dim: int = 11,
        global_dim: int = 2,
        edge_index: torch.Tensor | None = None,
        hidden_channels: int = 128,
    ) -> None:
        super().__init__(observation_space, features_dim, num_regions, feat_dim, global_dim)
        self.backbone = GNNBackbone(num_regions, feat_dim,
                                    hidden_channels=hidden_channels,
                                    out_dim=features_dim)
        if edge_index is not None:
            self.register_buffer("edge_index", edge_index)

    def _forward_backbone(
        self, node_feats: torch.Tensor, global_feats: torch.Tensor
    ) -> torch.Tensor:
        assert self.backbone is not None
        return self.backbone(node_feats, self.edge_index, global_feats)
