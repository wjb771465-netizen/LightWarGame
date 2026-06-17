from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv


def adj_to_edge_index(adj: np.ndarray) -> torch.Tensor:
    """邻接矩阵 (N,N) → PyG edge_index (2, E)。"""
    edges = np.argwhere(adj > 0.5)
    return torch.tensor(edges.T, dtype=torch.long)


class GNNBackbone(nn.Module):
    """2 层 GraphSAGE + mean pool + global concat → 固定维度 latent。

    不依赖游戏逻辑和 SB3，可独立测试。
    """

    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        hidden_channels: int = 128,
        out_dim: int = 256,
    ) -> None:
        super().__init__()
        self._num_nodes = num_nodes
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.head = nn.Linear(hidden_channels + 2, out_dim)
        self.act = nn.ReLU()

    def forward(
        self,
        node_feats: torch.Tensor,    # (B, N, F)
        edge_index: torch.Tensor,    # (2, E)
        global_feats: torch.Tensor,  # (B, 2)
    ) -> torch.Tensor:               # (B, out_dim)
        B = node_feats.shape[0]
        N = self._num_nodes
        x = node_feats.reshape(B * N, -1)
        batch_edge = self._batch_edge_index(edge_index, B, N)

        x = self.act(self.conv1(x, batch_edge))
        x = self.act(self.conv2(x, batch_edge))

        x = x.view(B, N, -1).mean(dim=1)          # (B, hidden_channels)
        x = torch.cat([x, global_feats], dim=-1)   # (B, hidden_channels + 2)
        return self.head(x)

    @staticmethod
    def _batch_edge_index(
        edge_index: torch.Tensor,
        batch_size: int,
        num_nodes: int,
    ) -> torch.Tensor:
        """把单图 edge_index 复制为 batch 模式，逐样本 offset 节点索引。"""
        offsets = torch.arange(batch_size, device=edge_index.device) * num_nodes
        return torch.cat([edge_index + off for off in offsets], dim=1)
