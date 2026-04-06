# ai/algorithm/gnn_policy.py
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

# NOTE: ensure sb3_contrib is installed and up-to-date

class GNNFeatureExtractor(BaseFeaturesExtractor):
    """
    Simple GCN-style feature extractor for Dict observation:
    Expects observation to be a dict with keys:
      - "node_features": float tensor shape (batch, N, F)
      - "edge_index": int tensor shape (2, E) OR (batch, 2, E)
    Output: (batch, features_dim)
    """
    def __init__(self, observation_space, features_dim: int = 256):
        # Determine input shapes from observation_space
        # observation_space is gym.spaces.Dict with node_features and edge_index
        super().__init__(observation_space, features_dim)

        # read node shape
        node_space = observation_space.spaces["node_features"]
        self.N = node_space.shape[0]
        self.F_in = node_space.shape[1]

        # feature dims
        self.hidden_dim = 128
        self.out_dim = features_dim

        # GCN-like linear transforms
        self.lin1 = nn.Linear(self.F_in, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.hidden_dim)

        # final MLP after pooling
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.out_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        node_feats = observations["node_features"]  # (B, N, F_in)
        edge_index = observations["edge_index"]

        if node_feats.dim() == 2:
            node_feats = node_feats.unsqueeze(0)

        B = node_feats.size(0)
        N = node_feats.size(1)
        F_in = node_feats.size(2)  # 改名！不要用 F

        if edge_index.dim() == 2:
            edge_index = edge_index.unsqueeze(0).repeat(B, 1, 1)

        _, _, E = edge_index.shape
        device = node_feats.device

        adj = th.zeros((B, N, N), device=device, dtype=node_feats.dtype)
        edge_index = edge_index.long()

        for b in range(B):
            ei = edge_index[b]
            srcs = ei[0]
            tgts = ei[1]
            adj[b].index_put_((srcs, tgts), th.ones_like(srcs, dtype=node_feats.dtype), accumulate=True)
            adj[b] = adj[b] + adj[b].t()

        idx = th.arange(N, device=device)
        adj[:, idx, idx] = adj[:, idx, idx] + 1.0

        deg = adj.sum(dim=-1, keepdim=True)
        deg[deg == 0] = 1.0
        adj_norm = adj / deg

        h = node_feats
        h = F.relu(self.lin1(h))  # 现在 F 是 functional
        h = th.matmul(adj_norm, h)  # 直接用 th.matmul
        h = F.relu(self.lin2(h))
        h = th.matmul(adj_norm, h)

        h_pool = h.mean(dim=1)
        features = self.mlp(h_pool)
        return features

def torch_bmm(A, B):
    # A: (B, N, N), B: (B, N, D) -> result (B, N, D)
    # implement via torch.matmul with transpose if necessary
    return th.matmul(A, B)

# -----------------------------
# Policy class
# -----------------------------
class CustomGNNPolicy(MaskableActorCriticPolicy):
    """
    Custom policy that uses the GNNFeatureExtractor.
    In MaskablePPO, passing this policy class will allow the algorithm to use the provided
    features extractor for observations of dict type.
    """
    def __init__(self, *args, **kwargs):
        # override features_extractor_class to use our GNNFeatureExtractor
        kwargs["features_extractor_class"] = GNNFeatureExtractor
        # choose features_dim (final feature vector size)
        kwargs["features_extractor_kwargs"] = dict(features_dim=256)
        super().__init__(*args, **kwargs)
