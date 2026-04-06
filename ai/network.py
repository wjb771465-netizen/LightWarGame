# ai/network.py
import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """
    最简 MLP 网络
    输入: (batch, 4, 31) → 124维
    输出: (batch, 1000) + (batch, 1)
    """

    def __init__(self, action_dim: int = 1000):
        super().__init__()

        # 4×31 = 124 维输入
        self.fc1 = nn.Linear(4 * 31, 256)
        self.fc2 = nn.Linear(256, 128)

        # 策略头 (action logits)
        self.policy_head = nn.Linear(128, action_dim)

        # 价值头 (state value)
        self.value_head = nn.Linear(128, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (B, 4, 31)
        x = x.flatten(start_dim=1)  # → (B, 124)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        logits = self.policy_head(x)  # (B, 1000)
        value = self.value_head(x)  # (B, 1)

        return logits, value