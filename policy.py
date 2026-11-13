"""策略网络：一个网络，直接输出统一执行决策动作空间的联合分布。

不拆 VAS/CE 两个头——"选设备"和"选执行类型"是同一个决策，扁平化 softmax 直接出
NUM_ACTIONS(=10) 个动作的概率。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from schema import NUM_ACTIONS
from state_encoder import STATE_DIM


class Policy(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, NUM_ACTIONS),   # 10 个动作 logit
        )

    def forward(self, state_features: torch.Tensor) -> torch.Tensor:
        """state_features: [B, state_dim] → [B, NUM_ACTIONS] logits。"""
        return self.net(state_features)

    def act(self, state_features: torch.Tensor, mask=None, greedy: bool = False) -> torch.Tensor:
        logits = self.forward(state_features)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        if greedy:
            return logits.argmax(dim=-1)
        return torch.distributions.Categorical(logits=logits).sample()