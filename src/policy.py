"""VAS + CE 策略网络（对齐 RLCascadeRouter 结构）。

动作空间：{stop} ∪ 3 档设备（weak/medium/strong）。
输出 logits = [logit_stop, logit_cont+score_0, logit_cont+score_1, logit_cont+score_2]。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .device_env import CAP_DIM
from .state_encoder import StateEncoder, STATE_DIM


class ComplementarityEncoder(nn.Module):
    """CE：self-attention over {stop} ∪ 设备 tokens → 每设备 score。"""

    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.device_proj = nn.Linear(CAP_DIM, out_dim)
        self.stop_embed = nn.Parameter(torch.randn(out_dim))
        self.attn = nn.MultiheadAttention(embed_dim=out_dim, num_heads=4, batch_first=True)
        self.score_head = nn.Sequential(nn.Linear(out_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, device_embeds: torch.Tensor):
        """device_embeds: [B, N, CAP_DIM] → (scores [B, N], tokens [B, 1+N, out_dim])"""
        B, N, _ = device_embeds.shape
        dev_tok = self.device_proj(device_embeds)                    # [B, N, out_dim]
        stop_tok = self.stop_embed.view(1, 1, -1).expand(B, 1, -1)   # [B, 1, out_dim]
        tokens = torch.cat([stop_tok, dev_tok], dim=1)               # [B, 1+N, out_dim]
        ctx, _ = self.attn(tokens, tokens, tokens)                   # self-attention
        scores = self.score_head(ctx[:, 1:, :]).squeeze(-1)          # [B, N]
        return scores, ctx


class ValueAwareStopper(nn.Module):
    """VAS：Stop Gate（消费 p_fail）+ Critic。"""

    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.stop_gate = nn.Sequential(
            nn.Linear(out_dim + out_dim + 1, 64), nn.ReLU(), nn.Linear(64, 2)
        )
        self.critic = nn.Sequential(
            nn.Linear(out_dim + out_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, h_s: torch.Tensor, pooled: torch.Tensor, p_fail: torch.Tensor):
        """→ ((logit_stop [B,1], logit_cont [B,1]), value [B,1])"""
        stop_logits = self.stop_gate(torch.cat([h_s, pooled, p_fail], dim=-1))  # [B,2]
        logit_stop = stop_logits[:, 0:1]
        logit_cont = stop_logits[:, 1:2]
        value = self.critic(torch.cat([h_s, pooled], dim=-1))
        return (logit_stop, logit_cont), value


class Policy(nn.Module):
    """策略 = Encoder + CE + VAS。"""

    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.encoder = StateEncoder(hidden_dim=128, out_dim=out_dim)
        self.ce = ComplementarityEncoder(out_dim=out_dim)
        self.vas = ValueAwareStopper(out_dim=out_dim)

    def forward(self, s_feats, device_embeds, device_mask, p_fail):
        """s_feats [B, STATE_DIM]; device_embeds [B, N, CAP_DIM]; device_mask [B, N]; p_fail [B, 1].
        → (logits [B, 1+N], value [B, 1])
        """
        h_s = self.encoder(s_feats)                                   # [B, out_dim]
        scores, ctx = self.ce(device_embeds)                          # scores [B,N], ctx [B,1+N,out_dim]
        pooled = ctx.mean(dim=1)                                      # [B, out_dim]
        (logit_stop, logit_cont), value = self.vas(h_s, pooled, p_fail)
        device_logits = logit_cont + scores                           # [B, N]
        logits = torch.cat([logit_stop, device_logits], dim=-1)       # [B, 1+N]
        mask = torch.cat([torch.ones_like(logit_stop), device_mask.float()], dim=-1)
        logits = logits.masked_fill(mask == 0, float("-inf"))
        return logits, value

    @torch.no_grad()
    def act(self, s_feats, device_embeds, device_mask, p_fail, greedy: bool = False):
        logits, value = self.forward(s_feats, device_embeds, device_mask, p_fail)
        if greedy:
            a = logits.argmax(dim=-1)
        else:
            a = torch.distributions.Categorical(logits=logits).sample()
        return a, logits, value