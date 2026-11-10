"""状态编码器：StepState → 定长特征 → h_s（d 维）。

M1 用固定 3 档设备池，故特征定长。动态设备池（变长）是 M2/M3 的事。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .device_env import CAP_DIM
from .schema import StepState

N_DEVICES = 3                       # M1 固定 3 档（weak/medium/strong）
PROGRESS_DIM = 2                    # 任务进度定长（完成度、剩余难度）
# 每台设备贡献：cap_embed(CAP_DIM) + battery/100 + bw/1000
DEVICE_FEAT_DIM = CAP_DIM + 2
# 全局：progress(PROGRESS_DIM) + p_fail(1) + cost(1) + step(1)
GLOBAL_FEAT_DIM = PROGRESS_DIM + 3
STATE_DIM = GLOBAL_FEAT_DIM + N_DEVICES * DEVICE_FEAT_DIM


def state_to_features(s: StepState, max_steps: int = 20) -> torch.Tensor:
    """把 StepState 转成定长特征向量（无参，手工特征）。"""
    feats: list = []

    # progress：pad/截断到 PROGRESS_DIM
    prog = list(s.progress) + [0.0] * PROGRESS_DIM
    feats += prog[:PROGRESS_DIM]

    feats += [s.p_fail, s.cost_accum, float(s.step_idx) / max_steps]

    # 设备池：固定 N_DEVICES，不足补零（M1 恒为 3）
    for i in range(N_DEVICES):
        if i < len(s.device_states):
            d = s.device_states[i]
            feats += list(d.cap_embed)[:CAP_DIM]
            feats += [d.battery_pct / 100.0, d.network_bw_mbps / 1000.0]
        else:
            feats += [0.0] * DEVICE_FEAT_DIM

    return torch.tensor(feats, dtype=torch.float32)


class StateEncoder(nn.Module):
    """轻量投影：定长特征 → h_s（d 维）。"""

    def __init__(self, hidden_dim: int = 128, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, STATE_DIM] → [B, out_dim]"""
        return self.net(x)
