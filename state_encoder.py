"""状态编码：StepState → 固定维特征向量（无 P(fail)，quality-estimator-free）。"""
from __future__ import annotations

import numpy as np

from schema import StepState

# 特征维度：进度(2) + 成本(1) + 步数(1) + λ(1) + 上一步输出长度(1) + 设备可执行(3) = 9
STATE_DIM = 9


def state_to_features(s: StepState) -> np.ndarray:
    f = []
    f.extend(s.progress)                          # 2：进度
    f.append(s.cost_accum)                        # 1：累计成本
    f.append(min(s.step_idx / 3.0, 1.0))          # 1：步数归一化（T_max=3）
    f.append(s.lam)                               # 1：λ 偏好
    f.append(min(len(s.last_output) / 200.0, 1.0))  # 1：上一步输出长度
    for d in s.device_states:                     # 3：设备是否可执行
        f.append(1.0 if (d.online and d.can_execute) else 0.0)
    while len(f) < STATE_DIM:
        f.append(0.0)
    return np.asarray(f[:STATE_DIM], dtype=np.float32)