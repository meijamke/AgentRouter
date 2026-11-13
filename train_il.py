"""IL 冷启动：标签 = 最便宜够用设备（答案表 GT），交叉熵训网络——只当"不崩"初始化。"""
from __future__ import annotations

import torch
import torch.nn as nn

from answer_table import AnswerTable
from policy import Policy
from schema import ACTION_SPACE, DEVICES, DeviceState, StepState
from state_encoder import state_to_features


def tier_to_action_index(tier: str, exec_type: str = "finalize") -> int:
    for i, (label, d, e, t) in enumerate(ACTION_SPACE):
        if d == tier and e == exec_type:
            return i
    return 0


def train_il(policy: Policy, table: AnswerTable, tasks: list, epochs: int = 5, lr: float = 1e-3):
    devices = [DeviceState(d, [0.2] * 4) for d in DEVICES]
    data = []
    for task in tasks:
        tier = table.cheapest_correct(task.task_id)
        label = tier_to_action_index(tier, task.exec_type)   # 该执行类型下最便宜够用设备的动作
        s = StepState(device_states=devices)
        data.append((s, label))

    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(epochs):
        tot, n = 0.0, 0
        for s, label in data:
            f = torch.tensor(state_to_features(s)).unsqueeze(0)
            logits = policy.forward(f)
            loss = loss_fn(logits, torch.tensor([label]))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); n += 1
        print(f"  IL epoch {ep} loss={tot/n:.4f}")
    return policy