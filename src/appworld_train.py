"""AppWorld 真实 IL + RL 训练（小规模演示）。

IL 标签 = 「最便宜够用设备」：依次试 weak→medium→strong，取第一个
生成的代码**不含禁用模块**（import requests/os/… 会被 AppWorld 沙盒拒绝，
即"不会正确用 App 的 API"）者——这是"API 使用正确性"的廉价静态代理。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .appworld_env import AppworldEnv
from .device_env import CAP_EMBED, TIERS, execute_device, make_device_pool
from .policy import Policy
from .schema import StepState
from .state_encoder import state_to_features

FORBIDDEN = ["import requests", "import os", "import subprocess",
             "import socket", "import urllib", "import http", "import sys", "open("]


def uses_forbidden(code: str) -> bool:
    return any(f in code for f in FORBIDDEN)


def il_label(env: AppworldEnv) -> int:
    """返回最便宜够用设备的档位索引（代码不含禁用模块 = 会正确用 API）。"""
    prompt = env.prompt()
    for tier in TIERS:                      # weak → medium → strong
        code, _, _ = execute_device(tier, [{"role": "user", "content": prompt}], mock=False)
        if not uses_forbidden(code):
            return TIERS.index(tier)
    return TIERS.index("strong")


def make_state(devices) -> StepState:
    return StepState(
        trajectory_summary=[], progress=[0.0, 0.5], last_output="",
        p_fail=0.0, cost_accum=0.0, device_states=devices, step_idx=0,
    )


def train_il(policy: Policy, task_ids, devices, epochs: int = 2, lr: float = 1e-3):
    print("=== IL 冷启动（真实 API + AppWorld）===")
    data = []
    for tid in task_ids:
        env = AppworldEnv(tid)
        label = il_label(env)
        s = make_state(devices)
        data.append((s, label))
        env.close()
        print(f"  {tid}: 指令={env.instruction[:40]!r} → 最便宜够用设备={TIERS[label]}")

    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    dev_emb = torch.tensor([CAP_EMBED[t] for t in TIERS], dtype=torch.float32).unsqueeze(0)
    mask = torch.ones(1, len(TIERS))
    for ep in range(epochs):
        tot, n = 0.0, 0
        for s, label in data:
            f = state_to_features(s).unsqueeze(0)
            pf = torch.tensor([[s.p_fail]])
            logits, _ = policy(f, dev_emb, mask, pf)
            loss = loss_fn(logits, torch.tensor([label + 1]))   # 动作 0=stop，设备 i=i+1
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); n += 1
        print(f"  IL epoch {ep} loss={tot/n:.4f}")
    return policy


if __name__ == "__main__":
    import os
    os.chdir(r"D:\dsh_space\跨设备模型路由代码")
    from appworld import load_task_ids
    torch.manual_seed(0)
    ids = load_task_ids("train")[:2]
    devices = make_device_pool()
    policy = Policy()
    train_il(policy, ids, devices, epochs=2)
    print("IL 演示完成（RL 复用 train_rl.py 的 shaping 策略梯度，把 benchmark 换成 AppworldEnv 即可）")