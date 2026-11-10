"""IL 冷启动：黄金轨迹上监督学习 VAS/CE。

标签 = 每个 step 的"最便宜够用设备"：依次试 weak→medium→strong，
取第一个和 oracle(strong) 答案一致的档位，作为该 step 的路由标签。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .device_env import (
    CAP_EMBED, TIERS, SyntheticBenchmark, make_device_pool, execute_device,
)
from .policy import Policy
from .schema import StepState
from .state_encoder import state_to_features


def compute_il_label(question: str, mock: bool = True) -> int:
    """返回该 step 的最便宜够用档位索引（TIERS 下标）。"""
    oracle = execute_device("strong", [{"role": "user", "content": question}], mock=mock)[0]
    for tier in TIERS:  # weak → medium → strong
        ans = execute_device(tier, [{"role": "user", "content": question}], mock=mock)[0]
        if ans == oracle:
            return TIERS.index(tier)
    return TIERS.index("strong")


def collect_golden(valid_tiers: list, mock: bool = True):
    """收集 (状态, 路由标签) 对。valid_tiers 为允许的路由档位子集。"""
    bench = SyntheticBenchmark(steps_per_task=3)
    devices = make_device_pool()
    data = []
    for t in range(20):
        task = bench.sample_task(f"t{t}")
        for i, step in enumerate(task.steps):
            label = compute_il_label(step.question, mock=mock)
            s = StepState(
                trajectory_summary=[],
                progress=[i / len(task.steps), 0.5],
                last_output="",
                p_fail=0.0,
                cost_accum=0.0,
                device_states=devices,
                step_idx=i,
            )
            data.append((s, label))
    return data


def train_il(policy: Policy, epochs: int = 3, lr: float = 1e-3, mock: bool = True):
    data = collect_golden(TIERS, mock=mock)
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
            target = torch.tensor([label + 1])  # action 0 = stop，设备 i → i+1
            loss = loss_fn(logits, target)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); n += 1
        print(f"[IL] epoch {ep} loss={tot/n:.4f}")
    return policy


if __name__ == "__main__":
    torch.manual_seed(0)
    p = Policy()
    train_il(p, epochs=2, mock=True)
    print("IL 完成")