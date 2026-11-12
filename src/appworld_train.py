"""AppWorld 真实 IL + RL 训练。

IL 标签 = 「最便宜够用设备」：依次试 weak→medium→strong，每个设备的代码
**在新 AppWorld 环境里真正执行**，取第一个「执行后任务真的成功」(evaluate().success)
的设备——真执行标签，非静态代理。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .appworld_env import AppworldEnv, run_episode_grad
from .device_env import CAP_EMBED, TIERS, execute_device, make_device_pool
from .policy import Policy
from .schema import StepState
from .state_encoder import state_to_features


def il_label(env: AppworldEnv) -> int:
    """返回最便宜够用设备：最便宜且其代码执行后任务真成功。"""
    from appworld import AppWorld
    prompt = env.prompt()
    for tier in TIERS:                          # weak → medium → strong
        code, _, _ = execute_device(tier, [{"role": "user", "content": prompt}], mock=False)
        fresh = AppWorld(env.task_id)           # 每个设备用全新环境（execute 会改状态）
        try:
            fresh.execute(code.strip())
            success = bool(fresh.evaluate().success)
        except Exception:
            success = False
        finally:
            fresh.close()
        if success:
            return TIERS.index(tier)
    return TIERS.index("strong")


def make_state(devices, step_idx=0) -> StepState:
    return StepState(
        trajectory_summary=[], progress=[0.0, 0.5], last_output="",
        p_fail=0.0, cost_accum=0.0, device_states=devices, step_idx=step_idx,
    )


def train_il(policy: Policy, task_ids, devices, epochs: int = 3, lr: float = 1e-3):
    print("=== IL 冷启动（真执行标签）===")
    data = []
    for tid in task_ids:
        env = AppworldEnv(tid)
        label = il_label(env)
        data.append((make_state(devices), label))
        env.close()
        print(f"  {tid}: 最便宜够用设备 = {TIERS[label]}")

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
            loss = loss_fn(logits, torch.tensor([label + 1]))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); n += 1
        print(f"  IL epoch {ep} loss={tot/n:.4f}")
    return policy


def train_rl(policy: Policy, task_ids, devices, epochs: int = 3, lam: float = 0.3):
    """REINFORCE：reward = task_success − λ·累计成本。"""
    print("=== cost-aware RL ===")
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    for ep in range(epochs):
        batch = []
        for tid in task_ids:
            env = AppworldEnv(tid)
            lps, acts, success, cost, steps = run_episode_grad(policy, env, devices, max_steps=4)
            reward = (1.0 if success else 0.0) - lam * cost
            batch.append((lps, reward, success, cost))
            env.close()

        base = sum(b[1] for b in batch) / len(batch)
        total_loss = 0.0
        n = 0
        for lps, reward, success, cost in batch:
            G = reward - base
            for lp in lps:
                total_loss = total_loss - lp * G
                n += 1
        opt.zero_grad(); total_loss.backward(); opt.step()

        succ = sum(1 for b in batch if b[2]) / len(batch)
        avg_c = sum(b[3] for b in batch) / len(batch)
        print(f"  RL epoch {ep} succ={succ:.2f} cost={avg_c:.3f} loss={total_loss.item()/max(n,1):.4f}")
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
    train_rl(policy, ids, devices, epochs=2)
    print("AppWorld IL+RL 演示完成")