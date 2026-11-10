"""cost-aware RL（REINFORCE + Critic baseline，简化 PPO）。

回报（延迟） = 任务成功（各步答案正确比例）− λ·累计成本归一化。
"""
from __future__ import annotations

import torch

from .device_env import CAP_EMBED, TIERS, SyntheticBenchmark, make_device_pool, execute_device
from .policy import Policy
from .schema import StepState
from .state_encoder import state_to_features


def rollout(policy: Policy, task, devices, mock: bool = True):
    """跑一条轨迹，返回 (log_probs, 动作, 每步对错, 累计成本, 正确数)。"""
    dev_emb = torch.tensor([CAP_EMBED[t] for t in TIERS], dtype=torch.float32).unsqueeze(0)
    mask = torch.ones(1, len(TIERS))

    log_probs, actions, corrects = [], [], []
    cost = 0.0
    for step in task.steps:
        s = StepState(
            trajectory_summary=[], progress=[0.0, 0.5], last_output="",
            p_fail=0.0, cost_accum=cost, device_states=devices,
            step_idx=len(actions),
        )
        f = state_to_features(s).unsqueeze(0)
        pf = torch.tensor([[s.p_fail]])
        logits, _ = policy.forward(f, dev_emb, mask, pf)   # 带梯度（不可用 no_grad 的 act）
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        lp = dist.log_prob(a)
        log_probs.append(lp); actions.append(a.item())

        if a.item() == 0:  # stop：提前结束，剩余步视为未完成
            break
        tier = TIERS[a.item() - 1]
        out, c, _ = execute_device(tier, [{"role": "user", "content": step.question}], mock=mock)
        cost += c
        corrects.append(1.0 if out.strip() == step.answer.strip() else 0.0)

    # 任务成功 = 正确比例（未执行的步算错）
    total_steps = len(task.steps)
    success = (sum(corrects) + 0.0) / total_steps
    return log_probs, actions, corrects, cost, success


def train_rl(policy: Policy, epochs: int = 5, lr: float = 3e-4, lam: float = 0.5, mock: bool = True):
    bench = SyntheticBenchmark(steps_per_task=3)
    devices = make_device_pool()
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    for ep in range(epochs):
        # 采样一批轨迹
        batch = []
        for t in range(10):
            task = bench.sample_task(f"rl-{ep}-{t}")
            lps, acts, corr, cost, success = rollout(policy, task, devices, mock=mock)
            reward = success - lam * cost
            batch.append((lps, acts, reward))

        # REINFORCE + baseline（用 reward 均值做 baseline）
        rewards = [b[2] for b in batch]
        base = sum(rewards) / len(rewards)
        total_loss = 0.0
        n = 0
        for lps, acts, reward in batch:
            G = reward - base
            for lp in lps:
                total_loss = total_loss - lp * G
                n += 1
        opt.zero_grad()
        total_loss.backward()      # 一次 backward（跨全部轨迹的图）
        opt.step()
        avg_r = sum(rewards) / len(rewards)
        print(f"[RL] epoch {ep} avg_reward={avg_r:.4f} loss={total_loss.item()/max(n,1):.4f}")
    return policy


if __name__ == "__main__":
    torch.manual_seed(0)
    p = Policy()
    train_rl(p, epochs=3, mock=True)
    print("RL 完成")