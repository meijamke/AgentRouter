"""cost-aware RL（REINFORCE + baseline，简化 PPO）。

奖励（v19 A1）：延迟主奖励 + 增量 shaping —— 每步 r_t = U(s_{t+1}) − U(s_t)，
U = α·质量̂(当前) + (1−α)·成本节省̂(累计)，Σ r_t 望远镜求和 = 终态效用。
"""
from __future__ import annotations

import torch

from .device_env import CAP_EMBED, TIERS, SyntheticBenchmark, make_device_pool, execute_device
from .policy import Policy
from .schema import StepState
from .state_encoder import state_to_features


def rollout(policy: Policy, task, devices, mock: bool = True, alpha: float = 0.5):
    """跑一条轨迹，返回 (log_probs, 动作, 每步 shaped 奖励)。"""
    dev_emb = torch.tensor([CAP_EMBED[t] for t in TIERS], dtype=torch.float32).unsqueeze(0)
    mask = torch.ones(1, len(TIERS))

    log_probs, actions, shaped_rewards = [], [], []
    cost = 0.0
    n_correct = 0.0
    n_total = len(task.steps)
    C_max = len(TIERS) * 1.0   # 归一化常数（3 设备 × 成本 1.0）

    def utility(nc: float, cc: float) -> float:
        q = nc / n_total if n_total else 0.0          # 质量̂：累计正确率
        cs = 1.0 - cc / C_max                         # 成本节省̂
        return alpha * q + (1.0 - alpha) * cs

    for step in task.steps:
        U_before = utility(n_correct, cost)

        s = StepState(
            trajectory_summary=[], progress=[0.0, 0.5], last_output="",
            p_fail=0.0, cost_accum=cost, device_states=devices,
            step_idx=len(actions),
        )
        f = state_to_features(s).unsqueeze(0)
        pf = torch.tensor([[s.p_fail]])
        logits, _ = policy.forward(f, dev_emb, mask, pf)   # 带梯度
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        lp = dist.log_prob(a)
        log_probs.append(lp); actions.append(a.item())

        if a.item() == 0:  # stop：提前结束，剩余步视为未完成
            break
        tier = TIERS[a.item() - 1]
        out, c, _ = execute_device(tier, [{"role": "user", "content": step.question}], mock=mock)
        cost += c
        if out.strip() == step.answer.strip():
            n_correct += 1.0

        U_after = utility(n_correct, cost)
        shaped_rewards.append(U_after - U_before)   # 增量 shaping

    return log_probs, actions, shaped_rewards, cost, (n_correct / n_total)


def train_rl(policy: Policy, epochs: int = 5, lr: float = 3e-4, mock: bool = True):
    bench = SyntheticBenchmark(steps_per_task=3)
    devices = make_device_pool()
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    for ep in range(epochs):
        batch = []
        for t in range(10):
            task = bench.sample_task(f"rl-{ep}-{t}")
            lps, acts, shaped, cost, success = rollout(policy, task, devices, mock=mock)
            # returns-to-go（望远镜求和后即各步累计未来 shaped 奖励）
            returns = []
            G = 0.0
            for r in reversed(shaped):
                G = G + r
                returns.insert(0, G)
            batch.append((lps, returns, success, cost))

        # baseline = 各轨迹"平均每步回报"的均值
        flat = [G for _, rets, _, _ in batch for G in rets]
        base = sum(flat) / len(flat) if flat else 0.0

        total_loss = 0.0
        n = 0
        for lps, returns, success, cost in batch:
            for lp, G in zip(lps, returns):
                total_loss = total_loss - lp * (G - base)
                n += 1
        opt.zero_grad()
        total_loss.backward()      # 一次 backward
        opt.step()

        avg_succ = sum(b[2] for b in batch) / len(batch)
        avg_cost = sum(b[3] for b in batch) / len(batch)
        print(f"[RL] epoch {ep} succ={avg_succ:.3f} cost={avg_cost:.3f} loss={total_loss.item()/max(n,1):.4f}")
    return policy


if __name__ == "__main__":
    torch.manual_seed(0)
    p = Policy()
    train_rl(p, epochs=3, mock=True)
    print("RL 完成")