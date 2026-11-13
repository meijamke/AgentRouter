"""cost-aware RL：查表采样 + REINFORCE + shaping（奖励=成功−λ·成本）。"""
from __future__ import annotations

import torch

from answer_table import AnswerTable
from policy import Policy
from schema import ACTION_SPACE, DEVICES, DeviceState, StepState
from state_encoder import state_to_features

T_MAX = 3   # = 设备数


def rollout(policy: Policy, task, table: AnswerTable, devices, greedy: bool = False):
    """查表采样一条轨迹，返回 (log_probs, actions, responses, success, cost)。"""
    log_probs, actions, responses = [], [], []
    cost = 0.0
    for step_idx in range(T_MAX):
        s = StepState(device_states=devices, step_idx=step_idx, cost_accum=cost)
        f = torch.tensor(state_to_features(s)).unsqueeze(0)
        logits = policy.forward(f)
        dist = torch.distributions.Categorical(logits=logits)
        a = logits.argmax(-1) if greedy else dist.sample()
        idx = a.item()
        label, tier, exec_type, tool = ACTION_SPACE[idx]
        if idx == 0:   # stop
            break
        correct, c = table.lookup(task.task_id, tier)
        cost += c
        log_probs.append(dist.log_prob(a))
        actions.append(idx)
        responses.append(correct)
    success = bool(responses and responses[-1])   # 最后一步（收尾动作）答对
    return log_probs, actions, responses, success, cost


def train_rl(policy: Policy, table: AnswerTable, tasks: list, devices, epochs: int = 5,
             lam: float = 0.5, lr: float = 3e-4):
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    for ep in range(epochs):
        batch = []
        for task in tasks:
            lps, acts, resps, success, cost = rollout(policy, task, table, devices)
            reward = (1.0 if success else 0.0) - lam * cost
            batch.append((lps, reward, success, cost))

        base = sum(b[1] for b in batch) / len(batch)
        total_loss, n = 0.0, 0
        for lps, reward, success, cost in batch:
            G = reward - base
            for lp in lps:
                total_loss = total_loss - lp * G
                n += 1
        opt.zero_grad(); total_loss.backward(); opt.step()

        succ = sum(1 for b in batch if b[2]) / len(batch)
        avg_c = sum(b[3] for b in batch) / len(batch)
        avg_r = sum(b[1] for b in batch) / len(batch)
        print(f"  RL epoch {ep} succ={succ:.2f} cost={avg_c:.3f} reward={avg_r:.3f} loss={total_loss.item()/max(n,1):.4f}")
    return policy