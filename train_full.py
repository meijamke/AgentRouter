"""90 任务全量训练（方案 B：IL 静态标签 + RL 子进程真执行）。

IL 冷启动 = 静态标签（代码不含禁用模块），便宜稳定（不 execute，不撞 env 生命周期 bug）。
RL = 真执行，每 episode 独立子进程（rl_worker.py，绕开多 env 冲突）。
"""
import json
import os
import subprocess
import sys
import time

import torch

os.chdir(r"D:\dsh_space\跨设备模型路由代码")
sys.path.insert(0, os.getcwd())

from src.device_env import CAP_EMBED, TIERS, execute_device, make_device_pool
from src.policy import Policy
from src.schema import StepState
from src.state_encoder import state_to_features
from appworld import AppWorld, load_task_ids

CHECKPOINT = "il_labels.json"
FORBIDDEN = ["import requests", "import os", "import subprocess", "import socket",
             "import urllib", "import http", "import sys", "open("]


def build_prompt(world):
    return (
        f"任务：{world.task.instruction}\n"
        f"可用 App：{str(world.task.app_descriptions)}\n"
        f"API 文档（节选）：{str(world.task.api_docs)[:3000]}\n"
        "请生成 Python 代码完成任务，只输出代码："
    )


def static_label(task_id: str) -> int:
    """静态标签：最便宜且代码不含禁用模块（会正确用 App API）。"""
    world = AppWorld(task_id)
    prompt = build_prompt(world)
    world.close()
    for tier in TIERS:                          # weak → medium → strong
        code, _, _ = execute_device(tier, [{"role": "user", "content": prompt}], mock=False)
        if not any(f in code for f in FORBIDDEN):
            return TIERS.index(tier)
    return TIERS.index("strong")


def collect_labels(task_ids):
    labels = {}
    if os.path.exists(CHECKPOINT):
        labels = json.load(open(CHECKPOINT, encoding="utf-8"))
        print(f"从检查点恢复 {len(labels)} 个标签", flush=True)
    for i, tid in enumerate(task_ids):
        if tid in labels:
            continue
        t0 = time.time()
        try:
            labels[tid] = static_label(tid)
            print(f"[{i+1}/{len(task_ids)}] {tid} -> {TIERS[labels[tid]]}  ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            labels[tid] = TIERS.index("strong")
            print(f"[{i+1}/{len(task_ids)}] {tid} FAILED -> strong  ({str(e)[:80]})", flush=True)
        json.dump(labels, open(CHECKPOINT, "w", encoding="utf-8"), ensure_ascii=False)
    return labels


def train_il(policy, labels, devices, epochs=5):
    import torch.nn as nn
    data = []
    for tid, label in labels.items():
        s = StepState(trajectory_summary=[], progress=[0.0, 0.5], last_output="",
                      p_fail=0.0, cost_accum=0.0, device_states=devices, step_idx=0)
        data.append((s, label))
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
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
        print(f"IL epoch {ep} loss={tot/n:.4f}", flush=True)
    torch.save(policy.state_dict(), "policy_il.pt")
    return policy


def train_rl(policy, task_ids, devices, epochs=3, lam=0.3, n_tasks=5):
    """子进程 rollout：每个 episode 独立子进程跑真执行，返回 (feats, actions, success, cost)。"""
    dev_emb = torch.tensor([CAP_EMBED[t] for t in TIERS], dtype=torch.float32).unsqueeze(0)
    mask = torch.ones(1, len(TIERS))
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    for ep in range(epochs):
        torch.save(policy.state_dict(), "policy_tmp.pt")
        batch = []
        for tid in task_ids[:n_tasks]:
            out_file = f"rl_out_{tid}.json"
            subprocess.run([sys.executable, "rl_worker.py", tid, "policy_tmp.pt", out_file],
                           check=True, cwd=os.getcwd())
            r = json.load(open(out_file, encoding="utf-8"))
            batch.append((r["feats"], r["actions"], r["success"], r["cost"]))

        rewards = [(1.0 if s else 0.0) - lam * c for _, _, s, c in batch]
        base = sum(rewards) / len(rewards)
        total_loss = 0.0; n = 0
        for (feats, acts, s, c), reward in zip(batch, rewards):
            G = reward - base
            for f, a in zip(feats, acts):
                feat = torch.tensor(f).unsqueeze(0)
                logits, _ = policy(feat, dev_emb, mask, torch.tensor([[0.0]]))
                dist = torch.distributions.Categorical(logits=logits)
                total_loss = total_loss - dist.log_prob(torch.tensor([a])) * G
                n += 1
        opt.zero_grad(); total_loss.backward(); opt.step()
        succ = sum(1 for b in batch if b[2]) / len(batch)
        avg_c = sum(b[3] for b in batch) / len(batch)
        print(f"RL epoch {ep} succ={succ:.2f} cost={avg_c:.3f} loss={total_loss.item()/max(n,1):.4f}", flush=True)
    torch.save(policy.state_dict(), "policy_rl.pt")
    return policy


if __name__ == "__main__":
    torch.manual_seed(0)
    ids = load_task_ids("train")
    devices = make_device_pool()
    policy = Policy()

    print(f"=== ① IL 静态标签（{len(ids)} 任务）===", flush=True)
    labels = collect_labels(ids)
    print("标签分布:", {t: list(labels.values()).count(i) for i, t in enumerate(TIERS)}, flush=True)

    print("=== ② IL 训练 ===", flush=True)
    train_il(policy, labels, devices, epochs=5)

    print("=== ③ RL（子进程真执行，前 5 任务）===", flush=True)
    train_rl(policy, ids, devices, epochs=3, n_tasks=5)

    print("全量训练完成", flush=True)