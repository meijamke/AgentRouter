"""90 任务全量训练（真执行标签 + save_state 重置 + 检查点断点续跑）。

三阶段：① 采集标签（慢，检查点）② IL 训练 ③ cost-aware RL。
"""
import json
import os
import time

import torch

os.chdir(r"D:\dsh_space\跨设备模型路由代码")
import sys
sys.path.insert(0, os.getcwd())

from src.device_env import CAP_EMBED, TIERS, execute_device, make_device_pool
from src.policy import Policy
from src.schema import StepState
from src.state_encoder import state_to_features
from src.appworld_env import run_episode_grad
from appworld import AppWorld, load_task_ids

CHECKPOINT = "il_labels.json"


def build_prompt(world):
    return (
        f"任务：{world.task.instruction}\n"
        f"可用 App：{str(world.task.app_descriptions)}\n"
        f"API 文档（节选）：{str(world.task.api_docs)[:3000]}\n"
        "请生成 Python 代码完成任务，只输出代码："
    )


def il_label(task_id: str) -> int:
    """真执行标签：一个环境 save_state，每设备 execute 后 load_state 重置。"""
    world = AppWorld(task_id)
    prompt = build_prompt(world)
    init_id = world.save_state()
    for tier in TIERS:                          # weak → medium → strong
        code, _, _ = execute_device(tier, [{"role": "user", "content": prompt}], mock=False)
        try:
            world.execute(code.strip())
            success = bool(world.evaluate().success)
        except Exception:
            success = False
        world.load_state(init_id)               # 重置到初始状态
        if success:
            world.close()
            return TIERS.index(tier)
    world.close()
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
            labels[tid] = il_label(tid)
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


def train_rl(policy, task_ids, devices, epochs=3, lam=0.3):
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    for ep in range(epochs):
        batch = []
        for tid in task_ids[:10]:   # RL 先用前 10 个任务（真执行慢）
            from src.appworld_env import AppworldEnv
            env = AppworldEnv(tid)
            lps, acts, success, cost, steps = run_episode_grad(policy, env, devices, max_steps=4)
            reward = (1.0 if success else 0.0) - lam * cost
            batch.append((lps, reward, success, cost))
            env.close()
        base = sum(b[1] for b in batch) / len(batch)
        total_loss = 0.0; n = 0
        for lps, reward, success, cost in batch:
            G = reward - base
            for lp in lps:
                total_loss = total_loss - lp * G; n += 1
        opt.zero_grad(); total_loss.backward(); opt.step()
        succ = sum(1 for b in batch if b[2]) / len(batch)
        avg_c = sum(b[3] for b in batch) / len(batch)
        print(f"RL epoch {ep} succ={succ:.2f} cost={avg_c:.3f} loss={total_loss.item()/max(n,1):.4f}", flush=True)
    torch.save(policy.state_dict(), "policy_rl.pt")
    return policy


if __name__ == "__main__":
    torch.manual_seed(0)
    ids = load_task_ids("train")   # 90 任务
    devices = make_device_pool()
    policy = Policy()

    print(f"=== ① 采集真执行标签（{len(ids)} 任务）===", flush=True)
    labels = collect_labels(ids)

    print("=== ② IL 训练 ===", flush=True)
    train_il(policy, labels, devices, epochs=5)

    print("=== ③ cost-aware RL ===", flush=True)
    train_rl(policy, ids, devices, epochs=3)

    print("全量训练完成", flush=True)